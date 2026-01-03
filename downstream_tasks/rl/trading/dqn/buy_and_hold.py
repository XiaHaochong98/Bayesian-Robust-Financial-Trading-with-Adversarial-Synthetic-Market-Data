import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import argparse
from mmengine.config import Config, DictAction
from copy import deepcopy
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import gym
import torch.optim as optim
import time
from torch.nn import functional as F
import json
import json5

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)

from downstream_tasks.dataset import AugmentatedDatasetStocks as Dataset_Stocks
from environment import EnvironmentRET
from wrapper import make_env
from policy import Agent
from module.metrics import ARR, SR, CR, SOR, MDD, VOL
from buffers import ReplayBuffer


def save_json(json_dict, file_path, indent=4):
    with open(file_path, mode='w', encoding='utf8') as fp:
        try:
            if indent == -1:
                json.dump(json_dict, fp, ensure_ascii=False)
            else:
                json.dump(json_dict, fp, ensure_ascii=False, indent=indent)
        except Exception as e:
            if indent == -1:
                json5.dump(json_dict, fp, ensure_ascii=False)
            else:
                json5.dump(json_dict, fp, ensure_ascii=False, indent=indent)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def update_data_root(cfg, root):
    cfg.root = root
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = root


def build_storage(shape, type, device):
    if type.startswith("int32"):
        type = torch.int32
    elif type.startswith("float32"):
        type = torch.float32
    elif type.startswith("int64"):
        type = torch.int64
    elif type.startswith("bool"):
        type = torch.bool
    else:
        type = torch.float32
    return torch.zeros(shape, dtype=type, device=device)


def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--config", default=os.path.join(CURRENT, "configs", "SPY.py"), help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--if_remove", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    cfg.merge_from_dict(args.cfg_options)

    update_data_root(cfg, root=args.root)
    assert cfg.tag == 'buy_and_hold'

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    if args.if_remove is None:
        args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
    if args.if_remove:
        import shutil
        shutil.rmtree(exp_path, ignore_errors=True)
        print(f"| Arguments Remove work_dir: {exp_path}")
    else:
        print(f"| Arguments Keep work_dir: {exp_path}")
    os.makedirs(exp_path, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset_Stocks(root_path=ROOT,
                             data_path=cfg.dataset.data_path,
                             train_stock_ticker=cfg.select_stock,
                             test_stock_ticker=cfg.select_stock,
                             features_name=cfg.dataset.features_name,
                             temporals_name=cfg.dataset.temporals_name,
                             target=cfg.dataset.labels_name,
                             flag="RL")

    train_env = EnvironmentRET(
        mode="train",
        dataset=dataset,
        select_stock=cfg.select_stock,
        timestamps=cfg.env.timestamps,
        if_norm=cfg.env.if_norm,
        if_norm_temporal=cfg.env.if_norm_temporal,
        scaler=cfg.env.scaler,
        start_date=cfg.train_start_date,
        end_date=cfg.train_end_date,
        initial_amount=cfg.env.initial_amount,
        transaction_cost_pct=cfg.env.transaction_cost_pct,
        level=cfg.level
    )
    train_env.mode = "val"

    val_env = EnvironmentRET(
        mode="val",
        dataset=dataset,
        select_stock=cfg.select_stock,
        timestamps=cfg.env.timestamps,
        if_norm=cfg.env.if_norm,
          if_norm_temporal=cfg.env.if_norm_temporal,
        scaler=train_env.scaler,
        start_date=cfg.valid_start_date,
        end_date=cfg.valid_end_date,
        initial_amount=cfg.env.initial_amount,
        transaction_cost_pct=cfg.env.transaction_cost_pct,
        level=cfg.level
    )

    test_env = EnvironmentRET(
        mode="test",
        dataset=dataset,
        select_stock=cfg.select_stock,
        timestamps=cfg.env.timestamps,
        if_norm=cfg.env.if_norm,
        if_norm_temporal=cfg.env.if_norm_temporal,
        scaler=train_env.scaler,
        start_date=cfg.test_start_date,
        end_date=cfg.test_end_date,
        initial_amount=cfg.env.initial_amount,
        transaction_cost_pct=cfg.env.transaction_cost_pct,
        level=cfg.level
    )

    train_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(train_env),
                                               transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
        range(1)
    ])
    valid_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(val_env),
                                               transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
        range(1)
    ])
    test_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(test_env),
                                               transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
        range(1)
    ])

    buy_and_hold(cfg, train_envs, exp_path, name='train')
    buy_and_hold(cfg, valid_envs, exp_path, name='valid')
    buy_and_hold(cfg, test_envs, exp_path, name='test')

    train_envs.close()
    valid_envs.close()
    test_envs.close()   

end_date = ["2017-12-27", "2020-12-29", "2023-12-27"]
def buy_and_hold(cfg, envs, exp_path, name='valid'):
    rets = []
    trading_records = {
        "timestamp": [],
        "value": [],
        "cash": [],
        "position": [],
        "ret": [],
        "price": [],
        "discount": [],
        "total_profit": [],
        "total_return": [],
        "action": [],
    }

    # TRY NOT TO MODIFY: start the game
    _, info = envs.reset()
    rets.append(info["ret"])

    # cnt = 0
    while True:
        

        # TRY NOT TO MODIFY: execute the game and log data.
        # cnt = cnt + 1
        # t = action.cpu().numpy()
        # print (666, t)
        actions = []
        for env in envs.envs:
            if env.timestamp_index + 1 < env.num_timestamps - 1:
                actions.append(2)
            else:
                actions.append(1)
        next_obs, reward, done, truncted, info = envs.step(actions)
        record_info = info
        if 'final_info' in info:
            record_info = {key: np.array([item[key] for item in info['final_info']]) for key in info['final_info'][0].keys()}
        rets.append(record_info["ret"])
        trading_records["timestamp"].append(record_info["timestamp"])
        trading_records["value"].append(record_info["value"])
        trading_records["cash"].append(record_info["cash"])
        trading_records["position"].append(record_info["position"])
        trading_records["ret"].append(record_info["ret"])
        trading_records["price"].append(record_info["price"])
        trading_records["discount"].append(record_info["discount"])
        trading_records["total_profit"].append(record_info["total_profit"])
        trading_records["total_return"].append(record_info["total_return"])
        trading_records["action"].append(actions)

        if trading_records["action"][-1] != record_info["action"]:
            trading_records["action"][-1] = record_info["action"]

        if "final_info" in info:
            break

    rets = np.array(rets)
    arr = ARR(rets)
    sr = SR(rets)
    dd = MDD(rets)
    mdd = MDD(rets)
    cr = CR(rets, mdd=mdd)
    sor = SOR(rets, dd=dd)
    vol = VOL(rets)

    for key in trading_records.keys():
        trading_records[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in trading_records[key]]

    save_json(trading_records, os.path.join(exp_path, f"{name}_records.json"))

if __name__ == '__main__':
    main()

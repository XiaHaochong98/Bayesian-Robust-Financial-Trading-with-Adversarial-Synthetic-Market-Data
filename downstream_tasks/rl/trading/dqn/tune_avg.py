import os

import optuna

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

from downstream_tasks.dataset import AugmentatedDatasetStocks as Dataset
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
    parser.add_argument("--config", default=os.path.join(CURRENT, "configs", "AAPL.py"), help="config file path")
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


# def validate_agent(cfg, agent, envs, writer, device, global_step, exp_path):
def validate_agent(cfg, agent, envs, device, global_step):
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
    state, info = envs.reset()
    rets.append(info["ret"])

    next_obs = torch.Tensor(state).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    while True:
        obs = next_obs
        dones = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            q_values = agent.target_network(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values, dim=1)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, truncted, info = envs.step(action.cpu().numpy())
        rets.append(info["ret"])
        trading_records["timestamp"].append(info["timestamp"])
        trading_records["value"].append(info["value"])
        trading_records["cash"].append(info["cash"])
        trading_records["position"].append(info["position"])
        trading_records["ret"].append(info["ret"])
        trading_records["price"].append(info["price"])
        trading_records["discount"].append(info["discount"])
        trading_records["total_profit"].append(info["total_profit"])
        trading_records["total_return"].append(info["total_return"])
        trading_records["action"].append(action.cpu().numpy())

        if trading_records["action"][-1] != info["action"]:
            trading_records["action"][-1] = info["action"]

        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor(done).to(device)

        if "final_info" in info:
            print("val final_info", info["final_info"])
            for info_item in info["final_info"]:
                if info_item is not None:
                    print(
                        f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                    # writer.add_scalar("val/total_return", info_item["total_return"], global_step)
                    # writer.add_scalar("val/total_profit", info_item["total_profit"], global_step)
            break

    rets = np.array(rets)
    arr = ARR(rets)
    sr = SR(rets)
    dd = MDD(rets)
    mdd = MDD(rets)
    cr = CR(rets, mdd=mdd)
    sor = SOR(rets, dd=dd)
    vol = VOL(rets)

    # writer.add_scalar("val/ARR", arr, global_step)
    # writer.add_scalar("val/SR", sr, global_step)
    # writer.add_scalar("val/CR", cr, global_step)
    # writer.add_scalar("val/SOR", sor, global_step)
    # writer.add_scalar("val/DD", dd, global_step)
    # writer.add_scalar("val/MDD", mdd, global_step)
    # writer.add_scalar("val/VOL", vol, global_step)

    # save_json(trading_records, os.path.join(exp_path, "valid_records.json"))

    return arr


def train(cfg, train_env, train_envs, val_env, val_envs, policy_learning_rate, device, exp_path):
    agent = Agent(input_dim=cfg.num_features,
                  timestamps=cfg.timestamps,
                  embed_dim=cfg.embed_dim,
                  depth=cfg.depth,
                  cls_embed=False,
                  action_dim=train_env.action_dim,
                  temporals_name=cfg.temporals_name,
                  device=device)

    # optimizer = optim.Adam(agent.q_network.parameters(), lr=cfg.policy_learning_rate)
    optimizer = optim.Adam(agent.q_network.parameters(), lr=policy_learning_rate)

    rb = ReplayBuffer(
        cfg.buffer_size,
        transition=cfg.transition,
        transition_shape=cfg.transition_shape,
        device=device,
    )

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = train_envs.reset()
    pre_global_step = 0

    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([train_envs.single_action_space.sample() for _ in range(cfg.num_envs)])
        else:
            q_values = agent.q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, info = train_envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in info:
            print("final_info", info["final_info"])
            for info_item in info["final_info"]:
                if info_item is not None:
                    print(
                        f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                    # writer.add_scalar("charts/total_return", info_item["total_return"], global_step)
                    # writer.add_scalar("charts/total_profit", info_item["total_profit"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = info["final_observation"][idx]

        rb.update((obs, actions, rewards, terminations, real_next_obs))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                states, actions, rewards, dones, next_states = rb.sample(cfg.batch_size)
                with torch.no_grad():
                    target_max, _ = agent.target_network(next_states).max(dim=1)
                    td_target = rewards.flatten() + cfg.gamma * target_max * (1 - dones.flatten())
                old_val = agent.q_network(states).gather(1, actions.unsqueeze(1).long()).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    # writer.add_scalar("losses/td_loss", loss, global_step)
                    # writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(agent.target_network.parameters(),
                                                                 agent.q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )

            os.makedirs(os.path.join(exp_path, cfg.save_path), exist_ok=True)
            if global_step // cfg.check_steps != pre_global_step // cfg.check_steps:
                validate_agent(cfg, agent, val_envs, device, global_step)
                torch.save(agent.state_dict(),
                           os.path.join(exp_path, cfg.save_path, "{}.pth".format(global_step // cfg.check_steps)))

            pre_global_step = global_step

    validate_agent(cfg, agent, val_envs, device, global_step)
    torch.save(agent.state_dict(),
               os.path.join(exp_path, cfg.save_path, "{}.pth".format(global_step // cfg.check_steps + 1)))

    train_envs.close()
    val_envs.close()

    return agent, global_step


def create_environments(cfg, stock, args):
    work_dir = "downstream_tasks/rl/optuna/workdir/exp02/" + stock + "/dqn"  # TODO

    exp_path = os.path.join(cfg.root, work_dir, cfg.tag)

    if args.if_remove is None:
        args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
    if args.if_remove:
        import shutil
        shutil.rmtree(exp_path, ignore_errors=True)
        print(f"| Arguments Remove work_dir: {exp_path}")
    else:
        print(f"| Arguments Keep work_dir: {exp_path}")
    os.makedirs(exp_path, exist_ok=True)

    cfg.dump(os.path.join(exp_path, "config.py"))

    print(f"| Arguments config: {args.config}")

    dataset = Dataset(root=ROOT,
                      data_path=cfg.dataset.data_path,
                      stocks_path=cfg.dataset.stocks_path,
                      features_name=cfg.dataset.features_name,
                      temporals_name=cfg.dataset.temporals_name,
                      labels_name=cfg.dataset.labels_name)

    train_env = EnvironmentRET(
        mode="train",
        dataset=dataset,
        select_stock=stock,  # cfg.select_stock,
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

    val_env = EnvironmentRET(
        mode="val",
        dataset=dataset,
        select_stock=stock,  # cfg.select_stock,
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

    train_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(train_env),
                                               transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
        range(cfg.num_envs)
    ])
    val_envs = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(val_env),
                                               transition_shape=cfg.transition_shape, seed=cfg.seed + i)) for i in
        range(1)
    ])

    return train_env, train_envs, val_env, val_envs, exp_path


def tune():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    cfg.merge_from_dict(args.cfg_options)

    update_data_root(cfg, root=args.root)

    # writer = SummaryWriter(exp_path)
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(configs).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        # params = {
        #     "lr": trial.suggest
        # }

        # update cfg with params
        # TODO find best value for policy_learning_rate 
        lr = trial.suggest_float('lr', 1e-6, 2.5e-4, log=True)

        total_validation_score = 0
        stock_list = ["AAPL", "AMZN", "MSFT", "GOOGL", "TSLA"]

        for stock in stock_list:
            train_env, train_envs, val_env, val_envs, exp_path = create_environments(cfg, stock, args)
            agent, global_step = train(cfg, train_env, train_envs, val_env, val_envs, lr, device, exp_path)
            arr = validate_agent(cfg, agent, val_envs, device, global_step)  # validation score
            total_validation_score += arr

        avg_validation_score = total_validation_score / len(stock_list)
        return avg_validation_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    print("Number of finished trials: ", len(study.trials))
    print('BEST TRAIL: ', study.best_trial.params)


if __name__ == "__main__":
    tune()

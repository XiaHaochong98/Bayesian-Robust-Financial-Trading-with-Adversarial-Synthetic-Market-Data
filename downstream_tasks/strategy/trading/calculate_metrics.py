import os
import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import argparse
from mmengine.config import Config, DictAction
from copy import deepcopy
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
import json
import json5
import datetime
ROOT = str(Path(__file__).resolve().parents[3])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)

from downstream_tasks.dataset import AugmentatedDatasetStocks as Dataset
from environment import EnvironmentRET
from policy import Agent
from module.metrics import ARR, SR, CR, SOR, MDD, VOL, DD
from module.plots.trading import plot_trading
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def update_data_root(cfg, root):
    cfg.root = root
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = root

# define the parameter space of each algorithm
strategy_paramters={
    "1":{'short_window': 7, 'long_window': 14},
    "2":{"ilong" : 9, "isig" : 3, "rsiPeriod" : 14, "rsiOverbought" : 60, "rsiOversold" : 40, "useRsiFilter" : True},
    "3":{"lookback":14, "sma_period":20, "std_dev":2.0, "overbought":80, "oversold":20},
    "4":{"lookback":20, "z_score_threshold":1.0},
    "5":{"atr_length":7, "atr_multiplier":2.0, "len_volat":7, "len_drift":7,"multiple_std":1.0}
}


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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    cfg.merge_from_dict(args.cfg_options)

    update_data_root(cfg, root=args.root)

    for strategy_number in cfg.strategy_number:

        params_save_path = os.path.join(cfg.root, cfg.resdir, cfg.select_stock, str(strategy_number), cfg.tag,
                                        "best_params.json")
        # create the folder to save the best params if not exist
        if not os.path.exists(os.path.dirname(params_save_path)):
            os.makedirs(os.path.dirname(params_save_path))
        exp_path = os.path.join(cfg.root, cfg.workdir,cfg.select_stock,str(strategy_number),cfg.tag)
        if args.if_remove is None:
            args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
        if args.if_remove:
            import shutil
            shutil.rmtree(exp_path, ignore_errors=True)
            print(f"| Arguments Remove work_dir: {exp_path}")
        else:
            print(f"| Arguments Keep work_dir: {exp_path}")
        os.makedirs(exp_path, exist_ok=True)

        valid_save_path = os.path.join(exp_path, "valid_records.json")
        with open(valid_save_path, "r") as f:
            trading_records = json.load(f)
            rets = trading_records["ret"]

        rets = np.array(rets)
        arr = ARR(rets)
        sr = SR(rets)
        dd = DD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)

        trading_records["summary"]["ARR"] = arr
        trading_records["summary"]["SR"] = sr
        trading_records["summary"]["CR"] = cr
        trading_records["summary"]["SOR"] = sor
        trading_records["summary"]["DD"] = dd
        trading_records["summary"]["MDD"] = mdd
        trading_records["summary"]["VOL"] = vol

        print("ARR%, SR, CR, SOR, MDD%, VOL: {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}".format(arr * 100, sr,cr, sor, mdd * 100, vol))


if __name__ == '__main__':
    main()

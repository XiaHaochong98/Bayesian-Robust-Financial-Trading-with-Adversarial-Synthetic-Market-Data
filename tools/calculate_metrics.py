import os

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["MKL_DEBUG_CPU_TYPE"] = '5'
import json
import finnhub
import pandas as pd
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
import argparse
import shutil
from mmengine.config import Config, DictAction

from dotenv import load_dotenv
load_dotenv(verbose=True)

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(CURRENT)

from module.utils.misc import update_data_root
from module.metrics import ARR, SR, CR, SOR, MDD, VOL, DD
from module.plots import plot_trading

pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)

def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "exp", "trading_mi_w_low_w_high_w_tool_w_decision", "AAPL.py"), help="config file path")
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

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    if args.if_remove is None:
        args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
    if args.if_remove:
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

    print("ARR%, SR, CR, SOR, MDD%, VOL: {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}".format(arr * 100, sr, cr, sor, mdd* 100, vol))

    # save the trading records back to the valid_records.json
    # add a summary
    trading_records["summary"] = {}
    trading_records["summary"]["ARR"] = arr
    trading_records["summary"]["SR"] = sr
    trading_records["summary"]["CR"] = cr
    trading_records["summary"]["SOR"] = sor
    trading_records["summary"]["DD"] = dd
    trading_records["summary"]["MDD"] = mdd
    trading_records["summary"]["VOL"] = vol
    with open(valid_save_path, "w") as f:
        json.dump(trading_records, f, indent=4)

    echarts_js_path = os.path.join(ROOT, "tools", "echarts-5.4.3", "dist", "echarts.min.js")

    if not os.path.exists(os.path.join(exp_path, "echarts.min.js")):
        shutil.copy(echarts_js_path, exp_path)

    output_path = os.path.join(exp_path, "valid.png")

    plot_trading(trading_records, output_path)

if __name__ == '__main__':
    main()
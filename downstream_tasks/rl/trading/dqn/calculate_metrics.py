import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import argparse
from mmengine.config import Config, DictAction
import numpy as np

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)
import json

from module.metrics import ARR, SR, CR, SOR, MDD, VOL, DD


def update_data_root(cfg, root):
    cfg.root = root
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = root


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

    for stage in ['train', 'valid', 'test']:
        save_path = os.path.join(exp_path, f"{stage}_records.json")
        try: 
            with open(save_path, "r") as f:
                trading_records = json.load(f)
                rets = trading_records["ret"]
        except Exception as e:
            print(f'Records file {save_path} does not exist!')
            continue

        rets = np.array(rets)
        arr = ARR(rets)
        sr = SR(rets)
        dd = DD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)

        print(f"Stage {stage}:")
        print("ARR%, SR, CR, SOR, MDD%, VOL: {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}".format(arr * 100, sr, cr,
                                                                                                        sor, mdd * 100,
                                                                                                        vol))


if __name__ == '__main__':
    main()

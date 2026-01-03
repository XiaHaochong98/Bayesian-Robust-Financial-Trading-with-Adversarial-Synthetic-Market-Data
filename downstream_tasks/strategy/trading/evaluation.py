import os
import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import argparse
from mmengine.config import Config, DictAction
from copy import deepcopy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import json5

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
strategy_paramters = {
    "0": {},
    "1": {'short_window': 5},
    "2": {"ilong": 9, "isig": 3, "rsiOverbought": 60, "rsiOversold": 40},
    "3": {"std_dev": 2.0, "overbought": 80, "oversold": 20},
    "4": {"z_score_threshold": 1.0},
    "5": {"atr_length": 7, "atr_multiplier": 2.0, "len_volat": 7, "len_drift": 7, "multiple_std": 1.0}
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

    # use the configs.train_flag to decide whether to train the agent or not

    for strategy_number in cfg.strategy_number:
        if cfg.train_flag:
            suffix = "trained"
        else:
            suffix = "default"
        params_save_path = os.path.join(cfg.root, cfg.resdir, cfg.select_stock, str(strategy_number), cfg.tag, suffix,
                                        "best_params.json")
        best_result_save_path = os.path.join(cfg.root, cfg.resdir, cfg.select_stock, str(strategy_number), cfg.tag,
                                             suffix,
                                             "best_result.json")
        # create the folder to save the best params if not exist
        if not os.path.exists(os.path.dirname(params_save_path)):
            os.makedirs(os.path.dirname(params_save_path))
        exp_path = os.path.join(cfg.root, cfg.workdir, cfg.select_stock, str(strategy_number), cfg.tag)
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
            dataset=dataset,
            select_stock=cfg.select_stock,
            timestamps=cfg.env.timestamps,
            if_norm=cfg.env.if_norm,
            if_norm_temporal=cfg.env.if_norm_temporal,
            start_date=cfg.train_start_date,
            end_date=cfg.train_end_date,
            initial_amount=cfg.env.initial_amount,
            transaction_cost_pct=cfg.env.transaction_cost_pct,
            level=cfg.level
        )
        val_env = EnvironmentRET(
            dataset=dataset,
            select_stock=cfg.select_stock,
            timestamps=cfg.env.timestamps,
            if_norm=cfg.env.if_norm,
            if_norm_temporal=cfg.env.if_norm_temporal,
            start_date=cfg.valid_start_date,
            end_date=cfg.valid_end_date,
            initial_amount=cfg.env.initial_amount,
            transaction_cost_pct=cfg.env.transaction_cost_pct,
            level=cfg.level
        )
        agent = Agent(strategy_number=strategy_number)

        # train
        params = strategy_paramters[str(strategy_number)]
        if cfg.train_flag:
            global_step = 0
            rets = run_agent(args, cfg, agent, val_env, global_step, exp_path, params,
                             save_records=True)
            arr_with_out_train = ARR(np.array(rets))
            global_step = 0
            sampler = TPESampler(seed=cfg.seed)  # Make the sampler behave in a deterministic way.
            study = optuna.create_study(sampler=sampler)
            study.optimize(lambda trial: objective(trial, args, cfg, agent, train_env, global_step, exp_path, params,
                                                   save_records=False), n_trials=50)
            best_params = study.best_params
            for para in best_params:
                params[para] = best_params[para]
            # save the best params to the res folder
            save_json(params, params_save_path)
            print("ARR with out train: ", arr_with_out_train * 100)
            print("best_params: ", params)
        global_step = 0
        # run the agent in the training environment to record the trading records
        rets = run_agent(args, cfg, agent, train_env, global_step, exp_path, params, save_records=False)
        rets = np.array(rets)
        arr = ARR(rets)
        sr = SR(rets)
        dd = DD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)
        # save these metrics to the best_result_save_path
        best_records = {}
        best_records["ARR"] = arr
        best_records["SR"] = sr
        best_records["CR"] = cr
        best_records["SOR"] = sor
        best_records["DD"] = dd
        best_records["MDD"] = mdd
        best_records["VOL"] = vol
        save_json(best_records, best_result_save_path)

        # run the agent in the validation environment to record the trading records
        global_step = 0
        if cfg.train_flag:
            rets = run_agent(args, cfg, agent, val_env, global_step, exp_path, params,
                             save_records=True, result_without_train=arr_with_out_train)
            arr_val = ARR(np.array(rets))
            print("ARR after training: ", arr_val * 100)
        else:
            rets = run_agent(args, cfg, agent, val_env, global_step, exp_path, params,
                             save_records=True)
            arr_val = ARR(np.array(rets))
            print("ARR without training: ", arr_val * 100)


def objective(trial, args, cfg, agent, train_env, global_step, exp_path, params, save_records=False):
    initial_params = deepcopy(params)
    params = {}
    for para in initial_params.keys():
        # if value of the parameter is int, then use suggest_int
        if type(initial_params[para]) == int:
            lower_bound = int(initial_params[para] * 0.8)
            upper_bound = int(initial_params[para] * 1.2)
            params[para] = trial.suggest_int(para, lower_bound, upper_bound)
        elif type(initial_params[para]) == bool:
            params[para] = trial.suggest_categorical(para, [True, False])
        elif type(initial_params[para]) == float:
            params[para] = trial.suggest_float(para, initial_params[para] * 0.8, initial_params[para] * 1.2)
        else:
            raise ValueError("The type of the parameter is not supported")
    rets = run_agent(args, cfg, agent, train_env, global_step, exp_path, params, save_records=False)
    rets = np.array(rets)
    arr = ARR(rets)
    return arr * 100 * -1


def run_agent(args, cfg, agent, envs, global_step, exp_path, params, save_records=True, result_without_train=None):
    rets = []
    trading_records = {
        "symbol": [],
        "day": [],
        "value": [],
        "cash": [],
        "position": [],
        "ret": [],
        "date": [],
        "price": [],
        "discount": [],
        "kline_path": [],
        "trading_path": [],
        "total_profit": [],
        "total_return": [],
        "action": [],
        "reasoning": [],
        "summary": {},
    }

    state, info = envs.reset()
    rets.append(info["ret"])
    done = False
    first_day = True
    next_obs = state

    while not done:
        obs = next_obs
        params["FirstDay"] = first_day
        trading_records["date"].append(info["timestamp"])
        signals, explanations = agent.decision(obs=obs, params=params)
        # remove "FirstDay" from params
        params.pop("FirstDay")
        action = signals[-1]
        next_obs, reward, done, truncted, info = envs.step(action)
        rets.append(info["ret"])
        trading_records["value"].append(info["value"])
        trading_records["cash"].append(info["cash"])
        trading_records["position"].append(info["position"])
        trading_records["ret"].append(info["ret"])
        trading_records["price"].append(info["price"])
        trading_records["discount"].append(info["discount"])
        trading_records["total_profit"].append(info["total_profit"])
        trading_records["total_return"].append(info["total_return"])
        trading_records["action"].append(action)
        trading_records["symbol"].append(cfg.select_stock)
        first_day = False

    if save_records:
        writer = SummaryWriter(exp_path)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        rets = np.array(rets)
        arr = ARR(rets)
        sr = SR(rets)
        dd = DD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)

        writer.add_scalar("val/ARR", arr, global_step)
        writer.add_scalar("val/SR", sr, global_step)
        writer.add_scalar("val/CR", cr, global_step)
        writer.add_scalar("val/SOR", sor, global_step)
        writer.add_scalar("val/DD", dd, global_step)
        writer.add_scalar("val/MDD", mdd, global_step)
        writer.add_scalar("val/VOL", vol, global_step)

        # add the scalar to the trading_records summary
        trading_records["summary"]["ARR"] = arr
        trading_records["summary"]["SR"] = sr
        trading_records["summary"]["CR"] = cr
        trading_records["summary"]["SOR"] = sor
        trading_records["summary"]["DD"] = dd
        trading_records["summary"]["MDD"] = mdd
        trading_records["summary"]["VOL"] = vol
        if result_without_train is not None:
            trading_records["summary"]["ARR without training"] = result_without_train

        save_json(trading_records, os.path.join(exp_path, "valid_records.json"))

        # process the action in trading records
        # for consecutive same action of "BUY" or "SELL", only keep the first one
        action_list = trading_records["action"]
        past_action = "HOLD"
        for i in range(len(action_list)):
            if action_list[i] == "BUY" or action_list[i] == "SELL":
                if action_list[i] == past_action:
                    action_list[i] = "HOLD"
                else:
                    past_action = action_list[i]

        # plot trading
        plot_path = os.path.join(exp_path, "trading.png")
        print(f"| Plot trading to {plot_path}")
        plot_trading(trading_records, plot_path)
        writer.close()

        print("ARR%, SR, CR, SOR, MDD%, VOL: {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}, {:.04f}".format(arr * 100, sr,
                                                                                                          cr, sor,
                                                                                                          mdd * 100,
                                                                                                          vol))
    return rets


if __name__ == '__main__':
    main()

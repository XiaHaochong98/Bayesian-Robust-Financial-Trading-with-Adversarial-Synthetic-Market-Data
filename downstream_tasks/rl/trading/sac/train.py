import os
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

    cfg.dump(os.path.join(exp_path, "config.py"))

    print(f"| Arguments config: {args.config}")

    writer = SummaryWriter(exp_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
                             test_stock_ticker=cfg.dataset.stocks_path,
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

    agent = Agent(input_dim=cfg.num_features,
                  timestamps=cfg.timestamps,
                  embed_dim=cfg.embed_dim,
                  depth=cfg.depth,
                  cls_embed=False,
                  action_dim=train_env.action_dim,
                  temporals_name=cfg.temporals_name,
                  device=device)

    q_optimizer = optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=cfg.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(agent.actor.parameters()), lr=cfg.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if cfg.autotune:
        target_entropy = -cfg.target_entropy_scale * torch.log(1 / torch.tensor(train_envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=cfg.q_lr, eps=1e-4)
    else:
        alpha = cfg.alpha

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
        if global_step < cfg.learning_starts:
            actions = np.array([train_envs.single_action_space.sample() for _ in range(cfg.num_envs)])
        else:
            actions, _, _ = agent.actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = train_envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            print("final_info", infos["final_info"])
            for info_item in infos["final_info"]:
                if info_item is not None:
                    print(
                        f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                    writer.add_scalar("charts/total_return", info_item["total_return"], global_step)
                    writer.add_scalar("charts/total_profit", info_item["total_profit"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.update((obs, actions, rewards, terminations, real_next_obs))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.update_frequency == 0:
                states, actions, rewards, dones, next_states = rb.sample(cfg.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = agent.actor.get_action(next_states)
                    qf1_next_target = agent.q1_target(next_states)
                    qf2_next_target = agent.q2_target(next_states)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = rewards.flatten() + (1 - dones.flatten()) * cfg.gamma * (
                        min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = agent.q1(states)
                qf2_values = agent.q2(states)
                qf1_a_values = qf1_values.gather(1, actions.unsqueeze(1).long()).view(-1)
                qf2_a_values = qf2_values.gather(1, actions.unsqueeze(1).long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = agent.actor.get_action(states)
                with torch.no_grad():
                    qf1_values = agent.q1(states)
                    qf2_values = agent.q2(states)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if cfg.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (
                            action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % cfg.target_network_frequency == 0:
                for param, target_param in zip(agent.q1.parameters(), agent.q1_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
                for param, target_param in zip(agent.q2.parameters(), agent.q2_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if cfg.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            os.makedirs(os.path.join(exp_path, cfg.save_path), exist_ok=True)
            if global_step // cfg.check_steps != pre_global_step // cfg.check_steps:
                validate_agent(cfg, agent, val_envs, writer, device, global_step, exp_path)
                torch.save(agent.state_dict(),
                           os.path.join(exp_path, cfg.save_path, "{}.pth".format(global_step // cfg.check_steps)))

            pre_global_step = global_step

    validate_agent(cfg, agent, val_envs, writer, device, global_step, exp_path)
    torch.save(agent.state_dict(),
               os.path.join(exp_path, cfg.save_path, "{}.pth".format(global_step // cfg.check_steps + 1)))

    train_envs.close()
    val_envs.close()
    writer.close()


def validate_agent(cfg, agent, envs, writer, device, global_step, exp_path):
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
            logits = agent.actor(torch.Tensor(obs).to(device))
            action = torch.argmax(logits, dim=1)

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
                    writer.add_scalar("val/total_return", info_item["total_return"], global_step)
                    writer.add_scalar("val/total_profit", info_item["total_profit"], global_step)
            break

    rets = np.array(rets)
    arr = ARR(rets)
    sr = SR(rets)
    dd = MDD(rets)
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

    for key in trading_records.keys():
        trading_records[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in trading_records[key]]

    save_json(trading_records, os.path.join(exp_path, "valid_records.json"))


if __name__ == '__main__':
    main()

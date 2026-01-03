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
from torch import nn
import numpy as np
import random
import gym
import torch.optim as optim
import time
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
            logits = agent.actor(next_obs)
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


def train(cfg, train_env, train_envs, val_env, val_envs, policy_learning_rate, value_learning_rate, device, exp_path):
    agent = Agent(input_dim=cfg.num_features,
                  timestamps=cfg.timestamps,
                  embed_dim=cfg.embed_dim,
                  depth=cfg.depth,
                  cls_embed=False,
                  action_dim=train_env.action_dim,
                  temporals_name=cfg.temporals_name,
                  device=device)

    policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, list(agent.actor.parameters())),
                                   lr=policy_learning_rate, eps=1e-5, weight_decay=0)
    value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.critic.parameters())),
                                 lr=value_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    transition_shape = cfg.transition_shape
    obs = build_storage(shape=(cfg.num_steps, *transition_shape["states"]["shape"]),
                        type=transition_shape["states"]["type"], device=device)
    actions = build_storage(shape=(cfg.num_steps, *transition_shape["actions"]["shape"]),
                            type=transition_shape["actions"]["type"], device=device)
    logprobs = build_storage(shape=(cfg.num_steps, *transition_shape["logprobs"]["shape"]),
                             type=transition_shape["logprobs"]["type"], device=device)
    rewards = build_storage(shape=(cfg.num_steps, *transition_shape["rewards"]["shape"]),
                            type=transition_shape["rewards"]["type"], device=device)
    dones = build_storage(shape=(cfg.num_steps, *transition_shape["dones"]["shape"]),
                          type=transition_shape["dones"]["type"], device=device)
    values = build_storage(shape=(cfg.num_steps, *transition_shape["values"]["shape"]),
                           type=transition_shape["values"]["type"], device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()

    state, info = train_envs.reset()

    next_obs = torch.Tensor(state).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)
    num_updates = cfg.total_timesteps // cfg.batch_size
    num_critic_warm_up_updates = cfg.critic_warm_up_steps // cfg.batch_size

    is_warmup = True
    for update in range(1, num_updates + 1 + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False

        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr and not is_warmup:
            frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
            policy_optimizer.param_groups[0]["lr"] = frac * policy_learning_rate
            value_optimizer.param_groups[0]["lr"] = frac * value_learning_rate

        for step in range(0, cfg.num_steps):
            global_step += 1 * cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, truncted, info = train_envs.step(action.cpu().numpy())
            # print(info)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            if "final_info" in info:
                print("final_info", info["final_info"])
                for info_item in info["final_info"]:
                    if info_item is not None:
                        print(
                            f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                        # writer.add_scalar("charts/total_return", info_item["total_return"], global_step)
                        # writer.add_scalar("charts/total_profit", info_item["total_profit"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + transition_shape["states"]["shape"][1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.view(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        """
        print("b_obs", b_obs.shape)
        print("b_logprobs", b_logprobs.shape)
        print("b_actions", b_actions.shape)
        print("b_advantages", b_advantages.shape)
        print("b_returns", b_returns.shape)
        print("b_values", b_values.shape)
        b_obs torch.Size([512, 30, 153])
        b_logprobs torch.Size([512])
        b_actions torch.Size([512])
        b_advantages torch.Size([512])
        b_returns torch.Size([512])
        b_values torch.Size([512])
        """

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        pg_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        old_approx_kl = torch.tensor(0)
        approx_kl = torch.tensor(0)
        total_approx_kl = torch.tensor(0)

        for epoch in range(cfg.update_epochs):
            if kl_explode:
                break
            # update value
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.value_minibatch_size):
                end = start + cfg.value_minibatch_size
                mb_inds = b_inds[start:end]
                newvalue = agent.get_value(b_obs[mb_inds])

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * cfg.vf_coef

                value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                value_optimizer.step()

            if is_warmup:
                continue

            policy_optimizer.zero_grad()
            # update policy
            for start in range(0, cfg.batch_size, cfg.policy_minibatch_size):
                if policy_update_steps % cfg.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + cfg.policy_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / cfg.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss
                loss /= cfg.gradient_checkpointing_steps

                loss.backward()

                if policy_update_steps % cfg.gradient_checkpointing_steps == 0:
                    if cfg.target_kl is not None:
                        if total_approx_kl > cfg.target_kl:
                            policy_optimizer.zero_grad()
                            kl_explode = True
                            policy_update_steps -= cfg.gradient_checkpointing_steps
                            # print("break", policy_update_steps)
                            break

                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    policy_optimizer.step()
                    policy_optimizer.zero_grad()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/policy_learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("charts/value_learning_rate", value_optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/total_approx_kl", total_approx_kl.item(), global_step)
        # writer.add_scalar("losses/policy_update_times", policy_update_steps // cfg.gradient_checkpointing_steps,
        #                   global_step)
        # writer.add_scalar("losses/clipfrac", num_clipfracs, global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", global_step, (time.time() - start_time))
        # writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

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


def tune():
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

    dataset = Dataset(root=ROOT,
                      data_path=cfg.dataset.data_path,
                      stocks_path=cfg.dataset.stocks_path,
                      features_name=cfg.dataset.features_name,
                      temporals_name=cfg.dataset.temporals_name,
                      labels_name=cfg.dataset.labels_name)

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

    def objective(trial):

        # update cfg with params
        # TODO find best value for policy_learning_rate 
        # params = {
        #     "policy_lr": trial.suggest_float('lr',2.5e-8, 1e-5, log=True),
        #     "value_lr" : trial.suggest_float('lr',5e-8, 1e-4, log=True)   
        # }

        policy_lr = trial.suggest_float('policy_lr', 2.5e-8, 1e-5, log=True)
        value_lr = trial.suggest_float('value_lr', 5e-8, 1e-4, log=True)

        agent, global_step = train(cfg, train_env, train_envs, val_env, val_envs, policy_lr, value_lr, device, exp_path)

        arr = validate_agent(cfg, agent, val_envs, device, global_step)  # validation score

        return arr

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print('BEST TRAIL: ', study.best_trial.params)


if __name__ == "__main__":
    tune()

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
from actor_continuous import ActorContinuous
from module.metrics import ARR, SR, CR, SOR, MDD, VOL
from buffers import ReplayBuffer, ReservoirReplayBuffer
from generator.GRT_GAN.models.API import GeneratorAPI


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


def build_storage(shape, type):
    if type.startswith("int32"):
        type = np.int32
    elif type.startswith("float32"):
        type = np.float32
    elif type.startswith("int64"):
        type = np.int64
    elif type.startswith("bool"):
        type = bool
    else:
        type = np.float32
    return np.zeros(shape, dtype=type)

def data_augmentation_function(data: np.ndarray, cfg, 
                               method: str = 'random', agent: Agent = None, 
                               device=None, adv_agent: ActorContinuous = None, generator=None,timestamp=None,scaler=None):
    epsilon = cfg.epsilon
    data_std = data.std(axis=(0,1))     # data = (num_envs, timestamps, num_features)
    if method == 'random':
        noise = np.random.normal(loc=0.0, scale=epsilon * data_std, size=data.shape)
        return (noise + data, None)
    elif method == 'min_q':
        alpha = cfg.alpha
        iterations = cfg.iterations
        obs_ori = torch.Tensor(data).to(device)
        obs_std = torch.Tensor(data_std).to(device)
        quantile_belief_ori = get_quantile_belief(cfg, obs_ori, agent.quantile_belief_network)
        q_values = agent.q_network(torch.Tensor(data).to(device), quantile_belief_ori)
        actions_ori = torch.argmax(q_values, dim=1)
        
        obs = obs_ori.clone()
        for _ in range(iterations):
            with torch.no_grad():
                quantile_belief = get_quantile_belief(cfg, obs, agent.quantile_belief_network)
            obs.requires_grad_(True)
            q_values = agent.q_network(obs, quantile_belief)
            cost = torch.gather(q_values, -1, actions_ori.unsqueeze(-1)).sum()
            cost.backward()
            noise = -obs.grad.sgn() * alpha * obs_std
            obs = obs_ori + torch.clamp(obs.detach() + noise - obs_ori, -epsilon * obs_std, epsilon * obs_std)
        return (obs.detach().cpu().numpy(), None)
    elif method == 'adv_agent':
        data_tensor = torch.Tensor(data).to(device)
        noise = adv_agent(data_tensor).detach().cpu().numpy()
        return ((data + epsilon * noise * data_std), noise)
    elif method == 'generator_noise':
        # the number of macro features is 46
        noise = np.random.normal(loc=0.0, scale=1, size=46)
        num_envs = data.shape[0]
        new_obs = np.zeros(data.shape)
        # print("data.shape", data.shape)
        for i in range(num_envs):
            generated_data = generator.call(timestamp[i],noise[i])
            # print("generated_data.shape", generated_data.shape)
            # get the last data.shape[1] features
            generated_feature = generated_data[-data.shape[1]:]
            # get the tempral features form data
            generated_feature = np.concatenate((generated_feature,data[i][:,len(cfg.dataset.features_name):]),axis=1)
            # normalize the new_obs
            if cfg.env.if_norm_temporal:
                new_obs[i] = scaler.transform(generated_feature)
            else:
                # only normalize the cfg.dataset.features_name features
                # select the features
                # print("generated_feature.shape", generated_feature.shape)
                generated_feature_norm=scaler.transform(generated_feature[:,:len(cfg.dataset.features_name)])
                new_obs[i] = np.concatenate((generated_feature_norm,generated_feature[:,len(cfg.dataset.features_name):]),axis=1)

        return (new_obs, noise)
    elif method == 'generator_adv_agent':
        data_tensor = torch.Tensor(data).to(device)
        noise = adv_agent(data_tensor).detach().cpu().numpy()
        noise_output = noise.copy()
        if '0.3' in cfg.tag:
            noise = noise * 0.3
        # generate new_obs with generator for each env
        num_envs = data.shape[0]
        new_obs = np.zeros(data.shape)
        for i in range(num_envs):
            generated_data = generator.call(timestamp[i],noise[i][-1])
            # print("generated_data.shape", generated_data.shape)
            # get the last data.shape[1] features
            generated_feature = generated_data[-data.shape[1]:]
            # get the tempral features form data
            generated_feature = np.concatenate((generated_feature,data[i][:,len(cfg.dataset.features_name):]),axis=1)
            # normalize the new_obs
            if cfg.env.if_norm_temporal:
                new_obs[i] = scaler.transform(generated_feature)
            else:
                # only normalize the cfg.dataset.features_name features
                # select the features
                generated_feature_norm=scaler.transform(generated_feature[:,:len(cfg.dataset.features_name)])
                new_obs[i] = np.concatenate((generated_feature_norm,generated_feature[:,len(cfg.dataset.features_name):]),axis=1)

        return (new_obs, noise_output)
    else:
        raise NotImplementedError
    
def compute_values(dones_b, masks_b, values_b, rewards_b, gamma):
    values_b[-1] = rewards_b[-1] * masks_b[-1]
    length = dones_b.shape[0]
    for i in range(length - 2, -1, -1):
        values_b[i] = (values_b[i + 1] * gamma + rewards_b[i] * masks_b[i]) * (1 - dones_b[i])
    values_b = (values_b - values_b.mean()) / (1e-6 + values_b.std())
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--config", default=os.path.join(CURRENT, "configs", "CORN.py"), help="config file path")
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
    
    quantile_heads_num = 0
    if cfg.use_quantile_belief:
        quantile_heads_num = len(cfg.quantile_heads)

    agent = Agent(input_dim=cfg.num_features,
                  timestamps=cfg.timestamps,
                  embed_dim=cfg.embed_dim,
                  depth=cfg.depth,
                  cls_embed=False,
                  action_dim=train_env.action_dim,
                  temporals_name=cfg.temporals_name,
                  device=device,
                  use_quantile_belief=cfg.use_quantile_belief,
                  quantile_heads_num=quantile_heads_num,
                  use_nfsp=cfg.use_nfsp)

    optimizer = optim.Adam(agent.q_network.parameters(), lr=cfg.policy_learning_rate)
    if cfg.use_quantile_belief:
        belief_optimizer = optim.Adam(agent.quantile_belief_network.parameters(), 
                                      lr=cfg.policy_learning_rate)
    
    if cfg.use_data_augmentation and cfg.augmentation_method == 'adv_agent':
        adv_agent = ActorContinuous(input_dim=cfg.num_features, 
                                    timestamps=cfg.timestamps,
                                    embed_dim=cfg.embed_dim,
                                    depth=cfg.depth,
                                    cls_embed=False,
                                    output_dim=cfg.num_features,
                                    temporals_name=cfg.temporals_name,
                                    device=device)
        adv_agent_optimizer = optim.Adam(adv_agent.parameters(), 
                                         lr=cfg.adv_policy_learning_rate)
    elif cfg.use_data_augmentation and cfg.augmentation_method == 'generator_adv_agent':
        # number of macro features is 46
        adv_agent = ActorContinuous(input_dim=cfg.num_features, 
                                    timestamps=cfg.timestamps,
                                    embed_dim=cfg.embed_dim,
                                    depth=cfg.depth,
                                    cls_embed=False,
                                    output_dim=46,
                                    temporals_name=cfg.temporals_name,
                                    device=device)
        adv_agent_optimizer = optim.Adam(adv_agent.parameters(), 
                                         lr=cfg.adv_policy_learning_rate)
    else:
        adv_agent = None
    
    if cfg.use_nfsp:
        nfsp_agent_optimizer = optim.Adam(agent.q_network_nfsp.parameters(),
                                          lr=cfg.policy_learning_rate)

    rb = ReplayBuffer(
        cfg.buffer_size,
        transition=cfg.transition,
        transition_shape=cfg.transition_shape,
        device=device,
    )
    if cfg.use_nfsp:
        nfsp_rb = ReservoirReplayBuffer(
            cfg.buffer_size,
            transition=["states", "actions"],
            transition_shape=cfg.transition_shape,
        )
    
    if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent']:
        transition_shape = cfg.transition_shape
        obs_b = build_storage(shape=(cfg.adv_training_length, *transition_shape["states"]["shape"]),
                              type=transition_shape["states"]["type"])
        if cfg.augmentation_method == 'adv_agent':
            adv_obs_b = build_storage(shape=(cfg.adv_training_length, *transition_shape["states"]["shape"]),
                                    type=transition_shape["states"]["type"])
        else:
            adv_obs_b = build_storage(shape=(cfg.adv_training_length, cfg.num_envs, cfg.timestamps, 46),
                                    type=transition_shape["states"]["type"])
        rewards_b = build_storage(shape=(cfg.adv_training_length, *transition_shape["rewards"]["shape"]),
                                  type=transition_shape["rewards"]["type"])
        # Whether the interaction has been finished
        dones_b = build_storage(shape=(cfg.adv_training_length, *transition_shape["dones"]["shape"]),
                                type=transition_shape["dones"]["type"])
        # Whether the data is valid
        masks_b = build_storage(shape=(cfg.adv_training_length, *transition_shape["dones"]["shape"]),
                                type=transition_shape["dones"]["type"])
        values_b = build_storage(shape=(cfg.adv_training_length, *transition_shape["rewards"]["shape"]),
                                 type=transition_shape["rewards"]["type"])

    # init generator
    if cfg.augmentation_method == 'generator_noise' or cfg.augmentation_method == 'generator_adv_agent':
        model_path="generator/GRT_GAN/output/etf_6_120length"
        ticker_name=cfg.select_stock
        # /data3/hcxia/Adahist2/datasets
        # /data3/hcxia/Adahist2/generator/GRT_GAN/models/API.py
        # Set CUDA_LAUNCH_BLOCKING to 1
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Verify the environment variable is set
        print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Using CUDA: {torch.cuda.is_available()}")
        generator=GeneratorAPI(model_path=model_path, ticker_name=ticker_name,obs_features=cfg.dataset.features_name,temporal_features=cfg.dataset.temporals_name)
    else:
        generator=None

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, info = train_envs.reset()
    # print('info', info)
    timestamp = info["timestamp"]
    # print("timestamp", timestamp)
    pre_global_step = 0
    # pick the scaler of the using ticker
    scalers=train_env.scaler
    ticker_list=train_env.dataset.stocks
    scaler_ticker=scalers[ticker_list.index(cfg.select_stock)]

    for global_step in range(cfg.total_timesteps):
        if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent']:
            obs_b[global_step % cfg.adv_training_length] = obs
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        random_action = random.random() < epsilon
        use_epsilon_greedy = (random.random() < cfg.nfsp_tau if cfg.use_nfsp else True)
        aug_obs_noise = 0
        if (not use_epsilon_greedy) or (not random_action):
            use_augmentation = (random.random() < cfg.augmentation_rate)
            if cfg.use_data_augmentation and use_augmentation:
                original_state = random.getstate()
                obs, aug_obs_noise = data_augmentation_function(obs, cfg, cfg.augmentation_method, agent, device, adv_agent, generator, timestamp=timestamp,scaler=scaler_ticker)
                random.setstate(original_state)

            if cfg.use_quantile_belief:
                quantile_belief = get_quantile_belief(cfg, torch.Tensor(obs).to(device), 
                                                    agent.quantile_belief_network)
            else:
                quantile_belief = None
            
            if not use_epsilon_greedy:
                q_values = agent.q_network_nfsp(torch.Tensor(obs).to(device), quantile_belief)
            else:
                q_values = agent.q_network(torch.Tensor(obs).to(device), quantile_belief)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        else:
            actions = np.array([train_envs.single_action_space.sample() for _ in range(cfg.num_envs)])

        if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent']:
            adv_obs_b[global_step % cfg.adv_training_length] = \
                aug_obs_noise if aug_obs_noise is not None else 0
        if cfg.use_nfsp and use_epsilon_greedy and (not random_action):
            nfsp_rb.update((obs, actions))

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, info = train_envs.step(actions)
        if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent']:
            rewards_b[global_step % cfg.adv_training_length] = rewards
            dones_b[global_step % cfg.adv_training_length] = terminations
            masks_b[global_step % cfg.adv_training_length] = ((not use_epsilon_greedy) or (not random_action)) and use_augmentation

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in info:
            print("final_info", info["final_info"])
            for info_item in info["final_info"]:
                if info_item is not None:
                    print(
                        f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                    writer.add_scalar("charts/total_return", info_item["total_return"], global_step)
                    writer.add_scalar("charts/total_profit", info_item["total_profit"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = info["final_observation"][idx]

        rb.update((obs, actions, rewards, terminations, real_next_obs))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        timestamp = info["timestamp"]

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent'] \
                and (global_step + 1) % cfg.adv_training_length == 0:
                compute_values(dones_b=dones_b, masks_b=masks_b, 
                               values_b=values_b, rewards_b=rewards_b, gamma=cfg.gamma)
                log_probs_obs_adv = adv_agent.evaluate_actions(obs_b, adv_obs_b)
                values_tensor = torch.tensor(values_b).to(device)
                masks_tensor = torch.tensor(masks_b).to(device)
                loss_obs_adv = (-log_probs_obs_adv * values_tensor * masks_tensor).sum()
                adv_agent_optimizer.zero_grad()
                loss_obs_adv.backward()
                adv_agent_optimizer.step()
                writer.add_scalar("losses/adv_obs_loss", loss_obs_adv.item(), global_step)
                
            if global_step % cfg.train_frequency == 0:
                if cfg.use_nfsp:
                    states, _ = nfsp_rb.sample(cfg.batch_size)
                    states = torch.tensor(states).to(device)
                    with torch.no_grad():
                        belief = get_quantile_belief(cfg, states, 
                                                     agent.quantile_belief_network) \
                                                     if cfg.use_quantile_belief else None
                    loss_nfsp = F.mse_loss(agent.q_network(states, belief), agent.q_network_nfsp(states, belief))
                    nfsp_agent_optimizer.zero_grad()
                    loss_nfsp.backward()
                    nfsp_agent_optimizer.step()
                    if global_step % 100 == 0:
                        writer.add_scalar("losses/nfsp_loss", loss_nfsp.item(), global_step)    
                
                states, actions, rewards, dones, next_states = rb.sample(cfg.batch_size)
                # if cfg.use_data_augmentation:
                #     aug_states, aug_actions, aug_rewards, aug_dones, aug_next_states = augmentation_rb.sample(int(cfg.batch_size * cfg.update_ratio))
                #     states = torch.cat([states, aug_states], dim=0)
                #     actions = torch.cat([actions, aug_actions], dim=0)
                #     rewards = torch.cat([rewards, aug_rewards], dim=0)
                #     dones = torch.cat([dones, aug_dones], dim=0)
                #     next_states = torch.cat([next_states, aug_next_states], dim=0)
                
                if cfg.use_quantile_belief:
                    loss_quantile = 0.
                    # price_index = 4 if cfg.level == 'day' else 3
                    price_index = 19
                    y_true = (states[:, -1, price_index]).unsqueeze(-1)     # batch, 1
                    y_pred = agent.quantile_belief_network(states[:, :-1])[:, -1] # batch, quantile heads num
                    for i, q in enumerate(cfg.quantile_heads):
                        e = y_true - y_pred[:, i]
                        loss_quantile += torch.mean(torch.max((q - 1) * e, q * e))
                    if global_step % 100 == 0:
                        writer.add_scalar("losses/quantile_loss", loss_quantile, global_step)
                    belief_optimizer.zero_grad()
                    loss_quantile.backward()
                    belief_optimizer.step()
                
                with torch.no_grad():
                    target_belief = get_quantile_belief(cfg, next_states, 
                                                        agent.quantile_belief_network) \
                                                        if cfg.use_quantile_belief else None
                    belief = get_quantile_belief(cfg, states, 
                                                 agent.quantile_belief_network) \
                                                 if cfg.use_quantile_belief else None
                    
                    target_max, _ = agent.target_network(next_states, target_belief).max(dim=1)
                    td_target = rewards.flatten() + cfg.gamma * target_max * (1 - dones.flatten())
                old_val = agent.q_network(states, belief).gather(1, actions.unsqueeze(1).long()).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
                validate_agent(cfg, agent, val_envs, writer, device, global_step, exp_path)
                torch.save(agent.state_dict(),
                           os.path.join(exp_path, cfg.save_path, "{}.pth".format(global_step // cfg.check_steps)))
                if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent']:
                    torch.save(adv_agent.state_dict(),
                            os.path.join(exp_path, cfg.save_path, "{}_adv.pth".format(global_step // cfg.check_steps + 1)))

            pre_global_step = global_step

    validate_agent(cfg, agent, val_envs, writer, device, global_step, exp_path)
    torch.save(agent.state_dict(),
               os.path.join(exp_path, cfg.save_path, "{}.pth".format(global_step // cfg.check_steps + 1)))
    if cfg.use_data_augmentation and cfg.augmentation_method in ['adv_agent', 'generator_adv_agent']:
        torch.save(adv_agent.state_dict(),
                os.path.join(exp_path, cfg.save_path, "{}_adv.pth".format(global_step // cfg.check_steps + 1)))

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
            if cfg.use_quantile_belief:
                quantile_belief = get_quantile_belief(cfg, 
                                                    torch.Tensor(obs).to(device),
                                                    agent.quantile_belief_network)
            else:
                quantile_belief = None
            if cfg.use_nfsp:
                q_values = agent.q_network_nfsp(torch.Tensor(obs).to(device), quantile_belief)
            else:
                q_values = agent.target_network(torch.Tensor(obs).to(device), quantile_belief)
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

def get_quantile_belief(cfg, obs: torch.Tensor, quantile_belief_network):
    quantile_logits = quantile_belief_network(obs[:, :-1, :])[:, -1]
    # if cfg.level == 'day':
    #     current_price = obs[:, -1, 4]
    # elif cfg.level == 'minute':
    #     current_price = obs[:, -1, 3]
    # else:
    #     raise NotImplementedError
    current_price = obs[:, -1, 19]
    difference = (quantile_logits - current_price.unsqueeze(-1)) ** 2
    return torch.argmin(difference, dim=-1)

if __name__ == '__main__':
    main()

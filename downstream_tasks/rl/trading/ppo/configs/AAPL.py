root = None
workdir = "downstream_tasks/rl/trading/workdir/exp02/trading/AAPL/ppo"
tag = "exp001"
save_path = "saved_model"
level = "day"
select_stock = "AAPL"

# setting parameters [do not change]
train_start_date = "2022-06-01"
train_end_date = "2023-06-01"
valid_start_date = "2023-06-01"
valid_end_date = "2024-01-01"
timestamps = 30
num_features = 153  # num features name (150) + num temporals name (3)
temporals_name = ['day', 'weekday', 'month']
embed_dim = 64
depth = 1  # 2 mlp layers
initial_amount = 1e4
transaction_cost_pct = 1e-3

# training parameters (adjust mainly)
policy_learning_rate = 5e-7
value_learning_rate = 1e-6
num_envs = 4
num_steps = 128
policy_num_minibatches = 128
value_num_minibatches = 16
gradient_checkpointing_steps = 32
total_timesteps = int(200000)  # 2e6
check_steps = int(10000)
seed = 10

# fixed parameters of training [do not change]
gae_lambda = 0.95
batch_size = int(num_envs * num_steps)
policy_minibatch_size = int(batch_size // policy_num_minibatches)
value_minibatch_size = int(batch_size // value_num_minibatches)
anneal_lr = True
clip_vloss = True
norm_adv = True
clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
max_grad_norm = 0.5
critic_warm_up_steps = 0
target_kl = 0.02
gamma = 0.99
update_epochs = 1

transition = ["states", "actions", "logprobs", "rewards", "dones", "values"]
transition_shape = dict(
    states=dict(shape=(num_envs, timestamps, num_features), type="float32"),
    actions=dict(shape=(num_envs,), type="int32"),
    logprobs=dict(shape=(num_envs,), type="float32"),
    rewards=dict(shape=(num_envs,), type="float32"),
    dones=dict(shape=(num_envs,), type="float32"),
    values=dict(shape=(num_envs,), type="float32"),
)

dataset = dict(
    root=root,
    data_path="datasets/processd_day_dj30/features",
    stocks_path="configs/_asset_list_/exp_stocks.txt",
    features_name=[
        'open',
        'high',
        'low',
        'close',
        'adj_close',
        'kmid',
        'kmid2',
        'klen',
        'kup',
        'kup2',
        'klow',
        'klow2',
        'ksft',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'rsv_5',
        'rsv_10',
        'rsv_20',
        'rsv_30',
        'rsv_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'corr_5',
        'corr_10',
        'corr_20',
        'corr_30',
        'corr_60',
        'cord_5',
        'cord_10',
        'cord_20',
        'cord_30',
        'cord_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
        'vma_5',
        'vma_10',
        'vma_20',
        'vma_30',
        'vma_60',
        'vstd_5',
        'vstd_10',
        'vstd_20',
        'vstd_30',
        'vstd_60',
        'wvma_5',
        'wvma_10',
        'wvma_20',
        'wvma_30',
        'wvma_60',
        'vsump_5',
        'vsump_10',
        'vsump_20',
        'vsump_30',
        'vsump_60',
        'vsumn_5',
        'vsumn_10',
        'vsumn_20',
        'vsumn_30',
        'vsumn_60',
        'vsumd_5',
        'vsumd_10',
        'vsumd_20',
        'vsumd_30',
        'vsumd_60',
        'log_volume'
    ],
    labels_name=[
        'ret1',
        'mov1'
    ],
    temporals_name=temporals_name
)

env = dict(
    mode="train",
    dataset=None,
    select_stock=select_stock,
    if_norm=True,
    if_norm_temporal=False,
    scaler=None,
    timestamps=timestamps,
    start_date=None,
    end_date=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    level=level
)

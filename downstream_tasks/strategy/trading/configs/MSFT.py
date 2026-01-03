root = None
workdir = "downstream_tasks/strategy/trading/workdir/exp/trading"
resdir = "res/strategy_record/trading"
tag = "exp001"
save_path = "saved_model"
level = "day"
select_stock = "MSFT"
seed = 42
train_flag = True
strategy_number = [0, 1, 2, 3, 4]
# setting parameters [do not change]
train_start_date = "2022-05-11"
train_end_date = "2023-06-01"
valid_start_date = "2023-05-11"
valid_end_date = "2024-01-01"
timestamps = 15
num_features = 153  # num features name (150) + num temporals name (3)
temporals_name = []
embed_dim = 64
depth = 1  # 2 mlp layers
initial_amount = 1e4
transaction_cost_pct = 1e-3

transition = ["states", "actions", "rewards", "dones", "next_states"]

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
    ],
    labels_name=[
        'ret1',
        'mov1'
    ],
    temporals_name=temporals_name
)

env = dict(
    mode="valid",
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

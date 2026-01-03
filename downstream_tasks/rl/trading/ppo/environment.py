import warnings
warnings.filterwarnings("ignore")
import numpy as np
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import gym
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)

from downstream_tasks.dataset import AugmentatedDatasetStocks as Dataset

class EnvironmentRET(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 select_stock: str = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = False,
                 scaler: List[StandardScaler] = None,
                 timestamps: int = 10,
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 level = "day"
                 ):
        super(EnvironmentRET, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.scaler = scaler
        self.timestamps = timestamps
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.level = level
        self.select_stock = select_stock

        if end_date is not None:
            assert end_date > start_date, "start date {}, end date {}, end date should be greater than start date".format(start_date, end_date)

        self.stocks = self.dataset.stocks
        self.stocks2id = self.dataset.stocks2id
        self.id2stocks = self.dataset.id2stocks

        self.features_name = self.dataset.features_name
        
        if self.level == "day":
            self.prices_name = ['open', 'high', 'low', 'close', 'adj_close']
        elif self.level == "minute":
            self.prices_name = ['open', 'high', 'low', 'close']
        else:
            raise NotImplementedError
            
        self.temporals_name = self.dataset.temporals_name
        self.labels_name = self.dataset.labels_name
        self.stocks_df = []

        prices = []
        if if_norm:
            print("normalize datasets")

            if self.mode == "train":
                self.scaler = []
                for df in self.dataset.stocks_df:

                    if end_date is not None:
                        df = df.loc[start_date:end_date]
                    else:
                        df = df.loc[start_date:]

                    df[self.prices_name] = df[[name for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = StandardScaler()
                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.fit_transform(df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.fit_transform(df[self.features_name])

                    self.scaler.append(scaler)
                    self.stocks_df.append(df)
            else:
                assert self.scaler is not None, "val mode or test mode is not None."

                for index, df in enumerate(self.dataset.stocks_df):

                    if end_date is not None:
                        df = df.loc[start_date:end_date]
                    else:
                        df = df.loc[start_date:]

                    df[self.prices_name] = df[[name for name in self.prices_name]]
                    price_df = df[self.prices_name]
                    prices.append(price_df.values)

                    scaler = self.scaler[index]

                    if self.if_norm_temporal:
                        df[self.features_name + self.temporals_name] = scaler.transform(df[self.features_name + self.temporals_name])
                    else:
                        df[self.features_name] = scaler.transform(df[self.features_name])

                    self.stocks_df.append(df)
        else:
            print("no normalize datasets")

        self.features = []
        self.select_stock_id = self.stocks2id[self.select_stock]
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)
        self.features = self.features[self.select_stock_id, :, :]

        self.prices = np.stack(prices)
        print(self.prices.shape)
        self.prices = self.prices[self.select_stock_id, :, :]

        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)
        self.labels = self.labels[self.select_stock_id, :, :]

        print("features shape {}, prices shape {}, labels shape {}, num timestamps {}".format(self.features.shape,
                                                                                        self.prices.shape,
                                                                                        self.labels.shape, self.features.shape[0]))
        self.num_timestamps = self.features.shape[0]
        self.hold_on_action = 1 # sell, hold, buy=>-1, 0, 1
        self.action_dim = 2 * self.hold_on_action + 1
        self.actions = ["SELL", "HOLD", "BUY"]

    def init_timestamp_index(self):
        if self.mode == "train":
            timestamp = random.randint(self.timestamps - 1, 3 * (self.num_timestamps // 4))
        else:
            timestamp = self.timestamps - 1
        return timestamp

    def get_current_timestamp_datetime(self):
        return self.stocks_df[self.select_stock_id].index[self.timestamp_index]

    def current_value(self, price):
        return self.cash + self.position * price

    def get_price(self):
        prices = self.prices[self.timestamp_index, :]

        if self.level == "day":
            o, h, l, c, adj = prices[0], prices[1], prices[2], prices[3], prices[4]
            price = adj
        elif self.level == "minute":
            o, h, l, c = prices[0], prices[1], prices[2], prices[3]
            price = c
        else:
            raise NotImplementedError

        return price

    def reset(self, **kwargs):

        self.timestamp_index = self.init_timestamp_index()
        self.timestamp_datetime = self.get_current_timestamp_datetime()
        self.price = self.get_price()

        state = self.features[self.timestamp_index - self.timestamps + 1: self.timestamp_index + 1, :]

        self.ret = 0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.value = self.initial_amount
        self.total_return = 0
        self.total_profit = 0
        self.action = "HOLD"

        info= {
            "timestamp": self.timestamp_datetime.strftime("%Y-%m-%d %H:%M:%S") if self.level == "minute" else self.timestamp_datetime.strftime("%Y-%m-%d"),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action
        }

        return state, info

    def eval_buy_position(self, price):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(self.cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self):
        # evaluate sell position
        return int(self.position)

    def buy(self, price, amount):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_buy_postion))

        self.cash -= buy_position * price * (1 + self.transaction_cost_pct)
        self.position += buy_position
        self.value = self.current_value(price)

        if buy_position == 0:
            self.action = "HOLD"
        else:
            self.action = "BUY"

    def sell(self, price, amount):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_sell_postion))

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

        if sell_position == 0:
            self.action = "HOLD"
        else:
            self.action = "SELL"

    def noop(self, price, amount):
        self.value = self.current_value(price)

        self.action = "HOLD"

    def step(self, action: int = 0):

        pre_value = self.value

        action = action - self.hold_on_action

        if action > 0:
            self.buy(self.price, amount=action)
        elif action < 0:
            self.sell(self.price, amount=action)
        else:
            self.noop(self.price, amount=action)

        post_value = self.value

        self.timestamp_index = self.timestamp_index + 1
        self.timestamp_datetime = self.get_current_timestamp_datetime()
        self.price = self.get_price()

        next_state = self.features[self.timestamp_index - self.timestamps + 1: self.timestamp_index + 1, :]
        reward = (post_value - pre_value) / pre_value

        self.state = next_state

        self.ret = reward
        self.discount *= 0.99
        self.total_return += self.discount * reward
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        if self.timestamp_index < self.num_timestamps - 1:
            done = False
            truncted = False
        else:
            done = True
            truncted = True

        info = {
            "timestamp": self.timestamp_datetime.strftime("%Y-%m-%d %H:%M:%S") if self.level == "minute" else self.timestamp_datetime.strftime("%Y-%m-%d"),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": str(self.action)
        }

        return next_state, reward, done, truncted, info

if __name__ == '__main__':
    temporals_name = ['day', 'weekday', 'month']

    dataset = dict(
        root=ROOT,
        data_path="datasets/dj30/features",
        stocks_path="configs/_stock_list_/dj30.txt",
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

    dataset = Dataset(
        **dataset
    )

    env = EnvironmentRET(
        mode="train",
        dataset=dataset,
        select_stock="AAPL",
        if_norm=True,
        if_norm_temporal=False,
        scaler=None,
        timestamps=10,
        start_date="2008-03-19",
        end_date="2020-12-01",
        initial_amount=1e4,
        transaction_cost_pct=1e-3,
        level="day"
    )

    state, info = env.reset()

    done = False
    while not done:
        action = 1
        next_state, reward, done, truncted, info = env.step(action)
        print(info)
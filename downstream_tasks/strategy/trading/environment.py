import warnings

warnings.filterwarnings("ignore")
import numpy as np
from typing import Any
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)


class EnvironmentRET():
    def __init__(self,
                 dataset: Any = None,
                 select_stock: str = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = False,
                 timestamps: int = 10,
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 level="day"
                 ):
        super(EnvironmentRET, self).__init__()

        self.dataset = dataset
        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.timestamps = timestamps
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.level = level
        self.select_stock = select_stock

        if end_date is not None:
            assert end_date > start_date, "start date {}, end date {}, end date should be greater than start date".format(
                start_date, end_date)

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

        # observations_start_data = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=timestamps)).strftime("%Y-%m-%d")

        for index, df in enumerate(self.dataset.stocks_df):

            if end_date is not None:
                df = df.loc[start_date:end_date]
            else:
                df = df.loc[start_date:]

            df[self.prices_name] = df[[name for name in self.prices_name]]
            price_df = df[self.prices_name]
            prices.append(price_df.values)

            self.stocks_df.append(df)

        self.features = []
        self.select_stock_id = self.stocks2id[self.select_stock]
        for df in self.stocks_df:
            df = df[self.features_name + self.temporals_name]
            self.features.append(df.values)
        self.features = np.stack(self.features)
        self.features = self.features[self.select_stock_id, :, :]

        self.prices = np.stack(prices)
        self.prices = self.prices[self.select_stock_id, :, :]

        self.labels = []
        for df in self.stocks_df:
            df = df[self.labels_name]
            self.labels.append(df.values)
        self.labels = np.stack(self.labels)
        self.labels = self.labels[self.select_stock_id, :, :]

        print("features shape {}, prices shape {}, labels shape {}, num timestamps {}".format(self.features.shape,
                                                                                              self.prices.shape,
                                                                                              self.labels.shape,
                                                                                              self.features.shape[0]))
        self.num_timestamps = self.features.shape[0]
        self.hold_on_action = 1  # buy, hold, sell=>-1, 0, 1
        self.action_dim = 2 * self.hold_on_action + 1

        # update the self.timestamps to the size of the dataset

    def init_timestamp_index(self):
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

        info = {
            "timestamp": self.timestamp_datetime.strftime(
                "%Y-%m-%d %H:%M:%S") if self.level == "minute" else self.timestamp_datetime.strftime("%Y-%m-%d"),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "data": self.features
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

    def sell(self, price, amount):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_sell_postion))

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

    def noop(self, price, amount):
        self.value = self.current_value(price)

    def action_text_to_value(self, trade_signal):
        if trade_signal == "BUY":
            return 2
        elif trade_signal == "SELL":
            return 0
        else:
            return 1

    def step(self, signal: int = 0):

        action = self.action_text_to_value(signal)
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
            "timestamp": self.timestamp_datetime.strftime(
                "%Y-%m-%d %H:%M:%S") if self.level == "minute" else self.timestamp_datetime.strftime("%Y-%m-%d"),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
        }

        return next_state, reward, done, truncted, info

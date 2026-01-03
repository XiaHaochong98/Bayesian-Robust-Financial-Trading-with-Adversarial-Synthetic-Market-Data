import os
from typing import List

import pandas as pd


class Dataset():
    def __init__(self,
                 root: str = None,
                 data_path: str = None,
                 stocks_path: str = None,
                 features_name: List[str] = None,
                 temporals_name: List[str] = None,
                 labels_name: List[str] = None, ):
        super(Dataset, self).__init__()

        self.root = root
        self.data_path = data_path
        self.stocks_path = stocks_path
        self.features_name = features_name
        self.temporals_name = temporals_name
        self.labels_name = labels_name

        self.data_path = os.path.join(root, self.data_path)
        self.stocks_path = os.path.join(root, self.stocks_path)

        self.stocks = self._init_stocks()

        self.stocks2id = {stock: i for i, stock in enumerate(self.stocks)}
        self.id2stocks = {i: stock for i, stock in enumerate(self.stocks)}

        self.stocks_df = self._init_stocks_df()

    def _init_stocks(self):
        print("init stocks...")
        stocks = []
        with open(self.stocks_path) as op:
            for line in op.readlines():
                line = line.strip()
                stocks.append(line)
        print("init stocks success...")
        return stocks

    def _init_stocks_df(self):
        print("init stocks dataframe...")
        stocks_df = []
        for stock in self.stocks:
            path = os.path.join(self.data_path, f"{stock}.parquet")
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d")
            df = df.set_index("timestamp")
            df = df[self.features_name + self.temporals_name + self.labels_name]
            print(stock, df.shape)
            stocks_df.append(df)
        print("init stocks dataframe success...")
        return stocks_df

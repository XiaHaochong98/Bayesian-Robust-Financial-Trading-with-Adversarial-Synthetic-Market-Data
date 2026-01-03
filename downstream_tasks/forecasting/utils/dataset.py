import numpy as np
import os
import pandas as pd
import torch
from typing import List


class StocksDataset():
    # this class is used to load the stocks data, where in the feature we have already included "ret1" and "mov1" as the regression and classification target
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



class RegressionDataset(torch.utils.data.Dataset_Stocks):
    # this class is used to load the dataset for regression task
    def __init__(self, data, mode="direct", prediction_len=1, label_features=[]):
        # Validate data dimensions (assuming data is a numpy array or similar)
        if data.ndim != 3:
            raise ValueError("Data should be three-dimensional (batch, sequence, features)")

        # Handling label features (assuming label_features is a list of indices)
        if label_features:
            data = data[:, :, label_features]

        # Prepare X and Y based on the mode
        if mode == "direct":
            # X is the data excluding the last prediction_len steps
            self.X = torch.FloatTensor(data[:, :-prediction_len, :])

            # Y is the return calculated only for the prediction_len steps
            future_data = np.roll(data, -prediction_len, axis=1)
            returns = (future_data[:, -prediction_len:, :] - data[:, -prediction_len - 1:-1, :]) / data[:,
                                                                                                   -prediction_len - 1:-1,
                                                                                                   :]
            self.Y = torch.FloatTensor(returns)
        elif mode == "overlapping":
            self.X = torch.FloatTensor(data[:, :-prediction_len, :])
            self.Y = self.calculate_returns(data, prediction_len)[:, prediction_len:, :]
        else:
            raise ValueError("mode should be either 'direct' or 'selective'")

        self.Y = torch.FloatTensor(self.Y)

    def calculate_returns(self, data, prediction_len):
        """
        Calculate returns over specified prediction length.
        Assumes data is three-dimensional (batch, sequence, features).
        """
        future_data = np.roll(data, -prediction_len, axis=1)
        returns = (future_data - data) / data
        return returns

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ClassificationDataset(torch.utils.data.Dataset_Stocks):
    # this class is used to load the dataset for classification task
    def __init__(self, data, mode="direct", prediction_len=1, label_features=[]):
        # Validate data dimensions (assuming data is a numpy array or similar)
        if data.ndim != 3:
            raise ValueError("Data should be three-dimensional (batch, sequence, features)")

        # Handling label features (assuming label_features is a list of indices)
        if label_features:
            data = data[:, :, label_features]

        # Prepare X and Y based on the mode
        if mode == "direct":
            # X is the data excluding the last prediction_len steps
            self.X = torch.FloatTensor(data[:, :-prediction_len, :])

            # Y is the return calculated only for the prediction_len steps
            future_data = np.roll(data, -prediction_len, axis=1)
            returns = (future_data[:, -prediction_len:, :] - data[:, -prediction_len - 1:-1, :]) / data[:,
                                                                                                   -prediction_len - 1:-1,
                                                                                                   :]
            self.Y = torch.FloatTensor(returns)
        elif mode == "overlapping":
            self.X = torch.FloatTensor(data[:, :-prediction_len, :])
            self.Y = self.calculate_returns(data, prediction_len)[:, prediction_len:, :]
        else:
            raise ValueError("mode should be either 'direct' or 'selective'")

        self.Y = torch.FloatTensor(self.Y)

        # cast to label using sign
        self.Y = torch.sign(self.Y)

    def calculate_returns(self, data, prediction_len):
        """
        Calculate returns over specified prediction length.
        Assumes data is three-dimensional (batch, sequence, features).
        """
        future_data = np.roll(data, -prediction_len, axis=1)
        returns = (future_data - data) / data
        return returns

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DiscriminatorDataset(torch.utils.data.Dataset_Stocks):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, ori_data, generated_data, ori_time, generated_time):
        self.Fake_data = torch.FloatTensor(generated_data)
        self.Real_data = torch.FloatTensor(ori_data)

    def __len__(self):
        return len(self.Fake_data)

    def __getitem__(self, idx):
        return self.Fake_data[idx], self.Real_data[idx]


class LabelPredictionDataset(torch.utils.data.Dataset_Stocks):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, data, label):
        self.X = torch.FloatTensor(data)
        self.Y = torch.FloatTensor(label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

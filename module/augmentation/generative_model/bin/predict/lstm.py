# make a torch LSTM network for time-sereis forecasting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (
            self.X[index : index + self.seq_len],
            self.y[index + self.seq_len - 1],
        )


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # x: (seq_len, batch, input_size)
        # h: (num_layers, batch, hidden_size)
        out, h = self.lstm(x, h)
        out = self.linear(out[-1])  # (batch, output_size)
        return out, h


class LSTMRegressor(pl.LightningModule):
    """
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(
        self,
        n_features,
        hidden_size,
        seq_len,
        batch_size,
        num_layers,
        dropout,
        learning_rate,
        criterion,
    ):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log("test_loss", loss)
        return result


class StockDataModule(pl.LightningDataModule):
    """
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    """

    def __init__(self, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        Data is resampled to hourly intervals.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        """

        if stage == "fit" and self.X_train is not None:
            return
        if stage == "test" and self.X_test is not None:
            return
        if (
            stage is None
            and self.X_train is not None
            and self.X_test is not None
        ):
            return

        path = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"

        df = pd.read_csv(
            path,
            sep=";",
            parse_dates={"dt": ["Date", "Time"]},
            infer_datetime_format=True,
            low_memory=False,
            na_values=["nan", "?"],
            index_col="dt",
        )

        df_resample = df.resample("h").mean()

        X = df_resample.dropna().copy()
        y = X["Global_active_power"].shift(-1).ffill()
        self.columns = X.columns

        X_cv, X_test, y_cv, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_cv, y_cv, test_size=0.25, shuffle=False
        )

        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if stage == "fit" or stage is None:
            self.X_train = preprocessing.transform(X_train)
            self.y_train = y_train.values.reshape((-1, 1))
            self.X_val = preprocessing.transform(X_val)
            self.y_val = y_val.values.reshape((-1, 1))

        if stage == "test" or stage is None:
            self.X_test = preprocessing.transform(X_test)
            self.y_test = y_test.values.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(
            self.X_train, self.y_train, seq_len=self.seq_len
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(
            self.X_val, self.y_val, seq_len=self.seq_len
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(
            self.X_test, self.y_test, seq_len=self.seq_len
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return test_loader

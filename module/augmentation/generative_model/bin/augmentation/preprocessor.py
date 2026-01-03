from .utils import get_attr
import pandas as pd
import os
import yfinance as yf
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression


class YfinancePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.data_path = get_attr(kwargs, "data_path", None)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.train_valid_test_portion = get_attr(
            kwargs, "train_valid_test_portion", [0.8, 0.1, 0.1]
        )

        self.train_path = get_attr(kwargs, "train_path", None)
        self.valid_path = get_attr(kwargs, "valid_path", None)
        self.test_path = get_attr(kwargs, "test_path", None)

        self.start_date = get_attr(kwargs, "start_date", "2000-01-01")
        self.end_date = get_attr(kwargs, "end_date", "2023-12-31")
        self.tickers = get_attr(kwargs, "tickers", None)

        self.indicator = get_attr(kwargs, "indicator", "basic")

    def download_data(self):
        df_list = []
        for ticker in self.tickers:
            df = yf.download(ticker, self.start_date, self.end_date)
            df["ticker"] = ticker
            df.index = df.index.values
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df = df.sort_values(by="ticker")
        df = df.sort_index()
        df["date"] = df.index
        return df

    def clean_data(self):
        initial_ticker_list = self.df[
            self.df.index == self.df.index.unique()[0]
        ]["ticker"].values.tolist()
        initial_ticker_list = set(initial_ticker_list)
        for index in tqdm(self.df.index.unique()):
            ticker_list = self.df[self.df.index == index][
                "ticker"
            ].values.tolist()
            ticker_list = set(ticker_list)
            initial_ticker_list = initial_ticker_list & ticker_list
        df_list = []
        for ticker in initial_ticker_list:
            new_df = self.df[self.df.ticker == ticker]
            df_list.append(new_df)
        df = pd.concat(df_list)
        df = df.sort_values(by="ticker")
        df = df.sort_index()
        df = df.copy()
        df = df.sort_values(["date", "ticker"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(
            index="date", columns="ticker", values="close"
        )
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.ticker.isin(tics)]
        return df

    def make_feature(self):
        """Function creates basic features for the stock data using the Close Price."""
        df_new = self.df.sort_values(by=["ticker", "date"])
        stock = df_new
        unique_ticker = stock.ticker.unique()
        # print(unique_ticker)
        df_indicator = pd.DataFrame()
        # print(df_indicator)
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.ticker == unique_ticker[i]]
            # print(temp_indicator)
            temp_indicator["zclose"] = (
                temp_indicator.close
                / (
                    temp_indicator.close.rolling(2).sum()
                    - temp_indicator.close
                )
            ) - 1
            temp_indicator["zd_5"] = (
                temp_indicator.close.rolling(5).sum() / 5
            ) / temp_indicator.close - 1
            temp_indicator["zd_10"] = (
                temp_indicator.close.rolling(10).sum() / 10
            ) / temp_indicator.close - 1
            temp_indicator["zd_15"] = (
                temp_indicator.close.rolling(15).sum() / 15
            ) / temp_indicator.close - 1
            temp_indicator["zd_20"] = (
                temp_indicator.close.rolling(20).sum() / 20
            ) / temp_indicator.close - 1
            temp_indicator["zd_25"] = (
                temp_indicator.close.rolling(25).sum() / 25
            ) / temp_indicator.close - 1
            temp_indicator["zd_30"] = (
                temp_indicator.close.rolling(30).sum() / 30
            ) / temp_indicator.close - 1
            temp_indicator["zd_60"] = (
                temp_indicator.close.rolling(60).sum() / 60
            ) / temp_indicator.close - 1

            # Calculate RSI for Close Prices with a span of 30 days as window(as this our genertaion window for diffison model)
            delta = temp_indicator["close"].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=30).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=30).mean()
            rs = gain / loss
            temp_indicator["RSI"] = 100 - (100 / (1 + rs))

            # Calculate EMA for Close Prices with a span of 30 days as window(as this our genertaion window for diffison model)
            temp_indicator["EMA"] = (
                temp_indicator["close"].ewm(span=30, adjust=False).mean()
            )
            df_indicator = df_indicator.append(
                temp_indicator, ignore_index=True
            )

        df_indicator = df_indicator.fillna(method="ffill").fillna(
            method="bfill"
        )

        return df_indicator

    def make_alpha(self):
        """Function creates alpha features from Alpha158 for the stock data using the Close Price only."""
        df_new = self.df.sort_values(by=["ticker", "date"])
        stock = df_new
        unique_ticker = stock.ticker.unique()
        df_indicator = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.ticker == unique_ticker[i]]
            # print(temp_indicator)
            temp_indicator["zclose"] = (
                temp_indicator.close
                / (
                    temp_indicator.close.rolling(2).sum()
                    - temp_indicator.close
                )
            ) - 1
            temp_indicator["zd_5"] = (
                temp_indicator.close.rolling(5).sum() / 5
            ) / temp_indicator.close - 1
            temp_indicator["zd_10"] = (
                temp_indicator.close.rolling(10).sum() / 10
            ) / temp_indicator.close - 1
            temp_indicator["zd_15"] = (
                temp_indicator.close.rolling(15).sum() / 15
            ) / temp_indicator.close - 1
            temp_indicator["zd_20"] = (
                temp_indicator.close.rolling(20).sum() / 20
            ) / temp_indicator.close - 1
            temp_indicator["zd_25"] = (
                temp_indicator.close.rolling(25).sum() / 25
            ) / temp_indicator.close - 1
            temp_indicator["zd_30"] = (
                temp_indicator.close.rolling(30).sum() / 30
            ) / temp_indicator.close - 1
            temp_indicator["zd_60"] = (
                temp_indicator.close.rolling(60).sum() / 60
            ) / temp_indicator.close - 1

            # Below are reimplementation of Microsoft Q lib Alpha 158 (with out volume-related) technical indicators.
            # For Below technical indicators: see factor reference at qlib alpha158: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

            # Define the window sizes to use
            windows = [5, 10, 20, 30, 60]
            # Loop over the window sizes and create the columns
            for w in windows:
                # ROC
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                temp_indicator[f"ROC{w}"] = (
                    temp_indicator["close"].shift(w) / temp_indicator["close"]
                )

                # MA
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                temp_indicator[f"MA{w}"] = (
                    temp_indicator["close"].rolling(window=w).mean()
                    / temp_indicator["close"]
                )

                # STD
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                temp_indicator["STD" + str(w)] = (
                    temp_indicator["close"].rolling(window=w).std()
                    / temp_indicator["close"]
                )

                # BETA
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                temp_indicator[f"BETA{w}"] = (
                    temp_indicator["close"]
                    .rolling(window=w)
                    .apply(lambda x: np.polyfit(range(w), x, 1)[0])
                    / temp_indicator["close"]
                )

                # RSQR
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                x = pd.Series(range(1, w + 1))
                y = temp_indicator["close"][-w:]
                x = x.values.reshape(-1, 1)
                y = y.values.reshape(-1, 1)
                reg = LinearRegression().fit(x, y)
                temp_indicator[f"RSQR{w}"] = reg.score(x, y)

                # RESI
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                # resi = y - reg.predict(x)
                # temp_indicator[f'RESI{w}'] = resi

                # MAX
                # The max price for past d days, divided by latest close price to remove unit
                temp_indicator["MAX" + str(w)] = (
                    temp_indicator["close"].rolling(window=w).max()
                    / temp_indicator["close"]
                )

                # MIN
                # The low price for past d days, divided by latest close price to remove unit
                temp_indicator["MIN" + str(w)] = (
                    temp_indicator["close"].rolling(window=w).min()
                    / temp_indicator["close"]
                )

                # QTLU
                # Used with MIN and MAX
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                temp_indicator["QTLU" + str(w)] = (
                    temp_indicator["close"].rolling(window=w).quantile(0.8)
                    / temp_indicator["close"]
                )

                # QTLD
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                temp_indicator["QTLD" + str(w)] = (
                    temp_indicator["close"].rolling(window=w).quantile(0.2)
                    / temp_indicator["close"]
                )

                # RANK
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                temp_indicator["RANK" + str(w)] = (
                    temp_indicator["close"].rolling(window=w).rank()
                )

                # RSV
                # Represent the price position between upper and lower resistent price for past d days.
                temp_indicator["RSV" + str(w)] = (
                    temp_indicator["close"] - temp_indicator["MIN" + str(w)]
                ) / (
                    temp_indicator["MAX" + str(w)]
                    - temp_indicator["MIN" + str(w)]
                    + 1e-12
                )

                # CNTP
                # The percentage of days in past d days that price go up.
                # temp_indicator[f'CNTP{w}'] = temp_indicator['close'].gt(temp_indicator['close'].shift(1)).rolling(window=w).mean()
                temp_indicator[f"CNTP{w}"] = (
                    (temp_indicator["close"].pct_change(1).gt(0))
                    .rolling(window=w)
                    .mean()
                )

                # CNTN
                # The percentage of days in past d days that price go down.
                # temp_indicator[f'CNTN{w}'] = temp_indicator['close'].lt(temp_indicator['close'].shift(1)).rolling(window=w).mean()
                temp_indicator[f"CNTN{w}"] = (
                    (temp_indicator["close"].pct_change(1).lt(0))
                    .rolling(window=w)
                    .mean()
                )

                # CNTD
                # The diff between past up day and past down day
                temp_indicator[f"CNTD{w}"] = (
                    temp_indicator[f"CNTP{w}"] - temp_indicator[f"CNTN{w}"]
                )
                temp_indicator["ret1"] = temp_indicator["close"].pct_change(1)
                temp_indicator["abs_ret1"] = np.abs(temp_indicator["ret1"])
                temp_indicator["pos_ret1"] = temp_indicator["ret1"]
                temp_indicator["pos_ret1"][
                    temp_indicator["pos_ret1"].lt(0)
                ] = 0

                # SUMP
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                # temp_indicator[f'SUMP{w}'] = temp_indicator['close'].rolling(window=w).apply(lambda x: sum(x[x > x.shift()])).fillna(0)
                temp_indicator[f"SUMP{w}"] = temp_indicator[
                    "pos_ret1"
                ].rolling(w).sum() / (
                    temp_indicator["abs_ret1"].rolling(w).sum() + 1e-12
                )

                # SUMN
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                # temp_indicator[f'SUMN{w}'] = temp_indicator['close'].rolling(window=w).apply(lambda x: sum(x[x < x.shift()])).fillna(0)
                temp_indicator[f"SUMN{w}"] = 1 - temp_indicator[f"SUMP{w}"]

                # SUMD
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                temp_indicator[f"SUMD{w}"] = 2 * temp_indicator[f"SUMP{w}"] - 1

            temp_indicator.drop(
                columns=["ret1", "abs_ret1", "pos_ret1"], inplace=True
            )
            df_indicator = df_indicator.append(
                temp_indicator, ignore_index=True
            )

        df_indicator = df_indicator.fillna(method="ffill").fillna(
            method="bfill"
        )

        return df_indicator

    def get_date(self, df):
        date = df.date.unique()
        start_date = pd.to_datetime(date[0])
        end_date = pd.to_datetime(date[-1])
        return start_date, end_date

    def data_split(self, df, start, end, target_date_col="date"):

        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "ticker"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def split(self, data, portion):
        """split the data by the portion into train, valid, test, which is convinent for the users to do the time rolling
        experiment"""
        portion = np.array(portion)
        if np.sum(portion) > 1:
            portion = self.normalization(portion)
        if (len(portion) in [3]) == False:
            raise Exception(
                "we can only split data 3 parts: train, valid ,test"
            )
        start, end = self.get_date(data)
        duration_length = end - start
        train_len = duration_length * portion[0]
        valid_len = duration_length * portion[1]
        test_len = duration_length * portion[2]
        train_end = start + train_len
        valid_start = train_end
        valid_end = valid_start + valid_len
        test_start = valid_end
        test_end = test_start + test_len
        start = str(start)
        train_end = str(train_end)
        valid_start = str(valid_start)
        valid_end = str(valid_end)
        test_start = str(test_start)
        test_end = str(test_end)
        train = self.data_split(data, start, train_end)
        valid = self.data_split(data, valid_start, valid_end)
        test = self.data_split(data, test_start, test_end)
        return train, valid, test

    def normalization(self, portion):
        portion = np.array(portion)
        sum = np.sum(portion)
        portion = portion / sum
        return portion

    def run(self, custom_data_path=None):

        if not custom_data_path:
            self.df = self.download_data()
        else:
            self.df = pd.read_csv(custom_data_path)
        self.df = self.download_data()
        self.df = self.df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjclose",
                "Volume": "volume",
                "tic": "ticker",
                "date": "date",
            }
        )
        self.df.index = self.df.date.values

        self.df = self.clean_data()

        self.df = self.make_feature()
        if self.indicator == "alpha158_novolume":
            self.df = self.make_alpha()

        train, valid, test = self.split(self.df, self.train_valid_test_portion)

        # Modify column names as needed.
        train = train.rename(columns={"ticker": "tic"})
        valid = valid.rename(columns={"ticker": "tic"})
        test = test.rename(columns={"ticker": "tic"})

        train.to_csv(self.train_path)
        valid.to_csv(self.valid_path)
        test.to_csv(self.test_path)

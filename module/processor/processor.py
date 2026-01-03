import multiprocessing
import os
import time
from copy import deepcopy
from datetime import datetime

import backoff
import numpy as np
import pandas as pd
from langchain_community.document_loaders import PlaywrightURLLoader
from tqdm.auto import tqdm

from module.registry import PROCESSOR

from module.augmentation.transformation.augmentation_utils import *


@backoff.on_exception(backoff.expo,(Exception,), max_tries=3, max_value=10, jitter=None)
def langchain_parse_url(url):

    print(">" * 30 + "Running langchain_parse_url" + ">" * 30)
    start = time.time()
    loader = PlaywrightURLLoader(urls = [url], remove_selectors=["header", "footer"])
    data = loader.load()
    if len(data) <= 0:
        return None
    content = data[0].page_content

    if "Please enable cookies" in content:
        return None
    if "Please verify you are a human" in content:
        return None
    if "Checking if the site connection is secure" in content:
        return None

    print("url: {} | content: {}".format(url, content[:100].split("\n")))
    end = time.time()
    print(">" * 30 + "Time elapsed: {}s".format(end - start) + ">" * 30)
    print("<" * 30 + "Finish langchain_parse_url" + "<" * 30)
    return content

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_news(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["source"] = df["source"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df

def cal_guidance(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df

def cal_sentiment(df, columns):
    for col in columns:
        if "sentiment" not in col:
            df[col] = df.groupby("timestamp")[col].transform("sum")
        else:
            df[col] = df.groupby("timestamp")[col].transform("mean")
            df[col] = df[col].fillna(0.5)
    return df


def cal_factor(df, level="day"):
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    # print("data type of df: {}".format(df.dtypes))
    df["kmid"] = (df["close"] - df["open"]) / df["close"]
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df["klen"] = (df["high"] - df["low"]) / df["open"]
    df['kup'] = (df['high'] - df['max_oc']) / df['open']
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df["ksft"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    for w in window:
        shift = df['close'].shift(w)
        min = df["low"].where(df["low"] < shift, shift)
        max = df["high"].where(df["high"] > shift, shift)
        df["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    for w in window:
        df1 = df["close"].rolling(w)
        df2 = np.log(df["volume"] + 1).rolling(w)
        df["corr_{}".format(w)] = df1.corr(pairwise = df2)

    for w in window:
        df1 = df["close"]
        df_shift1 = df1.shift(1)
        df2 = df["volume"]
        df_shift2 = df2.shift(1)
        df1 = df1 / df_shift1
        df2 = np.log(df2 / df_shift2 + 1)
        df["cord_{}".format(w)] = df1.rolling(w).corr(pairwise = df2.rolling(w))

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1

    for w in window:
        df["vma_{}".format(w)] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)

    for w in window:
        df["vstd_{}".format(w)] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)

    for w in window:
        shift = np.abs((df["close"] / df["close"].shift(1) - 1)) * df["volume"]
        df1 = shift.rolling(w).std()
        df2 = shift.rolling(w).mean()
        df["wvma_{}".format(w)] = df1 / (df2 + 1e-12)

    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    for w in window:
        df["vsump_{}".format(w)] = df["pos_vchg1"].rolling(w).sum() / (df["abs_vchg1"].rolling(w).sum() + 1e-12)
    for w in window:
        df["vsumn_{}".format(w)] = 1 - df["vsump_{}".format(w)]
    for w in window:
        df["vsumd_{}".format(w)] = 2 * df["vsump_{}".format(w)] - 1

    df["log_volume"] = np.log(df["volume"] + 1)

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1', 'volume'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    if level == "minute":
        df["minute"] = pd.to_datetime(df.index).minute
        df["hour"] = pd.to_datetime(df.index).hour

    df["day"] = pd.to_datetime(df.index).day
    df["weekday"] = pd.to_datetime(df.index).weekday
    df["month"] = pd.to_datetime(df.index).month

    return df

def cal_target(df):
    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)
    return df

@PROCESSOR.register_module(force=True)
class Processor():
    def __init__(self,
                 root=None,
                 path_params = None,
                 stocks_path = None,
                 start_date = None,
                 end_date = None,
                 interval="day",
                 if_parse_url = False,
                 workdir = None,
                 tag = None
                 ):
        self.root = root
        self.path_params = path_params
        self.stocks_path = os.path.join(root, stocks_path)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.if_parse_url = if_parse_url
        self.workdir = workdir
        self.tag = tag

        self.stocks = self._init_stocks()

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def augmentation(self,data,augmetation_configs):
        pass
    #TODO

    def _process_price_and_features(self,
                stocks = None,
                start_date = None,
                end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        # Convert to timezone-aware (UTC)
        start_date = pd.to_datetime(start_date).tz_localize("UTC")
        end_date = pd.to_datetime(end_date).tz_localize("UTC")

        stocks = stocks if stocks else self.stocks

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close"
        ]

        for stock in tqdm(stocks):
            price = self.path_params["prices"][0]
            price_type = price["type"]
            price_path = price["path"]

            price_path = os.path.join(self.root, price_path, "{}.csv".format(stock))

            if price_type == "fmp":
                price_column_map = {
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "adjClose": "adj_close",
                }
            elif price_type == "yahoofinance":
                price_column_map = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Date": "timestamp",
                    "Adj Close": "adj_close",
                }
            else:
                price_column_map = {
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "adjClose": "adj_close",
                }

            assert os.path.exists(price_path), "Price path {} does not exist".format(price_path)
            price_df = pd.read_csv(price_path)

            price_df = price_df.rename(columns=price_column_map)[["timestamp"] + price_columns]

            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
            # print(f"price_df[timestamp]: {price_df['timestamp'].head()}")
            # print(f"start_date: {start_date}")
            # print(f"end_date: {end_date}")
            price_df = price_df[(price_df["timestamp"] >= start_date) & (price_df["timestamp"] < end_date)]

            price_df = price_df.sort_values(by="timestamp")

            # # cast the [timestamp] column to string of "YYYY-MM-DD"
            # price_df["timestamp"] = price_df["timestamp"].dt.strftime("%Y-%m-%d")

            price_df = price_df.drop_duplicates(subset=["timestamp"], keep="first")
            price_df = price_df.reset_index(drop=True)

            for col in price_df.select_dtypes(include=['object']).columns:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

            outpath = os.path.join(self.root, self.workdir, self.tag, "price")
            os.makedirs(outpath, exist_ok=True)
            price_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

            features_df = cal_factor(deepcopy(price_df), level=self.interval)
            features_df = cal_target(features_df)
            outpath = os.path.join(self.root, self.workdir, self.tag, "features")
            os.makedirs(outpath, exist_ok=True)
            print("outpath: {}".format(outpath))
            features_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)
    
    def process(self,
                stocks = None,
                start_date = None,
                end_date = None):
        if "prices" in self.path_params:
            print(">" * 30 + "Running price and features..." + ">" * 30)
            self._process_price_and_features(stocks=stocks, start_date=start_date, end_date=end_date)
            print("<" * 30 + "Finish price and features..." + "<" * 30)
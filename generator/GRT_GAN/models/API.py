import os
import pickle
from typing import Dict, Union
import time
# 3rd party modules
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
# import the current file into python path
from pathlib import Path
# Add the parent directory of 'models' to the Python path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.timegan import TimeGAN

import pandas as pd
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GeneratorAPI:

    def __init__(self, model_path,ticker_name,obs_features,temporal_features):
        """Load the model and data for inference"""

        # Set the path to the output data folder relative to this script
        output_data_folder = os.path.join(os.path.dirname(__file__), '../../../datasets/output_data')
        # raise error if the output_data_folder is not found
        if not os.path.exists(output_data_folder):
            raise ValueError(f"Output data folder not found at {output_data_folder}")
        # Load data from npy files
        self.output_data = np.load(os.path.join(output_data_folder, 'output_data.npy'))
        self.output_history_data = np.load(os.path.join(output_data_folder, 'output_history_data.npy'))
        self.output_macro_data = np.load(os.path.join(output_data_folder, 'output_macro_data.npy'))
        self.output_mask_data = np.load(os.path.join(output_data_folder, 'output_mask_data.npy'))
        self.output_starting_date = np.load(os.path.join(output_data_folder, 'output_starting_date.npy'), allow_pickle=True)
        self.ticker_list = np.load(os.path.join(output_data_folder, 'ticker_list.npy'))
        self.time = np.load(os.path.join(output_data_folder, 'time.npy'))
        self.original_close = np.load(os.path.join(output_data_folder, 'output_original_close.npy'))
        self.original_open = np.load(os.path.join(output_data_folder, 'output_original_open.npy'))
        self.output_adj_factor=np.load(os.path.join(output_data_folder, 'output_adj_factor.npy'))
        self.ticker_name=ticker_name
        self.obs_features=obs_features
        self.temporal_features = temporal_features

        # self.X = np.concatenate((self.output_data, self.output_macro_data), axis=2)

        # load the args.pickle from the model_path
        with open(f"{model_path}/args.pickle", "rb") as fb:
            self.args = pickle.load(fb)

        print("All data and model initialized successfully.")
        

        # Load model for inference
        if not os.path.exists(model_path):
            raise ValueError(f"Model directory not found...")

        # Load arguments and model
        with open(f"{model_path}/args.pickle", "rb") as fb:
            self.args = pickle.load(fb)

        # model = nn.DataParallel(model)

        # model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))

        # Load the state_dict
        # Ensure proper device mapping
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print(f"Using device for generator: {self.args.device}")
        state_dict = torch.load(f"{model_path}/model.pt", map_location=self.args.device)

        # state_dict = torch.load(f"{model_path}/model.pt")
        self.model=TimeGAN(self.args)

        # Remove "module." prefix from the state_dict keys
        if torch.cuda.device_count() > 1:
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v  # remove "module." from the key
                else:
                    new_state_dict[k] = v

            # Load the modified state_dict into the model
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        print("\nGenerator loaded")
        # Initialize model to evaluation mode and run without gradientspi
        # model.to(args.device)
        # Wrap the model with DataParallel if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     model = nn.DataParallel(model)

        self.model.to(self.args.device)  # Send the model to the GPU(s)
        self.model.eval()

        

    def model_inference(self,real_data, target_macro, T):
        """The get generated data from the model"""
        args=self.args
        with torch.no_grad():
            # Generate synthetic data
            noise = torch.rand((len(real_data), args.max_seq_len//2, args.Z_dim))
            # repace the second half of the real data with noise and keep the second half as it 
            Z_= np.concatenate((real_data[:,:args.max_seq_len//2,:args.Z_dim],noise),axis=1)
            Z = np.concatenate((Z_,target_macro),axis=2)


            # cast T to tensor
            T = torch.tensor(T).to(args.device)
            
            generated_data = self.model(X=None, T=T, Z=Z, obj="inference",M=None)

        return generated_data.numpy()
    
    def transform_generated_pv_feature_to_data(self, df_features, original_close, original_open, underlying_close_caj):
        """
        Revert the processed features back to the original OHLC data.
        
        Parameters:
        df_features (pd.DataFrame): DataFrame with processed features
        original_close (pd.Series): Series with the original close values
        caj_column (pd.Series): Series with the caj values
        
        Returns:
        pd.DataFrame: DataFrame with original columns ["open", "close", "high", "low", "volume"]
        """
        # Reconstruct smoothed close values from the features
        # reconstructed the close from the fiorst close value and by accumulating the close_return
        # smoothed_close = ((original_close * (1 + df_features.iloc[:,0]).cumprod()).shift(1).fillna(original_close))/caj_column
        # reconstruct ht close from the orginal close price and the close return
        # print('orginal_close: ', original_close)
        # print('orginal_volume: ', original_volume)
        # smoothed_close = (original_close.shift(1) * (1 + df_features.iloc[:,0]).shift(1)).fillna(original_close.iloc[0])
        # print("df_features shape: ", df_features.shape)
        # print("original_close: ", original_close)
        # print("original_open: ", original_open)

        smoothed_close = [original_close]

        # Calculate the subsequent synthetic prices recursively
        for t in range(len(df_features)-1):
            next_price = smoothed_close[-1] * (1 + df_features.iloc[t, 0])  # Recursive compounding
            smoothed_close.append(next_price)
        # cast the smoothed_close to dataframe
        smoothed_close = pd.DataFrame(smoothed_close)
        smoothed_close = smoothed_close.squeeze()
        # print(smoothed_close.shift(1),(df_features.iloc[:,1]+1).shift(1))
        smoothed_open = (smoothed_close.shift(1) * (df_features.iloc[:,1]+1).shift(1)).fillna(original_open)
        smoothed_high = smoothed_close * (1 + df_features.iloc[:,2])
        smoothed_low = smoothed_close * (1 + df_features.iloc[:,3])

        # print("smoothed_close shape: ", smoothed_close.shape)
        # print("smoothed_open shape: ", smoothed_open.shape)
        # print("smoothed_high shape: ", smoothed_high.shape)
        # print("smoothed_low shape: ", smoothed_low.shape)

        # log calcualtion of the smoothed close step by step
        # print("1+close_return: ", (1 + df_features.iloc[:,0]))
        # print("cumprod: ", ((original_close * (1 + df_features.iloc[:,0]).cumprod()).shift(1).fillna(original_close))/caj_column)
        # # calcualte the nan in the smoothed_close
        # print("nan in smoothed_close: ", smoothed_close.isna().sum())
        # print(smoothed_close, smoothed_open, smoothed_high, smoothed_low) 
        
        # Unsmooth the OHLC data by multiplying by caj
        close = smoothed_close
        open_ = smoothed_open 
        high = smoothed_high 
        low = smoothed_low 
        # revert the volume with volume return
        # volume = (original_volume * (1 + df_features.iloc[:,4]).cumprod()).shift(1).fillna(original_volume)
                
        volumn = np.exp(df_features.iloc[:,4])/close

        # print('close: ', close)
        # print("underlying_close_caj: ", underlying_close_caj)
        # # print shape of the underlying_close_caj
        # print("underlying_close_caj shape: ", underlying_close_caj.shape)
        # # print shape of the close
        # print("close shape: ", close.shape)

        underlying_close = underlying_close_caj*close
        
        # Combine reconstructed values into a new DataFrame
        # print("open: ", open_)
        # print("close: ", close)
        # print("high: ", high)
        # print("low: ", low)
        # print("volumn: ", volumn)
        df_original = pd.DataFrame({
            'open': open_,
            'adj_close': close,
            'high': high,
            'low': low,
            'volume': volumn,
            "close": underlying_close
        })
        
        return df_original

    # def transform_data_to_feature(self,df):
    #     """Transform the pv data to feature"""

    #     def my_rank(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #     # intermediate values
    #     df['max_oc'] = df[["open", "close"]].max(axis=1)
    #     df['min_oc'] = df[["open", "close"]].min(axis=1)
    #     # print("data type of df: {}".format(df.dtypes))
    #     df["kmid"] = (df["close"] - df["open"]) / df["close"]
    #     df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    #     df["klen"] = (df["high"] - df["low"]) / df["open"]
    #     df['kup'] = (df['high'] - df['max_oc']) / df['open']
    #     df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    #     df['klow'] = (df['min_oc'] - df['low']) / df['open']
    #     df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    #     df["ksft"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    #     df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    #     df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    #     window = [5, 10, 20, 30, 60]
    #     for w in window:
    #         df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    #     for w in window:
    #         df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    #     for w in window:
    #         df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    #     for w in window:
    #         df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    #     for w in window:
    #         df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    #     for w in window:
    #         df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    #     for w in window:
    #         df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    #     for w in window:
    #         df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    #     for w in window:
    #         df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    #     for w in window:
    #         df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    #     for w in window:
    #         df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    #     for w in window:
    #         df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    #     for w in window:
    #         shift = df['close'].shift(w)
    #         min = df["low"].where(df["low"] < shift, shift)
    #         max = df["high"].where(df["high"] > shift, shift)
    #         df["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)

    #     df['ret1'] = df['close'].pct_change(1)
    #     for w in window:
    #         df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    #     for w in window:
    #         df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    #     for w in window:
    #         df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    #     for w in window:
    #         df1 = df["close"].rolling(w)
    #         df2 = np.log(df["volume"] + 1).rolling(w)
    #         df["corr_{}".format(w)] = df1.corr(pairwise = df2)

    #     for w in window:
    #         df1 = df["close"]
    #         df_shift1 = df1.shift(1)
    #         df2 = df["volume"]
    #         df_shift2 = df2.shift(1)
    #         df1 = df1 / df_shift1
    #         df2 = np.log(df2 / df_shift2 + 1)
    #         df["cord_{}".format(w)] = df1.rolling(w).corr(pairwise = df2.rolling(w))

    #     df['abs_ret1'] = np.abs(df['ret1'])
    #     df['pos_ret1'] = df['ret1']
    #     df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    #     for w in window:
    #         df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    #     for w in window:
    #         df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    #     for w in window:
    #         df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1

    #     for w in window:
    #         df["vma_{}".format(w)] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)

    #     for w in window:
    #         df["vstd_{}".format(w)] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)

    #     for w in window:
    #         shift = np.abs((df["close"] / df["close"].shift(1) - 1)) * df["volume"]
    #         df1 = shift.rolling(w).std()
    #         df2 = shift.rolling(w).mean()
    #         df["wvma_{}".format(w)] = df1 / (df2 + 1e-12)

    #     df['vchg1'] = df['volume'] - df['volume'].shift(1)
    #     df['abs_vchg1'] = np.abs(df['vchg1'])
    #     df['pos_vchg1'] = df['vchg1']
    #     df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    #     for w in window:
    #         df["vsump_{}".format(w)] = df["pos_vchg1"].rolling(w).sum() / (df["abs_vchg1"].rolling(w).sum() + 1e-12)
    #     for w in window:
    #         df["vsumn_{}".format(w)] = 1 - df["vsump_{}".format(w)]
    #     for w in window:
    #         df["vsumd_{}".format(w)] = 2 * df["vsump_{}".format(w)] - 1

    #     df["log_volume"] = np.log(df["volume"] + 1)

    #     df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1', 'volume'], inplace=True)

    #     df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     df = df.fillna(0)

    #     # # print columns of the df
    #     # print("df columns: ", df.columns)
    #     # # save the df.columns to txt
    #     # df.columns.to_list()
    #     # # save the df.columns to txt
    #     # with open("df_columns.txt", "w") as f:
    #     #     for item in df.columns:
    #     #         f.write("%s\n" % item)
    #     # select the self.obs_features from the df
    #     df = df[self.obs_features]

    #     return df


    def transform_data_to_feature(self, df):
        """Transform the pv data to feature, optimized and corrected for rolling corr calls."""

        def my_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        # Precompute open/close max/min
        oc_vals = df[["open", "close"]].to_numpy()
        df["max_oc"] = np.maximum(oc_vals[:, 0], oc_vals[:, 1])
        df["min_oc"] = np.minimum(oc_vals[:, 0], oc_vals[:, 1])

        # Candle-related features
        df["kmid"]  = (df["close"] - df["open"]) / df["close"]
        df["kmid2"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)
        df["klen"]  = (df["high"] - df["low"])   / df["open"]
        df["kup"]   = (df["high"] - df["max_oc"]) / df["open"]
        df["kup2"]  = (df["high"] - df["max_oc"]) / (df["high"] - df["low"] + 1e-12)
        df["klow"]  = (df["min_oc"] - df["low"])  / df["open"]
        df["klow2"] = (df["min_oc"] - df["low"])  / (df["high"] - df["low"] + 1e-12)
        df["ksft"]  = (2.0 * df["close"] - df["high"] - df["low"]) / df["open"]
        df["ksft2"] = (2.0 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"] + 1e-12)

        # Returns, logs, volume changes
        df["ret1"]       = df["close"].pct_change(1)
        df["abs_ret1"]   = df["ret1"].abs()
        df["pos_ret1"]   = df["ret1"].clip(lower=0)
        df["log_volume"] = np.log(df["volume"] + 1)

        df["vchg1"]      = df["volume"] - df["volume"].shift(1)
        df["abs_vchg1"]  = df["vchg1"].abs()
        df["pos_vchg1"]  = df["vchg1"].clip(lower=0)

        # For rolling correlation with daily ratio-of-close vs. ratio-of-volume
        df["close_chg_ratio"] = df["close"] / df["close"].shift(1)
        df["vol_chg_log"]     = np.log(df["volume"] / df["volume"].shift(1) + 1)

        # Precompute boolean arrays for returns up/down
        retpos = (df["ret1"] > 0).astype(float)
        retneg = (df["ret1"] < 0).astype(float)

        windows = [5, 10, 20, 30, 60]
        for w in windows:
            # ---------- SHIFT-BASED FEATURES -----------
            shifted_close = df["close"].shift(w)
            df[f"roc_{w}"]  = shifted_close / df["close"]
            df[f"beta_{w}"] = (shifted_close - df["close"]) / (w * df["close"])

            # rsv
            shifted_ = df["close"].shift(w)
            mn_ = df["low"].where(df["low"] < shifted_, shifted_)
            mx_ = df["high"].where(df["high"] > shifted_, shifted_)
            df[f"rsv_{w}"] = (df["close"] - mn_) / (mx_ - mn_ + 1e-12)

            # ---------- ROLLING ON CLOSE -----------
            c_rolling = df["close"].rolling(w)
            df[f"ma_{w}"]   = c_rolling.mean() / df["close"]
            df[f"std_{w}"]  = c_rolling.std()  / df["close"]
            df[f"max_{w}"]  = c_rolling.max()  / df["close"]
            df[f"min_{w}"]  = c_rolling.min()  / df["close"]
            df[f"qtlu_{w}"] = c_rolling.quantile(0.8) / df["close"]
            df[f"qtld_{w}"] = c_rolling.quantile(0.2) / df["close"]
            df[f"rank_{w}"] = c_rolling.apply(my_rank) / w

            # ---------- ARGMAX/ARGMIN ON HIGH/LOW -----------
            h_rolling = df["high"].rolling(w)
            l_rolling = df["low"].rolling(w)
            df[f"imax_{w}"] = h_rolling.apply(np.argmax) / w
            df[f"imin_{w}"] = l_rolling.apply(np.argmin) / w
            df[f"imxd_{w}"] = (
                h_rolling.apply(np.argmax) - l_rolling.apply(np.argmin)
            ) / w

            # ---------- COUNT UP/DOWN -----------
            df[f"cntp_{w}"] = retpos.rolling(w).sum() / w
            df[f"cntn_{w}"] = retneg.rolling(w).sum() / w
            df[f"cntd_{w}"] = df[f"cntp_{w}"] - df[f"cntn_{w}"]

            # ---------- ROLLING CORRELATIONS -----------
            #  Fix: pass the Series directly, NOT the rolling object
            df[f"corr_{w}"] = df["close"].rolling(w).corr(df["log_volume"])
            df[f"cord_{w}"] = df["close_chg_ratio"].rolling(w).corr(df["vol_chg_log"])

            # ---------- SUMP / SUMN / SUMD -----------
            sum_abs  = df["abs_ret1"].rolling(w).sum()
            sum_pos  = df["pos_ret1"].rolling(w).sum()
            df[f"sump_{w}"] = sum_pos / (sum_abs + 1e-12)
            df[f"sumn_{w}"] = 1.0 - df[f"sump_{w}"]
            df[f"sumd_{w}"] = 2.0 * df[f"sump_{w}"] - 1.0

            # ---------- VMA / VSTD / WVMA -----------
            v_rolling = df["volume"].rolling(w)
            df[f"vma_{w}"]  = v_rolling.mean() / (df["volume"] + 1e-12)
            df[f"vstd_{w}"] = v_rolling.std()  / (df["volume"] + 1e-12)

            shift_serie = np.abs(df["close"] / df["close"].shift(1) - 1) * df["volume"]
            df1_ = shift_serie.rolling(w).std()
            df2_ = shift_serie.rolling(w).mean()
            df[f"wvma_{w}"] = df1_ / (df2_ + 1e-12)

            # ---------- VOLUME CHANGE PATTERNS -----------
            sum_abs_v = df["abs_vchg1"].rolling(w).sum()
            sum_pos_v = df["pos_vchg1"].rolling(w).sum()
            df[f"vsump_{w}"] = sum_pos_v / (sum_abs_v + 1e-12)
            df[f"vsumn_{w}"] = 1.0 - df[f"vsump_{w}"]
            df[f"vsumd_{w}"] = 2.0 * df[f"vsump_{w}"] - 1.0

        # Drop unneeded columns all at once
        df.drop(columns=[
            "max_oc", "min_oc", 
            "ret1", "abs_ret1", "pos_ret1", 
            "vchg1", "abs_vchg1", "pos_vchg1",
            "volume",           # user originally drops volume
            "close_chg_ratio",  # used for cord_{w}
            "vol_chg_log",      # used for cord_{w}
        ],
        inplace=True,
        errors="ignore")

        # Clean up + final column selection
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df = df[self.obs_features]

        return df


    def call(self,timestamp,macro_epsilon):
        """
        generate the feature to feed downstream model
        1. do generator inference to get processed_pv_feature
        2. inverse transform the processed_pv_feature to pv_data
        3. process the pv_data to get the feature for downstream model
        """
        # set the random seed with using the seed
        os.environ['PYTHONHASHSEED'] = str(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # cast the timestamp to datetime if it is not
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        # Use the timestamp to locate the index from self.output_starting_date
        # time_stamp_index = self.output_starting_date.index(timestamp)
        # print("self.output_starting_date: ", self.output_starting_date)
        # print("timestamp: ", timestamp)
        time_stamp_index = np.where(self.output_starting_date == timestamp)[0][0]

        # startdate_index=time_stamp_index-self.args.max_seq_len//2
        startdate_index=time_stamp_index-self.args.max_seq_len//2

        # find ticker index from the ticker_list
        # ticker_index=self.ticker_list.index(self.ticker_name)
        # print(self.ticker_list)
        ticker_index = np.where(self.ticker_list == self.ticker_name)[0][0]
        # print("ticker_index: ", ticker_index)
    
        # Locate the history and macro
        real_data = self.output_data[startdate_index:startdate_index+1, :, :]
        history_data = self.output_history_data[startdate_index:startdate_index+1, :, :][0]
        cal_factor = self.output_adj_factor[startdate_index:startdate_index+1, :, :]
        macro = self.output_macro_data[startdate_index:startdate_index+1, :, :]


       
        T = self.time
        T = T[startdate_index:startdate_index+1]
        # cast T to ndarray
        T = np.array(T)
        # print("T1: ", T)
        # print("T1 shape: ", T.shape)
        # print("self time: ", T)


        # use eplsion to change the macro
        target_macro=macro+macro_epsilon

        # generate the feature to feed downstream model
        # print("real_data: ", real_data.shape)
        # print("target_macro: ", target_macro.shape)
        
        start_time=time.time()
        # print("real_data preview: ", real_data)
        # print("target_macro macro preview: ", target_macro)
        processed_pv_feature_all=self.model_inference(real_data, target_macro, T)
        # print("processed_pv_feature_all preview: ", processed_pv_feature_all)
        end_time=time.time()
        # print("Time taken for generator inference: ", end_time-start_time)
        #select the ticker_index from the processed_pv_feature,cose and open
        # each ticker have 5 features in processed_pv_feature_all
        # print("processed_pv_feature all shape: ", processed_pv_feature_all.shape)
        processed_pv_feature_all=processed_pv_feature_all[0]
        processed_pv_feature_all=pd.DataFrame(processed_pv_feature_all)
        # print("processed_pv_feature all shape: ", processed_pv_feature_all.shape)

        processed_pv_feature=processed_pv_feature_all.iloc[:,ticker_index*5:(ticker_index+1)*5]
        history_data_ticker=history_data[:,ticker_index*5:(ticker_index+1)*5]

        # print("processed_pv_feature shape: ", processed_pv_feature.shape)
        # print("self.output_history_data shape: ", self.output_history_data.shape)
        # print("history_data shape: ", history_data.shape)
        # print("history_data_ticker shape: ", history_data_ticker.shape)

        # denorm the processed_pv_feature using the history_data_ticker
        history_mean = history_data_ticker.mean(axis=0)
        history_std = history_data_ticker.std(axis=0)
        # print("history_mean: ", history_mean)
        # print("history_std: ", history_std)
        processed_pv_feature = (processed_pv_feature * history_std) + history_mean

        # do the same to the real_data
        # print("real_data shape: ", real_data.shape)
        # real_data_ticker = real_data[0][:,ticker_index*5:(ticker_index+1)*5]
        # real_data_ticker = real_data_ticker.squeeze()
        # real_data_ticker = pd.DataFrame(real_data_ticker)
        # real_data_ticker = (real_data_ticker * history_std) + history_mean
        # print("denorm real_data_ticker: ", real_data_ticker)

        # print("denorm processed_pv_feature: ", processed_pv_feature)


        caj_factor_ticker=cal_factor[:,ticker_index]
        caj_factor_ticker = np.array(caj_factor_ticker)  # Ensure it's a NumPy array
        # Reshape the array
        caj_factor_ticker = caj_factor_ticker.reshape(-1)
        # print("processed_pv_feature shape: ", processed_pv_feature.shape)
        close_all = self.original_close
        open_all = self.original_open
        close=close_all[startdate_index,ticker_index]
        open=open_all[startdate_index,ticker_index]
        # cast close and open to nd array
        close=np.array(close)
        # close shape (120,1)
        # get the first value of the close
        close=close[0]
        
        # print("close: ", close)
        # print("open: ", open)
        # print("open all shape: ", open_all.shape)
        # print("close all shape: ", close_all.shape)
        # print("close shape: ", close.shape)
        # print("close", close)
        # print("open", open)
        # cast the processed_pv_feature to dataframe
        # shape of processed_pv_feature: (1, 120, 90)
        # first cast it to (120, 90)

        start_time=time.time()
        pv_data=self.transform_generated_pv_feature_to_data(processed_pv_feature, close, open,caj_factor_ticker)
        # print("pv_data preview: ", pv_data)
        # do the same to the real_data
        # real_data_ticker=self.transform_generated_pv_feature_to_data(real_data_ticker, close, open,caj_factor_ticker)
        # print("real_data_ticker preview: ", real_data_ticker)
        end_time=time.time()
        # print("Time taken for inverse transform: ", end_time-start_time)
        start = time.time()
        feature=self.transform_data_to_feature(pv_data)
        # do the same to the real_data
        # real_feature=self.transform_data_to_feature(real_data_ticker)
        # print("real_feature: ", real_feature)
        # print("feature preview: ", feature)
        end = time.time()
        # print("Time taken for feature transformation: ", end-start)


        # return the last row of the feature

        return feature
        
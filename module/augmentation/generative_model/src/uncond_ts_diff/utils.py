# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from typing import Type, Dict
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset
from pandas.tseries.frequencies import to_offset

from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.transform import (
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    MapTransformation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.model.forecast import SampleForecast
from sklearn.linear_model import LinearRegression

sns.set(
    style="white",
    font_scale=1.1,
    rc={"figure.dpi": 125, "lines.linewidth": 2.5, "axes.linewidth": 1.5},
)


def filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss"}):
    return {m: metrics[m].item() for m in select}


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)


def plot_train_stats(df: pd.DataFrame, y_keys=None, skip_first_epoch=True):
    if skip_first_epoch:
        df = df.iloc[1:, :]
    if y_keys is None:
        y_keys = ["train_loss", "valid_loss"]

    fix, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for y_key in y_keys:
        sns.lineplot(
            ax=ax,
            data=df,
            x="epochs",
            y=y_key,
            label=y_key.replace("_", " ").capitalize(),
        )
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    plt.show()


def get_lags_for_freq(freq_str: str):
    offset = to_offset(freq_str)
    if offset.n > 1:
        raise NotImplementedError(
            "Lags for freq multiple > 1 are not implemented yet."
        )
    if offset.name == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif offset.name == "D" or offset.name == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    else:
        raise NotImplementedError(
            f"Lags for {freq_str} are not implemented yet."
        )
    return lags_seq


def create_transforms(
    num_feat_dynamic_real,
    num_feat_static_cat,
    num_feat_static_real,
    time_features,
    prediction_length,
):
    remove_field_names = []
    if num_feat_static_real == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if num_feat_dynamic_real == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not num_feat_static_cat > 0
            else []
        )
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not num_feat_static_real > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            ),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            # Calculates the age of each time step relative to the start of the series and adds it as a feature. The age can be in log scale if specified, which helps in modeling the impact of time on the series more effectively.
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            # # Custom feature for Stock data
            # AddZCloseFeature(
            #     target_field=FieldName.TARGET,
            #     pred_length=prediction_length,
            #     output_field=FieldName.FEAT_DYNAMIC_REAL,
            # ),
            # Computes and adds the mean and standard deviation of the target field to the dataset. This can be useful for normalization purposes or to provide additional context to the model.
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
            #  Combines various features (time features, age, dynamic real features if present) into a single feature set by vertically stacking them. This consolidated feature set is then ready to be used by the model.
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                # # + custom_fields
                # + (
                #     [FieldName.FEAT_DYNAMIC_REAL]
                #     if num_feat_dynamic_real > 0
                #     else []
                # ),
            ),
        ]
    )


def create_transforms_indicator(
    num_feat_dynamic_real,
    num_feat_static_cat,
    num_feat_static_real,
    time_features,
    prediction_length,
    indicator="basic",
):
    remove_field_names = []
    if num_feat_static_real == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if num_feat_dynamic_real == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    chain = Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not num_feat_static_cat > 0
            else []
        )
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not num_feat_static_real > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            ),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            # Calculates the age of each time step relative to the start of the series and adds it as a feature. The age can be in log scale if specified, which helps in modeling the impact of time on the series more effectively.
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            # Custom feature for Stock data
            # (
            #     AddZCloseFeature(
            #         target_field=FieldName.TARGET,
            #         pred_length=prediction_length,
            #         output_field=FieldName.FEAT_DYNAMIC_REAL,
            #     )
            #     if indicator == "zclose"
            #     else None
            # ),
            # Computes and adds the mean and standard deviation of the target field to the dataset. This can be useful for normalization purposes or to provide additional context to the model.
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
        ]
    )
    if indicator == "basic":
        print("Basic indicator")
        chain = chain.__add__(  #  Combines various features (time features, age, dynamic real features if present) into a single feature set by vertically stacking them. This consolidated feature set is then ready to be used by the model.
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                # # + custom_fields
                # + (
                #     [FieldName.FEAT_DYNAMIC_REAL]
                #     if num_feat_dynamic_real > 0
                #     else []
                # ),
            ),
        )
    elif indicator == "lagged":
        print("Lagged indicator")
        indicator_transformation = AddZCloseFeature(
            target_field=FieldName.TARGET,
            pred_length=prediction_length,
            output_field=FieldName.FEAT_DYNAMIC_REAL,
        )
        chain = chain.__add__(indicator_transformation)
        chain = chain.__add__(  #  Combines various features (time features, age, dynamic real features if present) into a single feature set by vertically stacking them. This consolidated feature set is then ready to be used by the model.
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                # # + custom_fields
                # + (
                #     [FieldName.FEAT_DYNAMIC_REAL]
                #     if num_feat_dynamic_real > 0
                #     else []
                # ),
            ),
        )
    elif indicator == "alpha158":
        print("Alpha158 indicator")
        indicator_transformation = Alpha158(
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_DYNAMIC_REAL,
            windows=[5, 10, 20, 30, 60],
        )
        chain = chain.__add__(indicator_transformation)
        chain = chain.__add__(  #  Combines various features (time features, age, dynamic real features if present) into a single feature set by vertically stacking them. This consolidated feature set is then ready to be used by the model.
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                # # + custom_fields
                # + (
                #     [FieldName.FEAT_DYNAMIC_REAL]
                #     if num_feat_dynamic_real > 0
                #     else []
                # ),
            ),
        )
    else:
        raise ValueError("Indicator not found")

    return chain


def create_splitter(
    past_length: int,
    future_length: int,
    mode: str = "train",
    num_feat_dynamic_real: int = 0,
):
    if mode == "train":
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1,
            min_past=past_length,
            min_future=future_length,
        )
    elif mode == "val":
        instance_sampler = ValidationSplitSampler(min_future=future_length)
    elif mode == "test":
        instance_sampler = TestSplitSampler()
    if num_feat_dynamic_real > 0:
        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=future_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
                FieldName.FEAT_DYNAMIC_REAL,
            ],
        )
    else:
        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=future_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        )
    return splitter


def get_next_file_num(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """Gets the next available file number in a directory.
    e.g., if `base_fname="results"` and `base_dir` has
    files ["results-0.yaml", "results-1.yaml"],
    this function returns 2.

    Parameters
    ----------
    base_fname
        Base name of the file.
    base_dir
        Base directory where files are located.

    Returns
    -------
        Next available file number
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and x.name.startswith(base_fname),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: x.name.startswith(base_fname),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    return max(run_nums) + 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser


class Alpha158(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str = FieldName.TARGET,
        output_field: str = "zclose",
        dtype: Type = np.float32,
        windows: list = [5],
    ) -> None:
        super().__init__()
        self.target_field = target_field
        self.output_field = output_field
        self.dtype = dtype
        self.windows = windows

    def compute_alpha158_features(
        self, close_prices: pd.Series
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=close_prices.index)
        temp_indicator = pd.DataFrame({"close": close_prices})
        temp_indicator["zclose"] = (
            temp_indicator.close
            / (temp_indicator.close.rolling(2).sum() - temp_indicator.close)
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
        for w in self.windows:
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
            # temp_indicator[f"BETA{w}"] = (
            #     temp_indicator["close"]
            #     .rolling(window=w)
            #     .apply(lambda x: np.polyfit(range(w), x, 1)[0])
            #     / temp_indicator["close"]
            # )

            # RSQR
            # The R-sqaure value of linear regression for the past d days, represent the trend linear
            x = pd.Series(range(1, w + 1))
            y = temp_indicator["close"][-w:]
            x = x.values.reshape(-1, 1)
            y = y.values.reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            temp_indicator[f"RSQR{w}"] = reg.score(x, y)

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
            # temp_indicator["pos_ret1"][temp_indicator["pos_ret1"].lt(0)] = 0
            # Modified line using .loc to avoid the warning
            temp_indicator.loc[
                temp_indicator["pos_ret1"].lt(0), "pos_ret1"
            ] = 0

            # SUMP
            # The total gain / the absolute total price changed
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            # temp_indicator[f'SUMP{w}'] = temp_indicator['close'].rolling(window=w).apply(lambda x: sum(x[x > x.shift()])).fillna(0)
            temp_indicator[f"SUMP{w}"] = temp_indicator["pos_ret1"].rolling(
                w
            ).sum() / (temp_indicator["abs_ret1"].rolling(w).sum() + 1e-12)

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
            columns=["ret1", "abs_ret1", "pos_ret1", "close"], inplace=True
        )
        temp_indicator = temp_indicator.fillna(method="ffill").fillna(
            method="bfill"
        )
        features = temp_indicator
        return features

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        close = data[self.target_field]
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        features_df = self.compute_alpha158_features(close)
        features_df = features_df.to_numpy()
        # print("Features shape: ", features_df.shape)
        # Convert DataFrame to the desired output format and append to data
        data[self.output_field] = features_df.astype(self.dtype).reshape(
            (features_df.shape[1], -1)
        )

        return data


class AddZCloseFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str = FieldName.TARGET,
        output_field: str = "zclose",
        pred_length: int = 0,
        dtype: Type = np.float32,
    ) -> None:
        super().__init__()
        self.target_field = target_field
        self.output_field = output_field
        self.dtype = dtype
        self.pred_length = pred_length

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        close = data[self.target_field]
        # print(type(close), close.shape)
        temp = pd.Series(close)
        zclose = (temp / (temp.rolling(2).sum() - temp)) - 1
        zclose = zclose.to_numpy()
        windows = [5, 10, 15, 20, 25, 30, 60]
        z_features = [zclose]  # Start with zclose
        for window in windows:
            zd_feature = (temp.rolling(window).sum() / window) / temp - 1
            z_features.append(zd_feature)
        zclose = np.stack(z_features, axis=0)
        # print(zclose.shape)
        data[self.output_field] = zclose.astype(self.dtype).reshape(
            (len(zclose), -1)
        )

        return data


class AddMeanAndStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dtype: Type = np.float32,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data[self.feature_name] = np.array(
            [data[self.target_field].mean(), data[self.target_field].std()]
        )

        return data


class ScaleAndAddMeanFeature(MapTransformation):
    def __init__(
        self, target_field: str, output_field: str, prediction_length: int
    ) -> None:
        """Scale the time series using mean scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the mean feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        scale = np.mean(
            np.abs(data[self.target_field][..., : -self.prediction_length]),
            axis=-1,
            keepdims=True,
        )
        scale = np.maximum(scale, 1e-7)
        scaled_target = data[self.target_field] / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = scale

        return data


class ScaleAndAddMinMaxFeature(MapTransformation):
    def __init__(
        self, target_field: str, output_field: str, prediction_length: int
    ) -> None:
        """Scale the time series using min-max scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the min-max feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        full_seq = data[self.target_field][..., : -self.prediction_length]
        min_val = np.min(full_seq, axis=-1, keepdims=True)
        max_val = np.max(full_seq, axis=-1, keepdims=True)
        loc = min_val
        scale = np.maximum(max_val - min_val, 1e-7)
        scaled_target = (full_seq - loc) / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = (loc, scale)

        return data


def descale(data, scale, scaling_type):
    if scaling_type == "mean":
        return data * scale
    elif scaling_type == "min-max":
        loc, scale = scale
        return data * scale + loc
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")


def predict_and_descale(predictor, dataset, num_samples, scaling_type):
    """Generates forecasts using the predictor on the test
    dataset and then scales them back to the original space
    using the scale feature from `ScaleAndAddMeanFeature`
    or `ScaleAndAddMinMaxFeature` transformation.

    Parameters
    ----------
    predictor
        GluonTS predictor
    dataset
        Test dataset
    num_samples
        Number of forecast samples
    scaling_type
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Yields
    ------
        SampleForecast objects

    Raises
    ------
    ValueError
        If the predictor generates Forecast objects other than SampleForecast
    """
    forecasts = predictor.predict(dataset, num_samples=num_samples)
    for input_ts, fcst in zip(dataset, forecasts):
        scale = input_ts["scale"]
        if isinstance(fcst, SampleForecast):
            fcst.samples = descale(
                fcst.samples, scale, scaling_type=scaling_type
            )
        else:
            raise ValueError("Only SampleForecast objects supported!")
        yield fcst


def to_dataframe_and_descale(input_label, scaling_type) -> pd.DataFrame:
    """Glues together "input" and "label" time series and scales
    the back using the scale feature from transformation.

    Parameters
    ----------
    input_label
        Input-Label pair generated from the test template
    scaling_type
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        A DataFrame containing the time series
    """
    start = input_label[0][FieldName.START]
    scale = input_label[0]["scale"]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    full_target = descale(full_target, scale, scaling_type=scaling_type)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)


def make_evaluation_predictions_with_scaling(
    dataset, predictor, num_samples: int = 100, scaling_type="mean"
):
    """A customized version of `make_evaluation_predictions` utility
    that first scales the test time series, generates the forecast and
    the scales it back to the original space.

    Parameters
    ----------
    dataset
        Test dataset
    predictor
        GluonTS predictor
    num_samples, optional
        Number of test samples, by default 100
    scaling_type, optional
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        A tuple of forecast and time series iterators
    """
    window_length = predictor.prediction_length + predictor.lead_time
    _, test_template = split(dataset, offset=-window_length)
    test_data = test_template.generate_instances(window_length)
    input_test_data = list(test_data.input)

    return (
        predict_and_descale(
            predictor,
            input_test_data,
            num_samples=num_samples,
            scaling_type=scaling_type,
        ),
        map(
            partial(to_dataframe_and_descale, scaling_type=scaling_type),
            test_data,
        ),
    )


class PairDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class GluonTSNumpyDataset:
    """GluonTS dataset from a numpy array.

    Parameters
    ----------
    data
        Numpy array of samples with shape [N, T].
    start_date, optional
        Dummy start date field, by default pd.Period("2023", "H")
    """

    def __init__(
        self, data: np.ndarray, start_date: pd.Period = pd.Period("2023", "H")
    ):
        self.data = data
        self.start_date = start_date

    def __iter__(self):
        for ts in self.data:
            item = {"target": ts, "start": self.start_date}
            yield item

    def __len__(self):
        return len(self.data)


class MaskInput(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        observed_field: str,
        context_length: int,
        missing_scenario: str,
        missing_values: int,
        dtype: Type = np.float32,
    ) -> None:
        # FIXME: Remove hardcoding of fields
        self.target_field = target_field
        self.observed_field = observed_field
        self.context_length = context_length
        self.missing_scenario = missing_scenario
        self.missing_values = missing_values
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = deepcopy(data)
        data["orig_past_target"] = data["past_target"].copy()
        if self.missing_scenario == "BM-E" and self.missing_values > 0:
            data["past_target"][-self.missing_values :] = 0
            data["past_observed_values"][-self.missing_values :] = 0
        elif self.missing_scenario == "BM-B" and self.missing_values > 0:
            data["past_target"][
                -self.context_length : -self.context_length
                + self.missing_values
            ] = 0
            data["past_observed_values"][
                -self.context_length : -self.context_length
                + self.missing_values
            ] = 0
        elif self.missing_scenario == "RM" and self.missing_values > 0:
            weights = torch.ones(self.context_length)
            missing_idxs = -self.context_length + torch.multinomial(
                weights, self.missing_values, replacement=False
            )
            data["past_target"][missing_idxs] = 0
            data["past_observed_values"][missing_idxs] = 0
        return data


class ConcatDataset:
    def __init__(self, test_pairs, axis=-1) -> None:
        self.test_pairs = test_pairs
        self.axis = axis

    def _concat(self, test_pairs):
        for t1, t2 in test_pairs:
            yield {
                "target": np.concatenate(
                    [t1["target"], t2["target"]], axis=self.axis
                ),
                "start": t1["start"],
            }

    def __iter__(self):
        yield from self._concat(self.test_pairs)

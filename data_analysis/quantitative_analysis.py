import ast
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.tsa.stattools as ts
from tqdm import tqdm

# Configure logging to write to a file, including the timestamp.
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')
# add root directory to the path
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)
print(sys.path)

# KPSS Test
def kpss_test(series):
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
    The null hypothesis of the test is that the series is stationary.

    Parameters:
    series (array-like): The time series data.

    Returns:
    tuple: test statistic, p-value, number of lags, critical values
    """
    statistic, p_value, lags, critical_values = ts.kpss(series, 'c')
    return statistic, p_value, critical_values

# Hurst Exponent
def hurst_exponent(time_series):
    """
    Calculates the Hurst Exponent of the time series.

    The Hurst Exponent is a measure of the long-term memory of the time series.
    It helps in understanding whether a series is a random walk (H ~ 0.5),
    mean-reverting (H < 0.5), or trending (H > 0.5).

    Parameters:
    time_series (array-like): The time series data.

    Returns:
    float: The Hurst Exponent
    """
    lags = range(2, 100)
    tau = [np.sqrt(
        np.std(np.subtract(time_series[lag:].reset_index(drop=True), time_series[:-lag].reset_index(drop=True)))) for
           lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

# Variance Ratio
def variance_ratio(series, lag=2):
    """
    Computes the variance ratio for a given time series.

    The variance ratio test is used to test for the presence of a random walk in a time series.
    A variance ratio significantly different from 1 indicates deviation from a random walk.

    Parameters:
    series (array-like): The time series data.
    lag (int): The lag at which to compute the variance ratio.

    Returns:
    float: The variance ratio
    """
    diffs = np.diff(series)
    var_diff = np.var(diffs)

    if var_diff == 0:
        print("Variance of differences is zero.")
        return 0

    cumulative_diffs = series[lag:].reset_index(drop=True) - series[:-lag].reset_index(drop=True)
    var_cumulative_diffs = np.var(cumulative_diffs) / lag

    # Debugging prints
    print("Differences Variance:", var_diff)
    print("Cumulative Differences Variance:", var_cumulative_diffs)

    return var_cumulative_diffs / var_diff

# Autocorrelation
def autocorrelation(series, lag=1):
    """
    Calculates the autocorrelation of the given time series at specified lag.

    Autocorrelation measures the correlation of a time series with its own lagged values.
    It helps in identifying the presence of trends or seasonal patterns.

    Parameters:
    series (array-like): The time series data.
    lag (int): The lag at which to compute the autocorrelation.

    Returns:
    float: The autocorrelation value
    """
    return series.autocorr(lag)

def run_all_tests(series):
    """
    Runs all the time series analysis tests on the given series.

    Parameters:
    series (array-like): The time series data.

    Returns:
    dict: A dictionary containing the results of all the tests.
    """
    # preprocess the series
    # only keep numeric columns
    if isinstance(series, pd.DataFrame):
        series = series.select_dtypes(include=[np.number])
    series = series.dropna()
    return {
        "kpss_test": kpss_test(series),
        "hurst_exponent": hurst_exponent(series),
        "variance_ratio": variance_ratio(series),
        "autocorrelation": autocorrelation(series)
    }


def plot_res(target_res_file):
    # Read the results file
    df = pd.read_csv(target_res_file)

    # Read the asset names from the config file
    asset_config_path = "/home/hcxia/HistoryIsNotEough/configs/_asset_list_/dj30.txt"
    with open(asset_config_path, 'r') as f:
        assets = f.read().splitlines()

    # Processing the KPSS test results (assuming it's a string representation of a tuple)
    df['kpss_test_statistic'] = df['kpss_test'].apply(lambda x: ast.literal_eval(x)[0])

    # Convert 'dataset' into a format that can be used as categorical labels in plotting
    df['dataset'] = df['dataset'].apply(lambda x: x.split('.')[0])  # Removes file extension for clarity

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    for ax, column in zip(axes.flatten(),
                          ['kpss_test_statistic', 'hurst_exponent', 'variance_ratio', 'autocorrelation']):
        plt.sca(ax)
        plt.xticks(rotation=90)
        # plot the points that are not in the asset in default color
        sns.scatterplot(data=df[~df['dataset'].apply(lambda x: x.split("_")[1]).isin(assets)], x='dataset', y=column,
                        ax=ax)
        # overlay points that are in assets with a different color
        sns.scatterplot(data=df[df['dataset'].apply(lambda x: x.split("_")[1]).isin(assets)], x='dataset', y=column,
                        ax=ax, color='red')

    # Adding titles and labels
    axes[0, 0].set_title('KPSS Test Statistic')
    axes[0, 1].set_title('Hurst Exponent')
    axes[1, 0].set_title('Variance Ratio')
    axes[1, 1].set_title('Autocorrelation')

    # Rotate x-labels for better visibility
    for ax in axes.flatten():
        plt.sca(ax)
        plt.xticks(rotation=90)

    # Adjust the layout
    plt.tight_layout()

    # save the plot to the same name as the results file
    plt.savefig(target_res_file.replace('.csv', '.png'))


workdir="workdir"
data_dir="../datasets"
quantitative_analysis_dir=os.path.join(workdir,"quantitative_analysis")
seed=42
def main():
    # set random seed
    np.random.seed(seed)
    # TSNE
    res_dir = os.path.join(quantitative_analysis_dir, "statistics")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    target_res_file = os.path.join(res_dir, "target_statistics.csv")
    # do recursive search for .parquet and .csv files in the data_dir
    for root, dirs, files in tqdm(os.walk(data_dir), desc="Calculating Statistics"):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(".parquet"):
                df = pd.read_parquet(full_path)
            elif file.endswith(".csv"):
                df = pd.read_csv(full_path)
            else:
                print(f"Skipping {file}")
                continue
            try:
                # select the target column
                # the target label
                # Benchmark Datasets: "OT"
                # Price Forecasting(Regression): "close"
                # Return Forecasting(Regression): "ret1"
                # Trend Forecasting(Classification): "mov1"
                target_columns = ["OT", "ret1"]
                for target_column in target_columns:
                    try:
                        target = df.loc[:, target_column]
                        x = df.drop(columns=[target_column])
                        print("len(target)", len(target))
                        target_res = run_all_tests(target)
                        # add results to the statistics file
                        target_res["dataset"] = target_column + "_" + file
                        target_res_df = pd.DataFrame([target_res])
                        if not os.path.exists(target_res_file):
                            target_res_df.to_csv(target_res_file, index=False)
                        else:
                            target_res_df.to_csv(target_res_file, mode='a', header=False, index=False)

                    except Exception as e:
                        logging.error(f"Error calculating {e}", exc_info=True)

                # # for each column in x, calculate the statistics and write to file
                # # parse filename from file path
                # filename = os.path.basename(full_path)
                # # create a directory for the results
                # if not os.path.exists(os.path.join(res_dir, filename)):
                #     os.makedirs(os.path.join(res_dir, filename))
                # x_res_file = os.path.join(res_dir, filename, f"columns_res.csv")
                # for col in x.columns:
                #     try:
                #         x_res = run_all_tests(x[col])
                #         x_res["column"] = col
                #         x_res_df = pd.DataFrame([x_res])
                #         if not os.path.exists(x_res_file):
                #             x_res_df.to_csv(x_res_file, index=False)
                #         else:
                #             x_res_df.to_csv(x_res_file, mode='a', header=False, index=False)
                #     except Exception as e:
                #         logging.error(f"Error calculating column {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Error calculating {file}: {e}", exc_info=True)
                continue
        # plot the results
    plot_res(target_res_file)


if __name__ == '__main__':
    main()
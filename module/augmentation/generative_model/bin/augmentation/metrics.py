import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def ARR(ret):
    res = (np.cumprod(ret + 1.0)[-1] - 1.0) / ret.shape[0] * 252
    return res


def VOL(ret):
    res = np.std(ret)
    return res


def DD(ret):
    res = np.std(ret[np.where(ret < 0, True, False)])
    return res


def MDD(ret):
    iter_ret = np.cumprod(ret + 1.0)
    peak = iter_ret[0]
    mdd = 0
    for value in iter_ret:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def SR(ret):
    res = 1.0 * np.mean(ret) * np.sqrt(ret.shape[0]) / np.std(ret)
    return res


def CR(ret, mdd):
    res = np.mean(ret) * 252 / mdd
    return res


def SOR(ret, dd):
    res = 1.0 * np.mean(ret) * 252 / dd
    return res


def get_metrics(real_prices, predicted_prices):
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(real_prices, predicted_prices)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(real_prices, predicted_prices)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate R-squared (R²) score
    r2 = r2_score(real_prices, predicted_prices)

    # Optionally, calculate Mean Absolute Percentage Error (MAPE) if relevant
    mape = (
        np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
    )

    return mae, mse, rmse, r2, mape


def error_metrics(real_prices, predicted_prices):
    # Calculate the error terms
    errors = predicted_prices - real_prices

    # Calculate the standard deviation of the error terms
    std_errors = np.std(errors)

    # Calculate the mean of the error terms
    mean_errors = np.mean(errors)

    # Calculate the median of the error terms
    median_errors = np.median(errors)

    # Calculate the maximum and minimum of the error terms
    max_error = np.max(errors)
    min_error = np.min(errors)

    # Optionally, calculate the interquartile range (IQR) of the error terms
    q75, q25 = np.percentile(errors, [75, 25])
    iqr_errors = q75 - q25
    return (
        std_errors,
        mean_errors,
        median_errors,
        max_error,
        min_error,
        iqr_errors,
    )


def error_distribution_plot(real_prices, predicted_prices, log_dir):
    errors = predicted_prices - real_prices
    print(errors.shape)
    # Create a histogram of the error terms
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.title("Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.savefig(f"{log_dir}/error_distribution.png")
    plt.close()


def distribution_metrics(series1, series2):
    # Calculate the Kolmogorov-Smirnov statistic
    ks_stat, ks_p_value = stats.ks_2samp(series1, series2)
    # Calculate the Anderson-Darling statistic
    # Note: Anderson-Darling test in scipy.stats is primarily for comparing a sample against a known distribution.
    # There isn't a direct two-sample Anderson-Darling test in scipy, but you can perform the test for each series against a normal distribution as an example.
    # ad_stat1, critical_values1, significance_level1 = stats.anderson(
    #     series1, dist="norm"
    # )


def mutual_info_score(target, samples):
    from sklearn.feature_selection import mutual_info_regression

    return mutual_info_regression(samples, target)

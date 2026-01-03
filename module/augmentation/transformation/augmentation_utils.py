import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from optuna.samplers import TPESampler
from pykalman import KalmanFilter
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import savgol_filter
from scipy.signal import welch
from scipy.stats import entropy
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
# from tcn import TCN  # If you have the tcn p /ackage installed
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import resample
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

# ### Initialize Parameters

# Assuming cfg is a configuration object with a seed attribute
cfg = type('config', (object,), {'seed': 42})


# ### Helper Functions

def z_score_normalize(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std


def denormalize_z_score(normalized_series, original_mean, original_std):
    return (normalized_series * original_std) + original_mean

def plot_correlation(df):
    correlation_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Adjust the plot as needed
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap

    # Show the plot
    plt.show()

def cut_mix(df1, df2, alpha=0.2):
    np.random.seed(42)

    assert df1.shape == df2.shape
    size = len(df1)
    cut_point = np.random.randint(0, size, )
    cut_length = int(size * alpha)

    mixed_df = df1.copy()
    mixed_df.iloc[cut_point:cut_point + cut_length] = df2.iloc[cut_point:cut_point + cut_length]

    return mixed_df

def binary_mix(data1, data2, alpha=0.2):
    np.random.seed(42)

    assert len(data1) == len(data2)
    size = data1.shape
    mask = np.random.binomial(1, alpha, size=size).astype(bool)

    mixed_data = np.where(mask, data1, data2)

    return pd.DataFrame(mixed_data, columns=data1.columns)


def linear_mix(data1, data2, alpha=0.2):
    assert len(data1) == len(data2)

    mixed_data = alpha * data1 + (1 - alpha) * data2

    return mixed_data


def geometric_mix(data1, data2, alpha=0.2):
    assert len(data1) == len(data2)

    mixed_data = data1 ** alpha * data2 ** (1 - alpha)

    return mixed_data

def amplitude_mix(data1, data2, alpha=0.2):
    assert len(data1) == len(data2)

    # Apply Fourier Transform to each column
    fft1 = np.fft.rfft(data1, axis=0)
    fft2 = np.fft.rfft(data2, axis=0)

    # Mix the magnitudes
    magnitude1 = np.abs(fft1)
    magnitude2 = np.abs(fft2)
    mixed_magnitude = alpha * magnitude1 + (1 - alpha) * magnitude2

    # Keep the phase of the first data
    phase1 = np.angle(fft1)
    mixed_fft = mixed_magnitude * np.exp(1j * phase1)

    # Perform the inverse FFT and ensure the result is two-dimensional
    mixed_data = np.fft.irfft(mixed_fft, axis=0)
    if mixed_data.ndim == 1:
        mixed_data = mixed_data.reshape(-1, 1)  # Reshape if the data is one-dimensional

    # Return a DataFrame with the same column names as data1
    return pd.DataFrame(mixed_data, columns=data1.columns)


### PROPOSE TECHNIQUE BELOW

def proposed_mixup(df1, df2, threshold=0.1, alpha=0.5):
    def proposed_mixup_feature(data1, data2, threshold, alpha):

        def get_significant_frequencies(data, threshold):
            """
            Perform Fourier Transform on data and identify frequencies with significant amplitude.

            Args:
            - data: Time series data.
            - threshold: Threshold for significance, relative to the max amplitude.

            Returns:
            - significant_freq: Frequencies with significant amplitude.
            - significant_ampl: Amplitude of the significant frequencies.
            - full_spectrum: Full Fourier spectrum for all frequencies.
            """
            # Perform Fourier Transform
            spectrum = rfft(data)
            frequencies = rfftfreq(data.size, d=1)  # Assuming unit time interval between samples

            # Find significant amplitudes
            amplitude = np.abs(spectrum)
            significant_indices = amplitude > (amplitude.max() * threshold)
            significant_freq = frequencies[significant_indices]
            significant_ampl = amplitude[significant_indices]

            return significant_freq, significant_ampl, spectrum

        def phase_mixup(sig_freq1, sig_ampl1, spectrum1, sig_freq2, sig_ampl2, spectrum2, alpha):
            mixed_spectrum = np.copy(spectrum1)
            freqs1 = rfftfreq(spectrum1.size, d=1)
            freqs2 = rfftfreq(spectrum2.size, d=1)

            for freq in sig_freq1:
                index1 = np.argmin(np.abs(freqs1 - freq))
                index2 = np.argmin(np.abs(freqs2 - freq))

                if index1 >= len(sig_ampl1) or index2 >= len(sig_ampl2):
                    continue  # Skip the frequency if the index is out of bounds

                phase1 = np.angle(spectrum1[index1])
                phase2 = np.angle(spectrum2[index2])

                phase_diff = (phase2 - phase1) % (2 * np.pi)
                phase_diff = phase_diff - 2 * np.pi if phase_diff > np.pi else phase_diff

                new_amplitude = alpha * sig_ampl1[index1] + (1 - alpha) * sig_ampl2[index2]
                new_phase = phase1 + alpha * phase_diff

                mixed_spectrum[index1] = new_amplitude * np.exp(1j * new_phase)

            return mixed_spectrum

        def reconstruct_time_series(mixed_spectrum):
            """
            Reconstruct time series from mixed spectrum using inverse Fourier Transform.

            Returns:
            - mixed_time_series: The reconstructed time series.
            """
            # Perform inverse Fourier Transform
            mixed_time_series = irfft(mixed_spectrum)

            return mixed_time_series

        # Step 1: Get significant frequencies and amplitude for both time series
        sig_freq1, sig_ampl1, spectrum1 = get_significant_frequencies(data1, threshold)
        sig_freq2, sig_ampl2, spectrum2 = get_significant_frequencies(data2, threshold)

        # Step 2: Identify significant frequencies (already done in step 1)

        # Step 3: Phase and Magnitude Mixup
        mixed_spectrum = phase_mixup(sig_freq1, sig_ampl1, spectrum1, sig_freq2, sig_ampl2, spectrum2, alpha)

        # Step 4: Reconstruction of the time series
        mixed_time_series = reconstruct_time_series(mixed_spectrum)
        return mixed_time_series

    output_df = pd.DataFrame()

    for feature in df1.columns:
        output_df[feature] = proposed_mixup_feature(df1[feature].values, df2[feature].values, threshold, alpha)

    return output_df

def apply_mixing(df1, df2, method, alpha=0.2):
    if method == 'cut_mix':
        df = cut_mix(df1, df2, alpha)
    elif method == 'binary_mix':
        df = binary_mix(df1, df2, alpha)
    elif method == 'linear_mix':
        df = linear_mix(df1, df2, alpha)
    elif method == 'geometrix_mix':
        df = geometric_mix(df1, df2, alpha)
    elif method == 'amplitude_mix':
        df = amplitude_mix(df1, df2, alpha)
    elif method == 'proposed_mix':
        df = proposed_mixup(df1, df2, alpha)

    # Original
    else:
        df = df1.copy()

    return df

def plot_TSNE(df1, df2):
    df1.columns = df1.columns.astype(str)
    df2.columns = df2.columns.astype(str)

    df1_log = np.log(df1 + 1)  # Adding 1 to avoid log(0)
    df2_log = np.log(df2 + 1)

    combined_data = pd.concat([df1_log, df2_log])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=1000, init='pca')
    tsne_results = tsne.fit_transform(combined_data)

    # Now we split the t-SNE results back into original and augmented parts
    tsne_df1 = tsne_results[:len(df1), :]
    tsne_df2 = tsne_results[len(df1):, :]

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_df1[:, 0], tsne_df1[:, 1], label='Original', alpha=0.5)
    plt.scatter(tsne_df2[:, 0], tsne_df2[:, 1], label='Augmented', alpha=0.5)
    plt.legend()
    plt.show()


def calculate_entropy(variable):
    value, counts = np.unique(variable, return_counts=True)
    return entropy(counts, base=2)

# Function to calculate normalized mutual information
def calculate_normalized_mi(variable_1, variable_2):
    mi = mutual_info_score(variable_1, variable_2)
    entropy_1 = calculate_entropy(variable_1)
    entropy_2 = calculate_entropy(variable_2)
    # Normalizing by the average entropy
    normalized_mi = mi / ((entropy_1 + entropy_2) / 2)
    return normalized_mi


def calculate_MI(original, augmented):
    # Assuming df_original and df_augmented are your dataframes
    for column in original.columns:
        # Ensure the data is in the correct format, e.g., continuous or discrete
        # For continuous variables, you'd typically bin them before calculating mutual information
        original_data = original[column].to_numpy()
        augmented_data = augmented[column].to_numpy()

        # Calculate normalized MI for each column
        normalized_mi = calculate_normalized_mi(original_data, augmented_data)
        print(f'Normalized Mutual Information for {column}: {normalized_mi}')

def apply_filter(df,method):
    if method == 'lowess':
        df = apply_lowess_smoothing(df)
    elif method == 'savgol':
        df = apply_savgol_filter(df)
    elif method == 'kalman':
        df = apply_kalman_filter(df)
    else:
        print(f'Method {method} not recognized. Returning original data.')
    return df


def apply_lowess_smoothing(df, frac=0.1):
    smoothed_data = pd.DataFrame(index=df.index)

    # Apply LOWESS to each column
    for column in df.columns:
        smoothed_values = lowess(df[column], df.index, frac=frac, return_sorted=False)
        smoothed_data[column] = smoothed_values

    return smoothed_data


def apply_savgol_filter(df, window_length=21, polyorder=3):
    filtered_data = pd.DataFrame(index=df.index)

    # Apply Savitzky-Golay filter to each column
    for column in df.columns:
        filtered_values = savgol_filter(df[column], window_length, polyorder)
        filtered_data[column] = filtered_values

    return filtered_data


def apply_kalman_filter(df):
    filtered_data = pd.DataFrame(index=df.index)

    for column in df.columns:
        # Set up the Kalman Filter
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

        # Use observed data to estimate parameters
        kf = kf.em(df[column].values, n_iter=5)

        # Apply the Kalman Filter to smooth data
        (filtered_state_means, _) = kf.filter(df[column].values)

        # Store the filtered data in a DataFrame
        filtered_data[column] = filtered_state_means.flatten()

    return filtered_data

def compare_distributions(original_df, augmented_df, features):
    num_features = len(features)
    fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=(15, 5 * num_features))

    for i, feature in enumerate(features):
        # Original Data
        sns.histplot(original_df[feature], kde=True, color="blue", ax=axes[i, 0])
        axes[i, 0].set_title(f'Original {feature}')

        # Augmented Data
        sns.histplot(augmented_df[feature], kde=True, color="orange", ax=axes[i, 1])
        axes[i, 1].set_title(f'Augmented {feature}')

    plt.tight_layout()
    plt.show()

def jittering(ts, noise_level=0.05):
    np.random.seed(42)
    noise = np.random.normal(loc=0, scale=noise_level, size=len(ts))
    return pd.Series(ts + noise)


def flipping(ts):
    return pd.Series(np.flip(ts))


def scaling(ts, scaling_factor=1.5):
    return pd.Series(ts * scaling_factor)


def magnitude_warping(ts, sigma=0.2, knot=4):
    np.random.seed(42)
    from scipy.interpolate import CubicSpline
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    indices = np.linspace(0, len(ts) - 1, num=knot + 2)
    sp = CubicSpline(indices, random_warps)
    warp_values = sp(np.arange(len(ts)))
    return pd.Series(ts * warp_values)

def permutation(ts, n_segments=5):
    np.random.seed(42)
    permutated_ts = np.copy(ts)
    segments = np.array_split(permutated_ts, n_segments)
    np.random.shuffle(segments)
    return pd.Series(np.concatenate(segments))


def time_warping(ts, sigma=0.2, knot=4):
    np.random.seed(42)
    from scipy.interpolate import CubicSpline
    time_steps = np.arange(ts.shape[0])
    random_steps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, ts.shape[1]))
    indices = np.linspace(0, len(ts) - 1, num=knot + 2)
    sp = CubicSpline(indices, random_steps)
    warp_values = sp(time_steps)
    return pd.Series(warp_values * ts)


def stl_augment(data, period=61):
    ts = data.asfreq('B')
    ts = ts.interpolate()
    # Apply STL decomposition
    stl = STL(ts, seasonal=period)
    result = stl.fit()
    seasonal, trend, remainder = result.seasonal, result.trend, result.resid
    bootstrapped_remainder = resample(remainder, replace=True, n_samples=len(remainder), random_state=42)
    bootstrapped_remainder.index = ts.index
    augmented_signal = trend + seasonal + bootstrapped_remainder
    augmented_signal = np.maximum(augmented_signal, 0)
    augmented_signal = augmented_signal[data.index]
    return augmented_signal


# Function to plot original and augmented series
def plot_augmented_ts(original_ts, augmented_ts, title='Time Series Augmentation'):
    augmented_ts.index = original_ts.index
    plt.figure(figsize=(14, 6))
    plt.plot(original_ts, label='Original')
    plt.plot(augmented_ts, label='Augmented')
    plt.title(title)
    plt.legend()
    plt.show()


# ### AAPL

# In[16]:

# In[17]:

def apply_augmentation(stock_df,augmentation_config):
    # filter the data
    filtered_stock_df = apply_filter(stock_df,augmentation_config['filter'])
    # apply transformation
    transformed_stock_df = apply_filter(filtered_stock_df,augmentation_config['transformation'])
    # apply mixing
    mixed_stock_df = apply_mixing(transformed_stock_df,augmentation_config['mixing'])

    return mixed_stock_df

def apply_transformation(stock_df, method):
    augmented_stockdf = pd.DataFrame()
    for col in stock_df.columns:
        val = stock_df[col]
        if method == 'jittering':
            aug_val = jittering(val)
        elif method == 'flipping':
            aug_val = flipping(val)
        elif method == 'scaling':
            aug_val = scaling(val)
        elif method == 'magnitude_warping':
            aug_val = magnitude_warping(val)
        elif method == 'permutation':
            aug_val = permutation(val)
        elif method == 'stl_augment':
            aug_val = stl_augment(val)

        augmented_stockdf[col] = aug_val
    return augmented_stockdf


# In[18]:


# iloc[1:] to match returns size
# lowess = apply_lowess_smoothing(aapl)
# cut_lowess = apply_mixing(aapl, lowess, method='cut_mix')
# binary_lowess = apply_mixing(aapl, lowess, method='binary_mix')
# linear_lowess = apply_mixing(aapl, lowess, method='linear_mix')
# geometric_lowess = apply_mixing(aapl, lowess, method='geometrix_mix')
# amplitude_lowess = apply_mixing(aapl, lowess, method='amplitude_mix')
# proposedmix_lowess = apply_mixing(aapl, lowess, method='proposed_mix')
#
# jittered_ts = apply_transformation(aapl, 'jittering')
# flipped_ts = apply_transformation(aapl, 'flipping')
# scaled_ts = apply_transformation(aapl, 'scaling')
# mag_warped_ts = apply_transformation(aapl, 'magnitude_warping')
# permuted_ts = apply_transformation(aapl, 'permutation')
# stl_ts = apply_transformation(aapl, 'stl_augment')

# augmented_datasets = {
#     'cut_mix': cut_lowess,
#     'binary_mix': binary_lowess,
#     'linear_mix': linear_lowess,
#     'geometric_mix': geometric_lowess,
#     'amplitude_mix': amplitude_lowess,
#     'proposed_mix': proposedmix_lowess,
#     'jittering': jittered_ts,
#     'flipping': flipped_ts,
#     'scaling': scaled_ts,
#     'magnitude_warping': mag_warped_ts,
#     'permutation': permuted_ts,
#     'stl_augment': stl_ts
# }

# Create a MultiIndex DataFrame
# keys = augmented_datasets.keys()
# multi_index_df = pd.concat(augmented_datasets.values(), keys=keys, axis=1)

# In[19]:


# # Check the plots of original vs
# time_series_original = aapl['close']
# plot_augmented_ts(time_series_original, cut_lowess.close, title='Cut_lowess Augmentation')
# plot_augmented_ts(time_series_original, binary_lowess.close, title='Binary_lowess Augmentation')
# plot_augmented_ts(time_series_original, linear_lowess.close, title='Linear_lowess Augmentation')
# plot_augmented_ts(time_series_original, geometric_lowess.close, title='Geometric_lowess Augmentation')
# plot_augmented_ts(time_series_original, amplitude_lowess.close, title='Amplitude_lowess Augmentation')
# plot_augmented_ts(time_series_original, proposedmix_lowess.close, title='Proposedmix_lowess Augmentation')

# plot_augmented_ts(time_series_original, jittered_ts.close, title='Jittering Augmentation')
# plot_augmented_ts(time_series_original, flipped_ts.close, title='Flipped Augmentation')
# plot_augmented_ts(time_series_original, scaled_ts.close, title='Scaled Augmentation')
# plot_augmented_ts(time_series_original, mag_warped_ts.close, title='Magnitude Warped Augmentation')
# plot_augmented_ts(time_series_original, permuted_ts.close, title='Permuted Augmentation')
# plot_augmented_ts(time_series_original, stl_ts.close, title='STL Augmentation')


# ### Rules and tests with the original
#
# To determine how much information is gone. we will use the following
# - Mutual Information
# Factors to Consider
# Data Granularity: The frequency of your data (e.g., daily, weekly, monthly) affects how finely you might want to discretize it. More frequent data might benefit from more bins, but too many bins can lead to overfitting or spurious relationships.
#     1. Volatility: Financial data can be highly volatile. More bins can capture nuances in volatile periods, but they should not capture noise as signal.
#     2. Distribution Shape: The distribution of returns or price changes in financial data often deviates from normal, with heavy tails. A strategy that adapts to the data distribution can be more informative.
# - KS Test
# - Spectral Analysis
# - MACD

# In[21]:


def calculate_ema(prices, span):
    return prices.ewm(span=span, adjust=False).mean()


def calculate_macd(close_prices):
    ema_12 = calculate_ema(close_prices, 12)
    ema_26 = calculate_ema(close_prices, 26)
    macd = ema_12 - ema_26
    signal_line = calculate_ema(macd, 9)
    return macd, signal_line


def calculate_rsi(close_prices, periods=14):
    delta = close_prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def discretize_series(series, n_bins=20):
    # Assuming 'series' is a pandas Series
    series_reshaped = series.values.reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized = discretizer.fit_transform(series_reshaped).flatten()
    return discretized


def augmentation_plots(original, augmented, working_dir):
    # Distributional Analysis
    ks_stat, ks_pvalue = ks_2samp(original.close, augmented.close)

    # Spectral Analysis
    freqs, power_original = welch(original.close)
    _, power_jittered = welch(augmented.close)
    # Compare power_original and power_jittered

    # Mutual Information
    original_discretized = discretize_series(original.close)
    augmented_discretized = discretize_series(augmented.close)
    mi_score = mutual_info_score(original_discretized, augmented_discretized)

    # MACD and RSI visual inspection
    macd, signal_line = calculate_macd(original.close)
    rsi = calculate_rsi(original.close)
    augmented_macd, augmented_signal_line = calculate_macd(augmented.close)
    augmented_rsi = calculate_rsi(augmented.close)

    # Output results
    print(f"KS Statistic: {ks_stat}, P-value: {ks_pvalue}")
    print(f"Mutual Information Score: {mi_score}")

    results = {
        'KS Statistic': ks_stat,
        'P-value': ks_pvalue,
        'Mutual Information Score': mi_score,
        # Include any other metrics you calculate
    }

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(macd, label='Original MACD')
    plt.plot(signal_line, label='Original Signal Line')
    plt.plot(augmented_macd, label='Augmented MACD', linestyle='--')
    plt.plot(augmented_signal_line, label='Augmented Signal Line', linestyle='--')
    plt.title("MACD Comparison")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(rsi, label='Original RSI')
    plt.plot(augmented_rsi, label='Augmented RSI', linestyle='--')
    plt.title("RSI Comparison")
    plt.legend()

    # plt.show()
    # save the res and plots to the workdir
    plt.savefig(working_dir + 'augmentation_plots.png')
    # save results to a csv
    pd.DataFrame(results, index=[0]).to_csv(working_dir + 'augmentation_results.csv', index=False)

    return results

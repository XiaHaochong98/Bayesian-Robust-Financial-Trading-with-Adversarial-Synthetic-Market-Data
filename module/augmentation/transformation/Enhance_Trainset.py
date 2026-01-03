#!/usr/bin/env python
# coding: utf-8

# ## Project Title: To determine if data augmentation using the method proposed in 'Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning' will lead to better 1 day prediction results.
# 
# 

# In[1]:


import numpy as np
import random
import random as python_random
import os
import pandas as pd
import tensorflow as tf
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.fft import rfft, rfftfreq, irfft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from sklearn.utils import resample
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from IPython.display import display, HTML
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings 

# Display and warnings settings
display(HTML("<style>.container { width:100% !important; }</style>"))
warnings.filterwarnings("ignore")

# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler


# ### Initialize parameters

# In[2]:


# Seed value
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)


# In[3]:


# Define constants
TIME_STEPS = 20
alpha = 0.8
seq_len = 20
test_size = 0.3


# ### Helper Functions

# #### Get data and engineer them

# In[4]:


# Function to import stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def z_score_normalize(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

def denormalize_z_score(normalized_series, original_mean, original_std):
    return (normalized_series * original_std) + original_mean

# Function to create model (make sure this is defined in your environment)
def create_model(best_params, input_shape):
    model = Sequential()
    model.add(LSTM(best_params['lstm_units'], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(best_params['dropout_rate']))
    model.add(LSTM(best_params['lstm_units']))  # Stacking LSTM for deep learning
    model.add(Dropout(best_params['dropout_rate']))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')
    return model

def engineer_features(data):
    df = data.copy(deep=True)
    delta = df['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.abs().rolling(window=14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # Volume Weighted Average Price (VWAP)
    vwap = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['VWAP'] = vwap

    # Price Ratios
    df['high_to_low_ratio'] = df['High'] / df['Low']
    df['open_to_close_ratio'] = df['Open'] / df['Close']

    # Volatility
    df['volatility_10'] = df['Close'].rolling(window=10).std()

    df1 = df.drop(columns=['Open', 'High', 'Low', 'Adj Close']).dropna()
    return df1


# #### Create plots

# In[5]:


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


# In[6]:


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
    plt.figure(figsize=(12,8))
    plt.scatter(tsne_df1[:, 0], tsne_df1[:, 1], label='Original', alpha=0.5)
    plt.scatter(tsne_df2[:, 0], tsne_df2[:, 1], label='Augmented', alpha=0.5)
    plt.legend()
    plt.show()


# #### Tests for augmented datasets

# In[7]:


from sklearn.feature_selection import mutual_info_regression
def calculate_entropy(variable):
    value,counts = np.unique(variable, return_counts=True)
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


# In[8]:


def find_cointegrated_pairs(data):
    n = data.shape[1] # Number of columns in dataset
    score_matrix = np.zeros((n,n))
    pvalue_matrix = np.ones((n,n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1,S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i,j] = score
            pvalue_matrix[i,j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i],keys[j]))
    return score_matrix, pvalue_matrix, pairs


# #### Data augmentation functions

# In[9]:


from statsmodels.nonparametric.smoothers_lowess import lowess
def apply_lowess_smoothing(df, frac=0.1):
    smoothed_data = pd.DataFrame(index=df.index)
    
    # Apply LOWESS to each column
    for column in df.columns:
        smoothed_values = lowess(df[column], df.index, frac=frac, return_sorted=False)
        smoothed_data[column] = smoothed_values
    
    return smoothed_data


# In[10]:


def cut_mix(df1, df2, alpha):
    np.random.seed(42)  # Set seed only once externally if needed for reproducibility
    assert df1.shape == df2.shape
    size = len(df1)
    cut_length = int(size * alpha)
    cut_point = np.random.randint(0, size - cut_length)  # Ensure slicing does not exceed the size
    
    mixed_df = df1.copy()
    mixed_df.iloc[cut_point:cut_point + cut_length] = df2.iloc[cut_point:cut_point + cut_length]
    
    return mixed_df

def binary_mix(data1, data2, alpha=alpha):
    np.random.seed(42)

    assert len(data1) == len(data2)
    size = data1.shape
    mask = np.random.binomial(1, alpha, size=size).astype(bool)
    
    mixed_data = np.where(mask, data1, data2)
    
    return pd.DataFrame(mixed_data, columns=data1.columns)

def linear_mix(data1, data2, alpha=alpha):
    assert len(data1) == len(data2)
    
    mixed_data = alpha * data1 + (1 - alpha) * data2
    
    return mixed_data

def geometric_mix(data1, data2, alpha=alpha):
    if len(data1) != len(data2):
        raise ValueError("The lengths of data1 and data2 must be the same.")
        
    # Replace zeros and negative values to avoid NaNs or complex numbers
    data1_clipped = np.clip(data1, a_min=1e-10, a_max=None)
    data2_clipped = np.clip(data2, a_min=1e-10, a_max=None)
    
    mixed_data = np.power(data1_clipped, alpha) * np.power(data2_clipped, (1 - alpha))
    
    return mixed_data
def amplitude_mix(data1, data2, alpha=alpha):
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
def proposed_mixup(df1, df2, threshold=0.1, alpha=alpha):
    
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


# In[11]:


import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.utils import resample
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt


# In[12]:


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
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, ))
    indices = np.linspace(0, len(ts)-1, num=knot+2)
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
    random_steps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, ts.shape[1]))
    indices = np.linspace(0, len(ts)-1, num=knot+2)
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


# #### Forecasting

# In[13]:


def augment_dataframe(stock_df, method):
    df = stock_df.copy()
    augmented_stockdf = pd.DataFrame()
    for col in df.columns:
        val = aapl[col]
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


# In[14]:


def create_augmented_data(df1_, df2_, method, alpha=alpha):
    df1 = df1_.copy()
    df2 = df2_.copy()
    
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


# In[15]:


# Define the LSTM model creation function
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_sequences(features, target, time_steps):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:(i + time_steps)])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

# Train the LSTM model and return it along with scalers and the test set
def train_evaluate_lstm(features, target, time_steps, epochs, batch_size):
    X, y = create_sequences(features, target, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
#     test_predictions = model.predict(X_test)
#     test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
#     print(f"Test RMSE: {test_rmse}")
    
    return model

# Predict on new data using the trained model and calculate prediction intervals
def predict_new_data(model, new_data, feature_scaler, target_scaler, quantile, time_steps):
    new_features_scaled = feature_scaler.transform(new_data)
    X_new, _ = create_sequences(new_features_scaled, np.zeros((len(new_features_scaled), new_data.shape[1])), time_steps)
    predictions = model.predict(X_new)
    return predictions


# In[16]:


def classification_accuracy(df_, features_list, X_test_og, y_test_og, scaler_aapl):
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # df should be a dataframe which contains all the features and Close (no Return column)
    df = df_.copy()
    # df should be a dataframe which contains all the features and Close (no Return column)
    df['Return'] = np.log(df['Close']).diff()
    df.dropna(subset=['Return'], inplace=True)
    features = df[features_list]
    target = df['Return']
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features.values)
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

    # Create sequences
    time_steps = 20  # Number of time steps for LSTM
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))  # Prediction of the next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1)

    # Evaluate the model
    predicted_returns = model.predict(X_test_og)
    rmse = np.sqrt(mean_squared_error(y_test_og, predicted_returns))
    print('Test RMSE: ', rmse)

    # Invert scaling to compare predictions against the actual returns
    predicted_returns = scaler.inverse_transform(predicted_returns)

    binary_predicted = (predicted_returns > 0).astype(int)

    # Do the same for actual returns
    binary_actual = (scaler_aapl.inverse_transform(y_test_og) > 0).astype(int)

    # Calculate the proportion of correct directional predictions
    directional_accuracy = np.mean(binary_predicted == binary_actual)
    print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%') 
    
    return directional_accuracy, predicted_returns


# ### Pull Data from Yahoo Finance

# In[17]:


start_date = '2010-01-01'
end_date = '2023-01-01'

# Define the list of Dow Jones Industrial Average companies
tickers = [
    "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS"
    , "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK",
    "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "V", "WBA", "WMT"
]

# tickers = ['AAPL']
# Create a dictionary to store historical data for each company
historical_data = {}

# Loop through the Dow companies and retrieve historical data
for ticker in tickers:
    stock_data = get_stock_data(ticker, start_date, end_date)
    historical_data[ticker] = stock_data


# ### Finding the best way to concat augmented data to the original training set

# In[18]:


df = pd.DataFrame()

for stock, data in historical_data.items():
    df[stock] = data['Adj Close']
    
display(df)


# In[19]:


aapl_df = historical_data['AAPL']
aapl_df = engineer_features(aapl_df)
aapl_df


# ### ONLY augmented data is used

# In[20]:


aapl = aapl_df.copy()
lowess = apply_lowess_smoothing(aapl)
cut_lowess = create_augmented_data(aapl, lowess, method='cut_mix')
binary_lowess = create_augmented_data(aapl, lowess, method='binary_mix')
linear_lowess = create_augmented_data(aapl, lowess, method='linear_mix')
geometric_lowess = create_augmented_data(aapl, lowess, method='geometrix_mix')
amplitude_lowess = create_augmented_data(aapl, lowess, method='amplitude_mix')
proposedmix_lowess = create_augmented_data(aapl, lowess, method='proposed_mix')

jittered_ts = augment_dataframe(aapl, 'jittering')
flipped_ts = augment_dataframe(aapl, 'flipping')
scaled_ts = augment_dataframe(aapl, 'scaling')
mag_warped_ts = augment_dataframe(aapl, 'magnitude_warping')
permuted_ts = augment_dataframe(aapl, 'permutation')
stl_ts = augment_dataframe(aapl, 'stl_augment')

augmented_datasets = {
    'original': aapl,
    'cut_mix': cut_lowess,
    'linear_mix': linear_lowess,
    'geometric_mix': geometric_lowess,
    'amplitude_mix': amplitude_lowess,
    'proposed_mix': proposedmix_lowess,
    'jittering': jittered_ts,
    'scaling': scaled_ts,
    'magnitude_warping': mag_warped_ts,
    'permutation': permuted_ts,
    'stl_augment': stl_ts
}


# ### MA

# In[21]:


features_list = ['Close', 'Volume', 'RSI', 'VWAP', 'high_to_low_ratio', 'open_to_close_ratio', 'volatility_10']
data = augmented_datasets.copy()

# init
aapl['Return'] = np.log(aapl['Close']).diff()
aapl.dropna(subset=['Return'], inplace=True)

features_aapl = aapl[['Close', 'Volume', 'RSI', 'VWAP', 'high_to_low_ratio', 'open_to_close_ratio', 'volatility_10']]
target_aapl = aapl['Return']

# Normalize features
scaler_aapl = MinMaxScaler(feature_range=(0, 1))
scaled_features_aapl = scaler_aapl.fit_transform(features_aapl.values)
scaled_target_aapl = scaler_aapl.fit_transform(target_aapl.values.reshape(-1, 1))

# Create sequences
time_steps = 20  # Number of time steps for LSTM
X, y = create_sequences(scaled_features_aapl, scaled_target_aapl, time_steps)

# Split the data
X_train_aapl, X_test_aapl, y_train_aapl, y_test_aapl = train_test_split(X, y, test_size=0.3, random_state=42)


# In[22]:


# results = {}
# for method, dataset in data.items():
#     res, predicted_returns = classification_accuracy(dataset, features_list, X_test_aapl, y_test_aapl, scaler_aapl)
#     results[method] = {
#         'direction_accuracy': round((res * 100), 2),
#         'predictions': predicted_returns
#     }


# In[23]:


# for k,v in results.items():
#     print(k,v['direction_accuracy'])


# In[45]:


def select_augmented_data(original_df, augmented_df, method, condition=None, short_window=10, long_window=50, alpha=0.2, random_state=42):
    """
    Select augmented data based on a specified method and condition.

    Parameters:
    - original_df: DataFrame with the original data.
    - augmented_df: DataFrame with the augmented data.
    - method: String indicating the method to use for selection ('ma_condition', 'random', etc.).
    - condition: Condition function that takes the DataFrame and returns a boolean mask for selection.
    - alpha: Proportion of the augmented data to add if method is 'random'.
    - random_state: Seed for the random number generator.

    Returns:
    - DataFrame with the original and selected augmented data.
    """
    np.random.seed(random_state)  # For reproducibility
    
    if method == 'ma_condition' and condition is not None:
        # Apply the condition to select rows from augmented_df
        mask = condition(augmented_df)
        selected_data = augmented_df[mask]
    
    elif method == 'random':
        # Randomly select a portion of the augmented data
        n_select = int(len(augmented_df) * alpha)
        selected_data = augmented_df.sample(n=n_select, random_state=random_state)
    
    else:
        raise ValueError(f"Unknown method: {method}")
        
    combined_df = pd.concat([original_df, selected_data]).reset_index(drop=True)
    print(combined_df.shape)

    df = original_df.copy()
    df['Return'] = np.log(df['Close']).diff()
    df.dropna(subset=['Return'], inplace=True)
    features = df[features_list]
    target = df['Return']
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features.values)
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))
    
    # Create sequences
    time_steps = 20  # Number of time steps for LSTM
    original_sequences, original_targets = create_sequences(scaled_features, scaled_target, time_steps)
    
    # Split the original sequences into training and validation sets
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        original_sequences, original_targets, test_size=0.3, random_state=42
    )
    
    selected_data_copy = selected_data.copy()
    selected_data_copy['Return'] = np.log(selected_data_copy['Close']).diff()
    selected_data_copy.dropna(subset=['Return'], inplace=True)
    features_aug = selected_data_copy[features_list]
    target_aug = selected_data_copy['Return']
    
    # Normalize features
    scaler_aug = MinMaxScaler(feature_range=(0, 1))
    scaled_features_aug = scaler_aug.fit_transform(features.values)
    scaled_target_aug = scaler_aug.fit_transform(target.values.reshape(-1, 1))
    
    # Create sequences for augmented data
    augmented_sequences, augmented_targets = create_sequences(scaled_features_aug, scaled_target_aug, time_steps)
    
    # Combine the original training sequences with the augmented sequences
    X_train_combined = np.concatenate((X_train_orig, augmented_sequences), axis=0)
    y_train_combined = np.concatenate((y_train_orig, augmented_targets), axis=0)

    # Shuffle the combined training sequences to ensure random distribution
    p = np.random.permutation(len(X_train_combined))
    X_train = X_train_combined[p]
    y_train = y_train_combined[p]

    return X_train, y_train, X_test, y_test, scaler

# Example condition function for moving average criteria
def ma_condition(df, short_window=20, long_window=240):
    ma_short = df['Close'].rolling(window=short_window).mean()
    ma_long = df['Close'].rolling(window=long_window).mean()
    return ma_short > ma_long


# In[46]:


def classification_accuracy2(X_train, y_train, X_test, y_test, scaler):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))  # Prediction of the next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1)

    # Evaluate the model
    predicted_returns = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicted_returns))
    print('Test RMSE: ', rmse)

    # Invert scaling to compare predictions against the actual returns
    predicted_returns = scaler.inverse_transform(predicted_returns)

    binary_predicted = (predicted_returns > 0).astype(int)

    # Do the same for actual returns
    binary_actual = (scaler_aapl.inverse_transform(y_test) > 0).astype(int)

    # Calculate the proportion of correct directional predictions
    directional_accuracy = np.mean(binary_predicted == binary_actual)
    print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%') 
    
    return directional_accuracy, predicted_returns


# In[47]:


def generate_ma_pairs(short_ma_range, long_ma_range, step=10):
    """
    Generate tuples of (short_ma, long_ma) where short_ma < long_ma with a specified step.

    Parameters:
    - short_ma_range: Tuple specifying the start and end of the range for short moving averages.
    - long_ma_range: Tuple specifying the start and end of the range for long moving averages.
    - step: The increment between each moving average in the range.

    Returns:
    - List of tuples with (short_ma, long_ma).
    """
    ma_pairs = []
    for short_ma in range(short_ma_range[0], short_ma_range[1] + 1, step):
        for long_ma in range(long_ma_range[0], long_ma_range[1] + 1, step):
            if short_ma < long_ma:
                ma_pairs.append((short_ma, long_ma))
    return ma_pairs

# Example usage:
short_ma_range = (10, 50)  # Start and end range for short moving averages
long_ma_range = (60, 200)  # Start and end range for long moving averages
step = 10  # The increment step
ma_pairs = generate_ma_pairs(short_ma_range, long_ma_range, step)

print(ma_pairs)


# In[53]:


results_2['cut_mix']


# In[51]:


results_2 = {}
original = data['original']
for pairs in ma_pairs:
    for method, dataset in data.items():
        if method == 'original':
            continue
        # Apply the condition to select augmented data
        X_train, y_train, X_test, y_test, scaler = select_augmented_data(
            aapl_df, 
            linear_lowess, 
            method='ma_condition', 
            condition=ma_condition,
            short_window=pairs[0], 
            long_window=pairs[1], 
            alpha=0.4,
            random_state=42
        )
        res, predicted_returns = classification_accuracy2(X_train, y_train, X_test, y_test, scaler)
        results_2[method] = {
            'ma_pairs': pairs,
            'direction_accuracy': round((res * 100), 2),
            'predictions': predicted_returns
        }


# In[ ]:


# Removed binary mix due to bugs with index
for k,v in results_2.items():
    print(k,v['direction_accuracy'])


# In[121]:


for k,v in results.items():
    print(k,v['direction_accuracy'])


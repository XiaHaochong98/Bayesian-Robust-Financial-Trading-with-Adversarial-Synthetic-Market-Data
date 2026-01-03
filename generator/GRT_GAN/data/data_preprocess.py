"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data 
"""
# Local packages
import os
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # def data_preprocess(
# #     file_name: str, 
# #     max_seq_len: int, 
# #     padding_value: float=-1.0,
# #     impute_method: str="mode", 
# #     scaling_method: str="minmax", 
# # ) -> Tuple[np.ndarray, np.ndarray, List]:
# #     """Load the data and preprocess into 3d numpy array.
# #     Preprocessing includes:
# #     1. Remove outliers
# #     2. Extract sequence length for each patient id
# #     3. Impute missing data 
# #     4. Normalize data
# #     6. Sort dataset according to sequence length

# #     Args:
# #     - file_name (str): CSV file name
# #     - max_seq_len (int): maximum sequence length
# #     - impute_method (str): The imputation method ("median" or "mode") 
# #     - scaling_method (str): The scaler method ("standard" or "minmax")

# #     Returns:
# #     - processed_data: preprocessed data
# #     - time: ndarray of ints indicating the length for each data
# #     - params: the parameters to rescale the data 
# #     """

# #     #########################
# #     # Load data
# #     #########################

# #     index = 'Idx'

# #     # Load csv
# #     print("Loading data...\n")
# #     ori_data = pd.read_csv(file_name)

# #     # Remove spurious column, so that column 0 is now 'admissionid'.
# #     if ori_data.columns[0] == "Unnamed: 0":  
# #         ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

# #     #########################
# #     # Remove outliers from dataset
# #     #########################
    
# #     no = ori_data.shape[0]
# #     z_scores = stats.zscore(ori_data, axis=0, nan_policy='omit')
# #     z_filter = np.nanmax(np.abs(z_scores), axis=1) < 3
# #     ori_data = ori_data[z_filter]
# #     print(f"Dropped {no - ori_data.shape[0]} rows (outliers)\n")

# #     # Parameters
# #     uniq_id = np.unique(ori_data[index])
# #     no = len(uniq_id)
# #     dim = len(ori_data.columns) - 1

# #     #########################
# #     # Impute, scale and pad data
# #     #########################
    
# #     # Initialize scaler
# #     if scaling_method == "minmax":
# #         scaler = MinMaxScaler()
# #         scaler.fit(ori_data)
# #         params = [scaler.data_min_, scaler.data_max_]
    
# #     elif scaling_method == "standard":
# #         scaler = StandardScaler()
# #         scaler.fit(ori_data)
# #         params = [scaler.mean_, scaler.var_]

# #     # Imputation values
# #     if impute_method == "median":
# #         impute_vals = ori_data.median()
# #     elif impute_method == "mode":
# #         impute_vals = stats.mode(ori_data).mode[0]
# #     else:
# #         raise ValueError("Imputation method should be `median` or `mode`")    

# #     # TODO: Sanity check for padding value
# #     # if np.any(ori_data == padding_value):
# #     #     print(f"Padding value `{padding_value}` found in data")
# #     #     padding_value = np.nanmin(ori_data.to_numpy()) - 1
# #     #     print(f"Changed padding value to: {padding_value}\n")
    
# #     # Output initialization
# #     output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
# #     output.fill(padding_value)
# #     time = []

# #     # For each uniq id
# #     for i in tqdm(range(no)):
# #         # Extract the time-series data with a certain admissionid

# #         curr_data = ori_data[ori_data[index] == uniq_id[i]].to_numpy()

# #         # Impute missing data
# #         curr_data = imputer(curr_data, impute_vals)

# #         # Normalize data
# #         curr_data = scaler.transform(curr_data)
        
# #         # Extract time and assign to the preprocessed data (Excluding ID)
# #         curr_no = len(curr_data)

# #         # Pad data to `max_seq_len`
# #         if curr_no >= max_seq_len:
# #             output[i, :, :] = curr_data[:max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
# #             time.append(max_seq_len)
# #         else:
# #             output[i, :curr_no, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
# #             time.append(curr_no)

# #     return output, time, params, max_seq_len, padding_value

# def data_preprocess(
#     file_name: str, 
#     max_seq_len: int, 
#     padding_value: float=-1.0,
#     impute_method: str="mode", 
#     scaling_method: str="minmax", 
# ) -> Tuple[np.ndarray, np.ndarray, List]:
#     """Load the data and preprocess into 3d numpy array.
#     Preprocessing includes:
#     1. Remove outliers
#     2. Extract sequence length for each patient id
#     3. Impute missing data 
#     4. Normalize data
#     6. Sort dataset according to sequence length

#     Args:
#     - file_name (str): CSV file name
#     - max_seq_len (int): maximum sequence length
#     - impute_method (str): The imputation method ("median" or "mode") 
#     - scaling_method (str): The scaler method ("standard" or "minmax")

#     Returns:
#     - processed_data: preprocessed data
#     - time: ndarray of ints indicating the length for each data
#     - params: the parameters to rescale the data 
#     """

#     #########################
#     # Load data
#     #########################

#     # index = 'date'

#     # # Load CSV
#     # print("Loading data...\n")
#     # ori_data = pd.read_csv(file_name)
#     # print("ori_data:", ori_data.head(), ori_data.shape, ori_data.columns)

#     # # Sort the data by date and bbid
#     # ori_data = ori_data.sort_values(by=[index, 'bbid'])

#     # # Group by 'bbid' and sort within each group
#     # ori_data_group_by_bbid = ori_data.groupby("bbid")
#     # ori_data = ori_data_group_by_bbid.apply(lambda x: x.sort_values(by=['date'], ascending=True)).reset_index(drop=True)

#     # Load the data
#     print("Loading data...\n")
#     ori_data = pd.read_csv(file_name)
#     print("Original data shape:", ori_data.shape)
#     print("Original data columns:", ori_data.columns)

#     # Sort the data by date and bbid
#     ori_data = ori_data.sort_values(by=['date', 'bbid'])

#     # Convert the 'date' column to datetime if it's not already
#     ori_data['date'] = pd.to_datetime(ori_data['date'])

#     # Determine the valid date range for each bbid
#     date_ranges = ori_data.groupby('bbid')['date'].agg(['min', 'max'])
#     print("Date ranges for each bbid:")
#     print(date_ranges)

#     # Find the maximum of the min dates and the minimum of the max dates
#     common_start_date = date_ranges['min'].max()
#     common_end_date = date_ranges['max'].min()

#     # Restrict the start date to be 2003-01-01 and end date to be 2009-12-31
#     restrict_start_date = pd.to_datetime("2003-01-01")
#     restrict_end_date = pd.to_datetime("2009-12-31")

#     # Drop the bbid that does not have full data in the restricted date range
#     bbid_to_drop = date_ranges[(date_ranges['min'] > restrict_start_date) | (date_ranges['max'] < restrict_end_date)].index
#     ori_data = ori_data[~ori_data['bbid'].isin(bbid_to_drop)]
#     # update the bbin_list
#     bbid_list = list(ori_data['bbid'].unique())
#     print("Dropped bbid number due to incomplete data in the restricted date range:", len(bbid_to_drop))

#     print("Common date range across all bbid:")
#     print(f"Start date: {common_start_date}")
#     print(f"End date: {common_end_date}")

#     # Filter the original data to only include rows within this restricted date range
#     filtered_data = ori_data[(ori_data['date'] >= restrict_start_date) & (ori_data['date'] <= restrict_end_date)]
#     print("Filtered data shape:", filtered_data.shape)

#     # for each bbid
#     len_by_bbid = {}
#     for bbid in filtered_data['bbid'].unique():
#         bbid_data = filtered_data[filtered_data['bbid'] == bbid]
#         # get the length of the data
#         data_len = len(bbid_data)
#         len_by_bbid[bbid] = data_len
#     # find the max length of the data
#     max_len = max(len_by_bbid.values())
#     # throw away the bbid that has less data than the max length
#     bbid_to_drop = [bbid for bbid, data_len in len_by_bbid.items() if data_len < max_len]
#     filtered_data = filtered_data[~filtered_data['bbid'].isin(bbid_to_drop)]
#     print(len_by_bbid)
#     print("Dropped bbid number due to incomplete data in the restricted date range:", len(bbid_to_drop))

#     # Pivot the data to create a 3D structure
#     features = filtered_data.columns.drop(['date', 'bbid'])
#     pivot_df = filtered_data.pivot_table(index='date', columns='bbid', values=features)

#     # Calculate dimensions for reshaping
#     time_steps = pivot_df.shape[0]
#     bbid_number = len(filtered_data['bbid'].unique())
#     feature_number = len(features)

#     # Print out the shapes for verification
#     print("Pivoted data shape:", pivot_df.shape)
#     print("Time steps:", time_steps)
#     print("BBID number:", bbid_number)
#     print("Feature number:", feature_number)

#     # Reshape to (time_steps, bbid_number, feature_number)
#     reshaped_data = pivot_df.values.reshape((time_steps, bbid_number, feature_number))

#     # Print the shape of the resulting array
#     print("Reshaped data shape:", reshaped_data.shape)

#     # Define feature sets
#     PV_features = ['close', 'open', 'high', 'low', 'volume', 'caj']
#     Macro_features = ['A191RP1Q027SBEA', 'FEDFUNDS_return', 'VIX', 'SPX_return', 'Crude_return']

#     # Precompute filtered data for each bbid to avoid redundant filtering
#     bbid_list = list(filtered_data['bbid'].unique())
#     bbid_data_dict = {bbid: filtered_data[filtered_data['bbid'] == bbid] for bbid in bbid_list}
#     print("Including the bbid number: ", len(bbid_list))

#     # Initializing outputs and loop variables
#     no = time_steps
#     time = []
#     output_data = []
#     output_macro_data = []
#     output_history_data = []
#     output_transformation_params = {
#         "original close": [],
#         "original open": [],
#         # "original volume": [],
#         "original caj": []
#     }

#     # Desired shape for the concatenated data
#     expected_shape = (max_seq_len, 5)

#     step=5

#     # Loop through the data to collect and process samples
#     for i in tqdm(range(max_seq_len, no - max_seq_len, step)):
#         # Initialize lists to store data for all bbid in the current window
#         curr_pv_sample_list = []
#         curr_macro_sample = None
#         history_pv_sample_list = []
#         bbid_include = []

#         for bbid in bbid_list:
#             bbid_data = bbid_data_dict[bbid]

#             # Ensure that the current window is within bounds for this bbid
#             if i + max_seq_len > len(bbid_data):
#                 print("The current window is out of bounds for this bbid,throwing it away")
#                 continue

#             # Process the PV data into features
#             pv_data_bbid = bbid_data[PV_features]
#             macro_data_bbid = bbid_data[Macro_features]
#             pv_data_bbid_copy = pv_data_bbid.copy()
#             processed_pv_features, pv_close, pv_open, pv_caj = process_data_into_feature(pv_data_bbid_copy)

#             # Rename the columns with the processed features
#             processed_pv_features_name = ['close_return', 'open_close_return', 'high_close_ratio_return', 'low_close_ratio_return', 'volume']
            
#             new_bbid_data = pd.concat([processed_pv_features, macro_data_bbid], axis=1)
#             new_bbid_data.columns = processed_pv_features_name + Macro_features

#             # Get the current and historical pv data slices (window) of length max_seq_len
#             curr_pv_sample = new_bbid_data.iloc[i:i + max_seq_len][processed_pv_features_name].values
#             history_pv_sample = new_bbid_data.iloc[i - max_seq_len:i][processed_pv_features_name].values

#             # Normalize the current pv data with the history data
#             history_mean = history_pv_sample.mean(axis=0)
#             history_std = history_pv_sample.std(axis=0)
#             curr_pv_sample_normalized = (curr_pv_sample - history_mean) / history_std
#             history_pv_sample_normalized = (history_pv_sample - history_mean) / history_std

#             # # use a identical normalization for both current and history data to keep postitive values
#             # curr_pv_sample_normalized = curr_pv_sample
#             history_pv_sample_normalized = history_pv_sample


#             if curr_macro_sample is None:
#                 macro_data_sample = new_bbid_data.iloc[i:i + max_seq_len][Macro_features].values
#                 # not not normalize the macro data
#                 # macro_data_sample = (macro_data_sample - macro_data_sample.mean(axis=0)) / macro_data_sample.std(axis=0)
#                 curr_macro_sample = macro_data_sample
            


#             # threshold = 10

#             # Check current data
#             # large_abs_indices = np.where(np.abs(curr_pv_sample_normalized) > threshold)
#             # if large_abs_indices[0].size > 0:
#             #     print("Original data corresponding to large normalized values in current sample:")
#             #     print(curr_pv_sample[large_abs_indices])
#             #     # save the data to csv for further analysis
#             #     pd.DataFrame(curr_pv_sample).to_csv("problem_data.csv")
#             #     print("bbid: ", bbid)
#             #     exit()

#             # Check historical data
#             # large_abs_indices_hist = np.where(np.abs(history_pv_sample_normalized) > threshold)
#             # if large_abs_indices_hist[0].size > 0:
#             #     print("Original data corresponding to large normalized values in historical sample:")
#             #     print(history_pv_sample[large_abs_indices_hist])
#             #     print("bbid: ", bbid)



#             # Pad the current and history samples if their shape is smaller than expected
#             if curr_pv_sample_normalized.shape != expected_shape:
#                 print("The shape of the current sample is not correct,throwing it away")
#                 continue
#                 # pad_width = ((0, max_seq_len - curr_pv_sample_normalized.shape[0]), (0, 0))
#                 # curr_pv_sample_normalized = np.pad(curr_pv_sample_normalized, pad_width, mode='constant', constant_values=np.nan)
#             if history_pv_sample_normalized.shape != expected_shape:
#                 print("The shape of the history sample is not correct,throwing it away")
#                 continue
#                 # pad_width = ((0, max_seq_len - history_pv_sample_normalized.shape[0]), (0, 0))
#                 # history_pv_sample_normalized = np.pad(history_pv_sample_normalized, pad_width, mode='constant', constant_values=np.nan)
#             if macro_data_sample.shape != expected_shape:
#                 print("The shape of the macro sample is not correct,throwing it away")
#                 continue
#                 # pad_width = ((0, max_seq_len - history_pv_sample_normalized.shape[0]), (0, 0))
#                 # history_pv_sample_normalized = np.pad(history_pv_sample_normalized, pad_width, mode='constant', constant_values=np.nan)
#             # Append the data for the current bbid to the lists
#             curr_pv_sample_list.append(curr_pv_sample_normalized)
#             history_pv_sample_list.append(history_pv_sample_normalized)
#             bbid_include.append(bbid)

#             # Append the original close, open, volume, and caj
#             output_transformation_params["original close"].append(pv_close)
#             output_transformation_params["original open"].append(pv_open)
#             # output_transformation_params["original volume"].append(pv_volume)
#             output_transformation_params["original caj"].append(pv_caj)

#         # Concatenate data from all bbid along the feature axis
#         curr_pv_sample_concat = np.concatenate(curr_pv_sample_list, axis=1)
#         history_pv_sample_concat = np.concatenate(history_pv_sample_list, axis=1)

#         # Ensure the shape is correct
#         if curr_pv_sample_concat.shape[-1] != 5 * len(bbid_list):
#             print(curr_pv_sample_concat.shape, curr_macro_sample.shape, history_pv_sample_concat.shape)
#             print("The shape of the concatenated data is not correct")
#             # print("bbid_include: ", bbid_include, len(bbid_include))
#             continue

#         # Append the concatenated data to the output lists
#         output_data.append(curr_pv_sample_concat)
#         output_macro_data.append(curr_macro_sample)
#         output_history_data.append(history_pv_sample_concat)

#         # Append the time step length
#         time.append(max_seq_len)

#     # Convert lists to numpy arrays
#     output_data = np.array(output_data)
#     output_macro_data = np.array(output_macro_data)
#     output_history_data = np.array(output_history_data)

#     # Check the shapes of the output data
#     print("Output data shape:", output_data.shape)
#     print("Output macro data shape:", output_macro_data.shape)
#     print("Output history data shape:", output_history_data.shape)


#     # Return the processed data
#     return output_data, output_macro_data, output_history_data, time, max_seq_len,bbid_list



# def imputer(
#     curr_data: np.ndarray, 
#     impute_vals: List, 
#     zero_fill: bool = True
# ) -> np.ndarray:
#     """Impute missing data given values for each columns.

#     Args:
#         curr_data (np.ndarray): Data before imputation.
#         impute_vals (list): Values to be filled for each column.
#         zero_fill (bool, optional): Whather to Fill with zeros the cases where 
#             impute_val is nan. Defaults to True.

#     Returns:
#         np.ndarray: Imputed data.
#     """

#     curr_data = pd.DataFrame(data=curr_data)
#     impute_vals = pd.Series(impute_vals)
    
#     # Impute data
#     imputed_data = curr_data.fillna(impute_vals)

#     # Zero-fill, in case the `impute_vals` for a particular feature is `nan`.
#     imputed_data = imputed_data.fillna(0.0)

#     # Check for any N/A values
#     if imputed_data.isnull().any().any():
#         raise ValueError("NaN values remain after imputation")

#     return imputed_data.to_numpy()


# def process_data_into_feature(df):
#     """
#     Process the input DataFrame to generate the required features.
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with columns ["open", "close", "high", "low", "volume", "caj"]
    
#     Returns:
#     pd.DataFrame: DataFrame with processed features
#     pd.Series: Original close series
#     pd.Series: caj column
#     """
#     # calculate the dollar volume by close * volume
#     dollar_volume = df['close'] * df['volume']
#     # calcualte the log of the dollar volume
#     log_dollar_volume = np.log(dollar_volume)

#     # Smooth the OHLC data by dividing by caj
#     smoothed_open = df['open'] / df['caj']
#     smoothed_close = df['close'] / df['caj']
#     smoothed_high = df['high'] / df['caj']
#     smoothed_low = df['low'] / df['caj']
    
#     # Calculate close(t+1) - close(t) return
#     close_return = (smoothed_close.shift(-1) - smoothed_close) / smoothed_close
    
#     # Calculate open(t+1) - close(t) return
#     open_close_return = (smoothed_open.shift(-1) - smoothed_close) / smoothed_close

#     # foward fill the nan in the close_return and open_close_return to avoid calculation error
#     close_return = close_return.ffill()
#     open_close_return = open_close_return.ffill()
    
#     # Calculate high/close ratio
#     high_close_ratio_return = smoothed_high / smoothed_close -1
    
#     # Calculate low/close ratio
#     low_close_ratio_return = smoothed_low / smoothed_close -1

#     # # calcluate the volume return 
#     # volume_return = (df['volume'].shift(-1) - df['volume']) / df['volume']
#     # volume_return = volume_return.ffill()
    
#     # Combine all features into a new DataFrame
#     df_features = pd.DataFrame({
#         'close_return': close_return,
#         'open_close_return': open_close_return,
#         'high_close_ratio_return': high_close_ratio_return,
#         'low_close_ratio_return': low_close_ratio_return,
#         'volume': log_dollar_volume
#     })
    
#     return df_features, smoothed_close, df["open"].iloc[0],  df['caj']

# def revert_feature_into_data(df_features, original_close, original_open , caj_column):
#     """
#     Revert the processed features back to the original OHLC data.
    
#     Parameters:
#     df_features (pd.DataFrame): DataFrame with processed features
#     original_close (pd.Series): Series with the original close values
#     caj_column (pd.Series): Series with the caj values
    
#     Returns:
#     pd.DataFrame: DataFrame with original columns ["open", "close", "high", "low", "volume"]
#     """
#     # Reconstruct smoothed close values from the features
#     # reconstructed the close from the fiorst close value and by accumulating the close_return
#     # smoothed_close = ((original_close * (1 + df_features.iloc[:,0]).cumprod()).shift(1).fillna(original_close))/caj_column
#     # reconstruct ht close from the orginal close price and the close return
#     # print('orginal_close: ', original_close)
#     # print('orginal_volume: ', original_volume)
#     smoothed_close = (original_close.shift(1) * (1 + df_features.iloc[:,0]).shift(1)).fillna(original_close.iloc[0])
#     # print(smoothed_close.shift(1),(df_features.iloc[:,1]+1).shift(1))
#     smoothed_open = (smoothed_close.shift(1) * (df_features.iloc[:,1]+1).shift(1)).fillna(original_open/caj_column.iloc[0])
#     smoothed_high = smoothed_close * (1 + df_features.iloc[:,2])
#     smoothed_low = smoothed_close * (1 + df_features.iloc[:,3])

#     # log calcualtion of the smoothed close step by step
#     # print("1+close_return: ", (1 + df_features.iloc[:,0]))
#     # print("cumprod: ", ((original_close * (1 + df_features.iloc[:,0]).cumprod()).shift(1).fillna(original_close))/caj_column)
#     # # calcualte the nan in the smoothed_close
#     # print("nan in smoothed_close: ", smoothed_close.isna().sum())
#     # print(smoothed_close, smoothed_open, smoothed_high, smoothed_low) 
    
#     # Unsmooth the OHLC data by multiplying by caj
#     close = smoothed_close * caj_column
#     open_ = smoothed_open * caj_column
#     high = smoothed_high * caj_column
#     low = smoothed_low * caj_column
#     # revert the volume with volume return
#     # volume = (original_volume * (1 + df_features.iloc[:,4]).cumprod()).shift(1).fillna(original_volume)
            
#     volumn = np.exp(df_features.iloc[:,4])/close
    
#     # Combine reconstructed values into a new DataFrame
#     df_original = pd.DataFrame({
#         'open': open_,
#         'close': close,
#         'high': high,
#         'low': low,
#         'volume': volumn
#     })
    
#     return df_original

def data_preprocess(
    file_name: str, 
    macro_file_name: str,
    max_seq_len: int, 
    padding_value: float=-1.0,
    impute_method: str="mode", 
    scaling_method: str="minmax", 
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load the data and preprocess into 3d numpy array.
    Preprocessing includes:
    1. Remove outliers
    2. Extract sequence length for each patient id
    3. Impute missing data 
    4. Normalize data
    6. Sort dataset according to sequence length

    Args:
    - file_name (str): CSV file name
    - max_seq_len (int): maximum sequence length
    - impute_method (str): The imputation method ("median" or "mode") 
    - scaling_method (str): The scaler method ("standard" or "minmax")

    Returns:
    - processed_data: preprocessed data
    - time: ndarray of ints indicating the length for each data
    - params: the parameters to rescale the data 
    """

    #########################
    # Load data
    #########################

    # index = 'date'

    # # Load CSV
    # print("Loading data...\n")
    # ori_data = pd.read_csv(file_name)
    # print("ori_data:", ori_data.head(), ori_data.shape, ori_data.columns)

    # # Sort the data by date and ticker
    # ori_data = ori_data.sort_values(by=[index, 'ticker'])

    # # Group by 'ticker' and sort within each group
    # ori_data_group_by_ticker = ori_data.groupby("ticker")
    # ori_data = ori_data_group_by_ticker.apply(lambda x: x.sort_values(by=['date'], ascending=True)).reset_index(drop=True)

    # Load the data
    pv_projection={"Adj Close":"close","High":"high","Low":"low","Open":"open","Volume":"volume"}
    selected_macro_path = "/data3/hcxia/Adahist2/generator/GRT_GAN/data/macro_list.txt"
    with open(selected_macro_path, "r") as f:
        Macro_features = f.read().splitlines()
    print("Loading data...\n")
    ori_data = pd.read_csv(file_name)
    print("Original data shape:", ori_data.shape)
    print("Original data columns:", ori_data.columns)

    # load the macro data
    print("Loading macro data...\n")
    macro_data = pd.read_csv(macro_file_name)
    macro_data= macro_data[["Date"]+Macro_features]
    print("Macro data shape:", macro_data.shape)
    print("Macro data columns:", macro_data.columns)

    # rename the columns "Date" to "date" 
    ori_data.rename(columns={"Date": "date"}, inplace=True)
    macro_data.rename(columns={"Date": "date"}, inplace=True)

    # Sort the data by date and ticker
    ori_data = ori_data.sort_values(by=['date', 'ticker'])

    # Convert the 'date' column to datetime if it's not already
    ori_data['date'] = pd.to_datetime(ori_data['date'])

    # Determine the valid date range for each ticker
    date_ranges = ori_data.groupby('ticker')['date'].agg(['min', 'max'])
    print("Date ranges for each ticker:")
    print(date_ranges)

    # Find the maximum of the min dates and the minimum of the max dates
    common_start_date = date_ranges['min'].max()
    common_end_date = date_ranges['max'].min()

    # Restrict the start date to be 2003-01-01 and end date to be 2009-12-31
    restrict_start_date = pd.to_datetime("2000-01-01")
    restrict_end_date = pd.to_datetime("2017-12-31")
    # restrict_start_date = pd.Timestamp(restrict_start_date).tz_localize('UTC')
    # restrict_end_date = pd.Timestamp(restrict_end_date).tz_localize('UTC')
    # filter the data with the restricted date range
    print("Filtering data with the restricted date range...", restrict_start_date, restrict_end_date)
    # remove the timezone in ori_data
    ori_data['date'] = ori_data['date'].dt.tz_localize(None)
    ori_data = ori_data[(ori_data['date'] >= restrict_start_date) & (ori_data['date'] <= restrict_end_date)]
    # cast the date column to datetime
    macro_data['date'] = pd.to_datetime(macro_data['date'])
    macro_data = macro_data[(macro_data['date'] >= restrict_start_date) & (macro_data['date'] <= restrict_end_date)]

    # Drop the ticker that does not have full data in the restricted date range
    # ticker_to_drop = date_ranges[(date_ranges['min'] > restrict_start_date) | (date_ranges['max'] < restrict_end_date)].index
    # ori_data = ori_data[~ori_data['ticker'].isin(ticker_to_drop)]
    # update the bbin_list
    ticker_list = list(ori_data['ticker'].unique())
    # print("Dropped ticker number due to incomplete data in the restricted date range:", len(ticker_to_drop))

    print("Common date range across all ticker:")
    print(f"Start date: {common_start_date}")
    print(f"End date: {common_end_date}")

    # Filter the original data to only include rows within this restricted date range
    # filtered_data = ori_data[(ori_data['date'] >= restrict_start_date) & (ori_data['date'] <= restrict_end_date)]
    # print("Filtered data shape:", filtered_data.shape)
    filtered_data = ori_data

    # for each ticker
    # len_by_ticker = {}
    # for ticker in filtered_data['ticker'].unique():
    #     ticker_data = filtered_data[filtered_data['ticker'] == ticker]
    #     # get the length of the data
    #     data_len = len(ticker_data)
    #     len_by_ticker[ticker] = data_len
    # # find the max length of the data
    # max_len = max(len_by_ticker.values())
    # # throw away the ticker that has less data than the max length
    # ticker_to_drop = [ticker for ticker, data_len in len_by_ticker.items() if data_len < max_len]
    # filtered_data = filtered_data[~filtered_data['ticker'].isin(ticker_to_drop)]
    # print(len_by_ticker)
    # print("Dropped ticker number due to incomplete data in the restricted date range:", len(ticker_to_drop))




    # rename the columns in filtered_dat with pv_projection
    filtered_data.rename(columns=pv_projection, inplace=True)

    # insepect the date range and number of Nan value in each ticker, find the gcd of the date index 
    full_data_index = (filtered_data['date'].unique())
    # extend each ticker data to the full date index, fill the missing data with Nan
    for ticker in filtered_data['ticker'].unique():
        ticker_data = filtered_data[filtered_data['ticker'] == ticker]
        # ticker_data = ticker_data.set_index('date').reindex(full_data_index).reset_index()
        print("Ticker: ", ticker)
        print("Date range: ", ticker_data['date'].min(), ticker_data['date'].max())
        print("Number of unique date: ", len(ticker_data['date'].unique()))
        print("Number of Nan value in each feature:")
        print(ticker_data.isnull().sum())

    # # generate
    
    # # backward fill the Nan value in the data
    # filtered_data.fillna(method='bfill', inplace=True)
    # # foward fill the Nan value in the data
    # filtered_data.fillna(method='ffill', inplace=True)

    # # check if there is any Nan value in the data
    # print("Number of Nan value in each feature after backward fill:")
    # print(filtered_data.isnull().sum())

    # Pivot the data to create a 3D structure
    features = filtered_data.columns.drop(['date', 'ticker'])
    pivot_df = filtered_data.pivot_table(index='date', columns='ticker', values=features)

    # Calculate dimensions for reshaping
    time_steps = pivot_df.shape[0]
    ticker_number = len(filtered_data['ticker'].unique())
    feature_number = len(features)

    # Print out the shapes for verification
    print("Pivoted data shape:", pivot_df.shape)
    print("Time steps:", time_steps)
    print("ticker number:", ticker_number)
    print("Feature number:", feature_number)


    # Reshape to (time_steps, ticker_number, feature_number)
    reshaped_data = pivot_df.values.reshape((time_steps, ticker_number, feature_number))


    # get the masking matrix for the reshaped data, where 1 indicates the data is missing
    mask_matrix = np.isnan(reshaped_data)
    # print the number of missing data in each feature
    print("Number of missing data in each feature:")
    print(np.sum(mask_matrix, axis=0))
    
    print("filling the missing data")
    # do filling for the missing data
    reshaped_data = np.where(np.isnan(reshaped_data), np.nan, reshaped_data)
    for i in range(reshaped_data.shape[1]):
        reshaped_data[:,i,:] = pd.DataFrame(reshaped_data[:,i,:]).fillna(method='bfill').fillna(method='ffill').values
    
    # get the date range of the reshaped data, get the same date range for the macro data
    # left join the macro data with the reshaped data with the 'date' column
    reshaped_data_date = pivot_df.index
    print("Data range of the reshaped data:", reshaped_data_date[0], reshaped_data_date[-1])
    print("Number of unique date in the reshaped data:", len(reshaped_data_date))
    macro_data = macro_data[macro_data['date'].isin(reshaped_data_date)]
    # fill the missing data in the macro data
    macro_data.fillna(method='bfill', inplace=True)
    macro_data.fillna(method='ffill', inplace=True)
    # Print the shape of the resulting array
    print("Reshaped data shape:", reshaped_data.shape)

    # do the normalization for the macro data
    # do the minmax normalization for the macro data
    scaler = MinMaxScaler()
    macro_data[Macro_features] = scaler.fit_transform(macro_data[Macro_features])
    # Define feature sets
    PV_features = ['close', 'open', 'high', 'low', 'volume']




    # Precompute filtered data for each ticker to avoid redundant filtering
    ticker_list = list(filtered_data['ticker'].unique())
    ticker_data_dict = {ticker: filtered_data[filtered_data['ticker'] == ticker] for ticker in ticker_list}
    print("Including the ticker number: ", len(ticker_list))

    # Initializing outputs and loop variables
    no = time_steps
    time = []
    output_data = []
    output_macro_data = []
    output_history_data = []
    output_mask_data = []
    output_transformation_params = {
        "original close": [],
        "original open": [],
        # "original volume": [],
        "original caj": []
    }

    # Desired shape for the concatenated data
    expected_shape = (max_seq_len, 5)

    step=1

    # Loop through the data to collect and process samples
    # reset the index of the macro_data
    macro_data.reset_index(drop=True, inplace=True)
    for i in tqdm(range(max_seq_len, no - max_seq_len, step)):
        # Initialize lists to store data for all ticker in the current window
        curr_pv_sample_list = []
        curr_macro_sample = None
        curr_mask_sample_list = []
        history_pv_sample_list = []
        ticker_include = []

        for ticker in ticker_list:
            ticker_data = ticker_data_dict[ticker]
            # reset the index of the ticker_data
            # extend the date index to the full date index
            ticker_data = ticker_data.set_index('date').reindex(full_data_index).reset_index()
            ticker_data.reset_index(drop=True, inplace=True)

            # Ensure that the current window is within bounds for this ticker
            if i + max_seq_len > len(ticker_data):
                print("The current window is out of bounds for this ticker,throwing it away")
                continue

            # Process the PV data into features
            macro_data_ticker = macro_data[Macro_features]
            # check if macro data and ticker data have same date for each row
            # if not (macro_data['date'] == ticker_data['date']).all():
            #     print("The macro data and ticker data do not have the same date for each row")
            #     raise ValueError
            if not macro_data['date'].equals(ticker_data['date']):
                print(macro_data['date'])
                print(ticker_data['date'])
                print("The macro data and ticker data do not have the same date for each row")
                raise ValueError


            pv_data_ticker = ticker_data[PV_features]

            pv_data_ticker_copy = pv_data_ticker.copy()
            processed_pv_features, pv_close, pv_open, pv_caj = process_data_into_feature(pv_data_ticker_copy)

            # Rename the columns with the processed features
            processed_pv_features_name = ['close_return', 'open_close_return', 'high_close_ratio_return', 'low_close_ratio_return', 'volume']
            

            # print("processed_pv_features.shape: ", processed_pv_features.shape)
            # print("macro_data_ticker.shape: ", macro_data_ticker.shape)
            # print("Macro_features: ", len(Macro_features))

            new_ticker_data = pd.concat([processed_pv_features, macro_data_ticker], axis=1)
            new_ticker_data.columns = processed_pv_features_name + Macro_features

            # Get the current and historical pv data slices (window) of length max_seq_len
            curr_pv_sample = new_ticker_data.iloc[i:i + max_seq_len][processed_pv_features_name].values
            history_pv_sample = new_ticker_data.iloc[i - max_seq_len:i][processed_pv_features_name].values
            curr_mask_sample = mask_matrix[i:i + max_seq_len, ticker_list.index(ticker), :]

            # Normalize the current pv data with the history data rolling window
            history_mean = history_pv_sample.mean(axis=0)
            history_std = history_pv_sample.std(axis=0)
            curr_pv_sample_normalized = (curr_pv_sample - history_mean) / history_std
            history_pv_sample_normalized = (history_pv_sample - history_mean) / history_std



            # if curr_macro_sample is None:
            macro_data_sample = new_ticker_data.iloc[i:i + max_seq_len][Macro_features].values
            # not not normalize the macro data as we have already normalized the macro data before with the whole period
            # macro_data_sample = (macro_data_sample - macro_data_sample.mean(axis=0)) / macro_data_sample.std(axis=0)
            curr_macro_sample = macro_data_sample
            


            # threshold = 10

            # Check current data
            # large_abs_indices = np.where(np.abs(curr_pv_sample_normalized) > threshold)
            # if large_abs_indices[0].size > 0:
            #     print("Original data corresponding to large normalized values in current sample:")
            #     print(curr_pv_sample[large_abs_indices])
            #     # save the data to csv for further analysis
            #     pd.DataFrame(curr_pv_sample).to_csv("problem_data.csv")
            #     print("ticker: ", ticker)
            #     exit()

            # Check historical data
            # large_abs_indices_hist = np.where(np.abs(history_pv_sample_normalized) > threshold)
            # if large_abs_indices_hist[0].size > 0:
            #     print("Original data corresponding to large normalized values in historical sample:")
            #     print(history_pv_sample[large_abs_indices_hist])
            #     print("ticker: ", ticker)



            # Pad the current and history samples if their shape is smaller than expected
            if curr_pv_sample_normalized.shape != expected_shape:
                print("The shape of the current sample is not correct,throwing it away")
                continue
                # pad_width = ((0, max_seq_len - curr_pv_sample_normalized.shape[0]), (0, 0))
                # curr_pv_sample_normalized = np.pad(curr_pv_sample_normalized, pad_width, mode='constant', constant_values=np.nan)
            if history_pv_sample_normalized.shape != expected_shape:
                print("The shape of the history sample is not correct,throwing it away")
                continue
                # pad_width = ((0, max_seq_len - history_pv_sample_normalized.shape[0]), (0, 0))
                # history_pv_sample_normalized = np.pad(history_pv_sample_normalized, pad_width, mode='constant', constant_values=np.nan)
            # if macro_data_sample.shape != expected_shape:
            #     print("The shape of the macro sample is not correct,throwing it away")
            #     continue
                # pad_width = ((0, max_seq_len - history_pv_sample_normalized.shape[0]), (0, 0))
                # history_pv_sample_normalized = np.pad(history_pv_sample_normalized, pad_width, mode='constant', constant_values=np.nan)
            # Append the data for the current ticker to the lists
            curr_pv_sample_list.append(curr_pv_sample_normalized)
            history_pv_sample_list.append(history_pv_sample_normalized)
            curr_mask_sample_list.append(curr_mask_sample)
            ticker_include.append(ticker)

            # Append the original close, open, volume, and caj
            output_transformation_params["original close"].append(pv_close)
            output_transformation_params["original open"].append(pv_open)
            # output_transformation_params["original volume"].append(pv_volume)
            output_transformation_params["original caj"].append(pv_caj)

        # Concatenate data from all ticker along the feature axis
        curr_pv_sample_concat = np.concatenate(curr_pv_sample_list, axis=1)
        history_pv_sample_concat = np.concatenate(history_pv_sample_list, axis=1)
        curr_mask_sample_concat = np.concatenate(curr_mask_sample_list, axis=1)

        # Ensure the shape is correct
        if curr_pv_sample_concat.shape[-1] != 5 * len(ticker_list):
            print(curr_pv_sample_concat.shape, curr_macro_sample.shape, history_pv_sample_concat.shape)
            print("The shape of the concatenated data is not correct")
            # print("ticker_include: ", ticker_include, len(ticker_include))
            continue

        # Append the concatenated data to the output lists
        output_data.append(curr_pv_sample_concat)
        output_macro_data.append(curr_macro_sample)
        output_history_data.append(history_pv_sample_concat)
        output_mask_data.append(curr_mask_sample_concat)

        # Append the time step length
        time.append(max_seq_len)

    # Convert lists to numpy arrays
    output_data = np.array(output_data)
    output_macro_data = np.array(output_macro_data)
    output_history_data = np.array(output_history_data)
    output_mask_data = np.array(output_mask_data)

    # Check the shapes of the output data
    print("Output data shape:", output_data.shape)
    print("Output macro data shape:", output_macro_data.shape)
    print("Output history data shape:", output_history_data.shape)
    print("Output mask data shape:", output_mask_data.shape)


    # Return the processed data
    return output_data, output_macro_data, output_history_data,output_mask_data, time, max_seq_len,ticker_list



def imputer(
    curr_data: np.ndarray, 
    impute_vals: List, 
    zero_fill: bool = True
) -> np.ndarray:
    """Impute missing data given values for each columns.

    Args:
        curr_data (np.ndarray): Data before imputation.
        impute_vals (list): Values to be filled for each column.
        zero_fill (bool, optional): Whather to Fill with zeros the cases where 
            impute_val is nan. Defaults to True.

    Returns:
        np.ndarray: Imputed data.
    """

    curr_data = pd.DataFrame(data=curr_data)
    impute_vals = pd.Series(impute_vals)
    
    # Impute data
    imputed_data = curr_data.fillna(impute_vals)

    # Zero-fill, in case the `impute_vals` for a particular feature is `nan`.
    imputed_data = imputed_data.fillna(0.0)

    # Check for any N/A values
    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()


def process_data_into_feature(df):
    """
    Process the input DataFrame to generate the required features.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ["open", "close", "high", "low", "volume", "caj"]
    
    Returns:
    pd.DataFrame: DataFrame with processed features
    pd.Series: Original close series
    pd.Series: caj column
    """
    # calculate the dollar volume by close * volume
    dollar_volume = df['close'] * df['volume']
    # calcualte the log of the dollar volume
    log_dollar_volume = np.log(dollar_volume)

    # Smooth the OHLC data by dividing by caj
    smoothed_open = df['open'] 
    smoothed_close = df['close'] 
    smoothed_high = df['high'] 
    smoothed_low = df['low'] 
    
    # Calculate close(t+1) - close(t) return
    close_return = (smoothed_close.shift(-1) - smoothed_close) / smoothed_close
    
    # Calculate open(t+1) - close(t) return
    open_close_return = (smoothed_open.shift(-1) - smoothed_close) / smoothed_close

    # foward fill the nan in the close_return and open_close_return to avoid calculation error
    close_return = close_return.ffill()
    open_close_return = open_close_return.ffill()
    
    # Calculate high/close ratio
    high_close_ratio_return = smoothed_high / smoothed_close -1
    
    # Calculate low/close ratio
    low_close_ratio_return = smoothed_low / smoothed_close -1

    # # calcluate the volume return 
    # volume_return = (df['volume'].shift(-1) - df['volume']) / df['volume']
    # volume_return = volume_return.ffill()
    
    # Combine all features into a new DataFrame
    df_features = pd.DataFrame({
        'close_return': close_return,
        'open_close_return': open_close_return,
        'high_close_ratio_return': high_close_ratio_return,
        'low_close_ratio_return': low_close_ratio_return,
        'volume': log_dollar_volume
    })
    
    return df_features, smoothed_close, df["open"].iloc[0],  None

def revert_feature_into_data(df_features, original_close, original_open , caj_column):
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
    smoothed_close = (original_close.shift(1) * (1 + df_features.iloc[:,0]).shift(1)).fillna(original_close.iloc[0])
    # print(smoothed_close.shift(1),(df_features.iloc[:,1]+1).shift(1))
    smoothed_open = (smoothed_close.shift(1) * (df_features.iloc[:,1]+1).shift(1)).fillna(original_open)
    smoothed_high = smoothed_close * (1 + df_features.iloc[:,2])
    smoothed_low = smoothed_close * (1 + df_features.iloc[:,3])

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
    
    # Combine reconstructed values into a new DataFrame
    df_original = pd.DataFrame({
        'open': open_,
        'close': close,
        'high': high,
        'low': low,
        'volume': volumn
    })
    
    return df_original
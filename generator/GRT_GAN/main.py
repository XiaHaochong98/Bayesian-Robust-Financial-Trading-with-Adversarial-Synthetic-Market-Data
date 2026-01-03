# -*- coding: UTF-8 -*-
# Local modules
import argparse
import logging
import os
import pickle
import random
import shutil
import time

# 3rd-Party Modules
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split

# Self-Written Modules
from data.data_preprocess import data_preprocess
from metrics.metric_utils import (
    feature_prediction, one_step_ahead_prediction, reidentify_score
)

from models.timegan import TimeGAN
from models.utils import timegan_trainer, timegan_generator
import torch.nn as nn

def main(args):
    ##############################################
    # Initialize output directories
    ##############################################

    ## Runtime directory
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}.")

    ## Data directory
    data_path = os.path.abspath("./data")
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at {data_path}.")
    data_dir = os.path.dirname(data_path)
    data_file_name = os.path.basename(data_path)

    ## Output directories
    args.model_path = os.path.abspath(f"./output/{args.exp}/")
    out_dir = os.path.abspath(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # TensorBoard directory
    tensorboard_path = os.path.abspath("./tensorboard")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    print(f"\nCode directory:\t\t\t{code_dir}")
    print(f"Data directory:\t\t\t{data_path}")
    print(f"Output directory:\t\t{out_dir}")
    print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # if args.device != "cpu" and torch.cuda.is_available():
    #     print("Using CUDA\n")
    #     args.device = torch.device(args.device)
    #     # torch.cuda.manual_seed_all(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # else:
    #     print("Using CPU\n")
    #     args.device = torch.device("cpu")

    

    # multi gpu config, not sure if if it trains properly
    # if torch.cuda.is_available():
    #     # Set to CUDA (GPU)
    #     args.device = torch.device("cuda")
    #     print(f"Using {torch.cuda.device_count()} GPU(s)\n")

    #     # # Wrap the model with DataParallel if multiple GPUs are available
    #     # if torch.cuda.device_count() > 1:
    #     #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     #     model = nn.DataParallel(model)

    #     # model.to(args.device)  # Send the model to the GPU(s)
    # else:
    #     # Fallback to CPU
    #     args.device = torch.device("cpu")
    #     print("Using CPU\n")

    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("Using CUDA\n")
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(f"Using {args.device}\n")

    #########################
    # Load and preprocess data for model
    #########################

    # data_path = "data/stock.csv"
    # X, T, _, args.max_seq_len, args.padding_value = data_preprocess(
    #     data_path, args.max_seq_len
    # )

    # data_path = "data/by_industry/20051.csv"
    # data_path = "data/by_industry/20050.csv"
    data_path = args.data_path
    # preprossed_data_folder = args.preprossed_data_folder
    preprossed_data_folder="/data3/hcxia/Adahist2/datasets/output_data"
    print("Data path: ", data_path)
    # instead of processing the data each time, read the prossesed data from npy file

    output_data = np.load(f"{preprossed_data_folder}/output_data.npy")
    output_macro_data = np.load(f"{preprossed_data_folder}/output_macro_data.npy")
    output_history_data = np.load(f"{preprossed_data_folder}/output_history_data.npy")
    T = np.load(f"{preprossed_data_folder}/time.npy")
    bbid_list = np.load(f"{preprossed_data_folder}/ticker_list.npy")
    output_mask_data = np.load(f"{preprossed_data_folder}/output_mask_data.npy")
    # output_data, output_macro_data, output_history_data, T, max_seq_len, bbid_list= data_preprocess(
    #     data_path, args.max_seq_len
    # )
    args.padding_value = 0 
    args.feature_number_per_stock = 5
    import pandas as pd

    output_data_flat = output_data.reshape(-1, output_data.shape[-1])  # Flatten along the first two dimensions
    output_mask_data_flat = output_mask_data.reshape(-1, output_mask_data.shape[-1])
    output_macro_data_flat = output_macro_data.reshape(-1, output_macro_data.shape[-1])
    output_history_data_flat = output_history_data.reshape(-1, output_history_data.shape[-1])

    print(f"output_data: {pd.DataFrame(output_data_flat).describe()}")
    print(f"output_mask_data: {pd.DataFrame(output_mask_data_flat).describe()}")
    print(f"output_macro_data: {pd.DataFrame(output_macro_data_flat).describe()}")
    print(f"output_history_data: {pd.DataFrame(output_history_data_flat).describe()}")

    # # # print the nan number by column name
    # print(f"output_mask_data contains nan: {pd.DataFrame(output_mask_data).isnull().sum()}")
    # # # print the nan number by column name
    # print(f"output_history_data contains nan: {pd.DataFrame(output_history_data).isnull().sum()}")
    # # # print the nan number by column name
    # print(f"output_data contains nan: {pd.DataFrame(output_data).isnull().sum()}")
    # # print the number of nan in the output_macro_data
    # print(f"output_macro_data contains nan: {pd.DataFrame(output_macro_data).isnull().sum()}")

    # Function to analyze and handle 3D data
    def process_and_check_nan(data, name):
        print(f"\n{name} shape: {data.shape}")

        # Check if the input contains NaNs
        nan_count = np.isnan(data).sum()
        print(f"{name} total NaNs: {nan_count}")

        # Reshape 3D data to 2D (flatten second and third dimensions)
        data_flattened = data.reshape(data.shape[0], -1)
        print(f"{name} reshaped to 2D: {data_flattened.shape}")

        # Convert to DataFrame
        df = pd.DataFrame(data_flattened)
        nan_column_sum = df.isnull().sum()

        # Print NaN information for each column
        print(f"{name} NaNs by column:\n{nan_column_sum}")

        return df
    
    # Process and analyze each dataset
    df_output_mask_data = process_and_check_nan(output_mask_data, "output_mask_data")
    df_output_history_data = process_and_check_nan(output_history_data, "output_history_data")
    df_output_data = process_and_check_nan(output_data, "output_data")
    df_output_macro_data = process_and_check_nan(output_macro_data, "output_macro_data")

    # check if the output_data contains nan
    if np.isnan(output_data).any():

        raise ValueError("output_data contains nan")

    # check if the output_mask_data contains nan
    if np.isnan(output_mask_data).any():

        
        raise ValueError("output_mask_data contains nan")
    
    # check if the output_history_data contains nan
    if np.isnan(output_history_data).any():

        
        raise ValueError("output_history_data contains nan")
    
    # check if the output_macro_data contains nan
    if np.isnan(output_macro_data).any():

        
        raise ValueError("output_macro_data contains nan")



    args.synthetic_data_dim=output_data.shape[-1]
    args.history_data_dim=output_history_data.shape[-1]
    args.macro_data_dim=output_macro_data.shape[-1]

    # X= (output_data + output_macro_data + output_history_data)
    args.hist_mode="dim0"
    if args.hist_mode=="dim1":
        X = np.concatenate((output_data, output_history_data, output_macro_data), axis=2)
    elif args.hist_mode=="dim0":
        X = np.concatenate((output_data, output_macro_data), axis=2)
    print("X shape: ", X.shape)
    args.input_data_dim = X.shape[-1]


    # check if the X contains nan
    if np.isnan(X).any():
        # # print the nan number by column name
        # print(f"X contains nan: {pd.DataFrame(X).isnull().sum()}")
        
        raise ValueError("X contains nan")



    # # Z= (noise + macro_data + history_data)
    # output_data_from_X = X[:, :, :output_data.shape[2]]

    # # Print to compare
    # print("output_data shape: ", output_data.shape)
    # print("output_data_from_X shape: ", output_data_from_X.shape)

    # print("output_data preview: ", output_data[:1, :1, :])
    # print("output_data_from_X preview: ", output_data_from_X[:1, :1, :])

    # # Check if they are the same
    # assert np.array_equal(output_data, output_data_from_X), "output_data and output_data_from_X do not match!"


    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
    print(f"output_data preview: {output_data[:1, :1, :args.synthetic_data_dim]}\n")
    print(f"Original data preview:\n{X[:1, :1, :output_data.shape[2]]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = args.synthetic_data_dim

    # Train-Test Split data and time
    train_data, test_data, train_time, test_time,history_normalizor_train,history_normalizor_test,train_mask,test_mask = train_test_split(
        X, T, output_history_data, output_mask_data, test_size=1-args.train_rate, random_state=args.seed
    )
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    test_history_data = test_data[:, :, args.synthetic_data_dim:args.synthetic_data_dim + args.history_data_dim]
    test_macro_data = test_data[:, :, args.synthetic_data_dim + args.history_data_dim:]
    if args.hist_mode=="dim1":
        train_history_data = train_data[:, :, args.synthetic_data_dim:args.synthetic_data_dim + args.history_data_dim]
        print(f"Train history data shape: {train_history_data.shape}")
    elif args.hist_mode=="dim0":
        train_history_data = train_data[:, :args.max_seq_len//2, :args.synthetic_data_dim]
        print(f"Train history data shape: {train_history_data.shape}")
    train_macro_data = train_data[:, :, -args.macro_data_dim:]
    #train macro data shape
    print(f"Train macro data shape: {train_macro_data.shape}")

    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()

    model = TimeGAN(args)
    if args.is_train == True:
        timegan_trainer(model, train_data, train_time, train_mask, args)
    print("saving args", args)
    with open(f"{args.model_path}/args.pickle", "wb") as fb:
        pickle.dump(args, fb,protocol=pickle.HIGHEST_PROTOCOL)
    generated_data = timegan_generator(model, train_time, args, train_history_data, train_macro_data, train_data)
    generated_time = train_time
    print(f"Generated data shape: {generated_data.shape}")

    # denormalize the generated data with the train_history_data
    # Assuming train_history_data is your historical data used for normalization
    history_mean = history_normalizor_train.mean(axis=0)
    history_std = history_normalizor_train.std(axis=0)

    # Assuming generated_data is the data generated by the TimeGAN model
    generated_data_denormalized = (generated_data * history_std) + history_mean
    # denormalize the orginal data
    # Assuming train_data is the original data
    orginal_data=train_data[:,:,:args.synthetic_data_dim]
    orginal_data_denormalized = (orginal_data * history_std) + history_mean
    original_macro=train_data[:,:,]

    # we did not apply normalization to the generated data so the denormalized data is the same as the generated data
    # generated_data_denormalized = generated_data
    # orginal_data_denormalized = orginal_data



    # generate the long sequence by concatenating the generated data
    if args.hist_mode=="dim0":
        # for this mode, we generate the [:,max_seq_len//2:,:] data a time so we step by max_seq_len//2
        # 0. construte the all_in_one_generated_data and all_in_one_orginal_data
        step_size=args.max_seq_len - args.max_seq_len//2
        for i in range(0, generated_data_denormalized.shape[0], step_size):
            if i==0:
                all_in_one_generated_data=generated_data_denormalized[i,args.max_seq_len//2:,:]
                all_in_one_generated_orginal_data=orginal_data_denormalized[i,args.max_seq_len//2:,:]
                all_in_one_macro_data=
            else:
                all_in_one_generated_data=np.concatenate((all_in_one_generated_data,generated_data_denormalized[i,args.max_seq_len//2:,:]),axis=0)
                all_in_one_generated_orginal_data=np.concatenate((all_in_one_generated_orginal_data,orginal_data_denormalized[i,args.max_seq_len//2:,:]),axis=0)
        # 1. save the generated data
        with open(f"{args.model_path}/all_in_one_generated_data.pickle", "wb") as fb:
            pickle.dump(all_in_one_generated_data, fb)
        with open(f"{args.model_path}/all_in_one_generated_orginal_data.pickle", "wb") as fb:
            pickle.dump(all_in_one_generated_orginal_data, fb)






    # Log end time
    end = time.time()
    # print history data preview
    print(f"Train history data preview:\n{train_history_data[:1, -5:, :1]}\n")
    print(f"Generated data preview:\n{generated_data[:1, -5:, :1]}\n")
    print(f"Generated data denormalized preview:\n{generated_data_denormalized[:1, -5:, :1]}\n")
    # describe the train data and generated data
    print(f"orginal_data data describe: {pd.DataFrame(orginal_data.reshape(-1, orginal_data.shape[-1])).describe()}")
    print(f"Generated data describe: {pd.DataFrame(generated_data.reshape(-1, generated_data.shape[-1])).describe()}")
    # denormalized data
    print(f"orginal_data data denormalized describe: {pd.DataFrame(orginal_data_denormalized.reshape(-1, orginal_data_denormalized.shape[-1])).describe()}")
    print(f"Generated data denormalized describe: {pd.DataFrame(generated_data_denormalized.reshape(-1, generated_data_denormalized.shape[-1])).describe()}")

    # describe the all_in_one_generated_data and all_in_one_orginal_data
    if args.hist_mode=="dim0":
        print(f"all_in_one_generated_data describe: {pd.DataFrame(all_in_one_generated_data).describe()}")
        print(f"all_in_one_generated_orginal_data describe: {pd.DataFrame(all_in_one_generated_orginal_data).describe()}")
    # calculate the correlation between stocks and compare the correlation of the original data with that of the generated data
    # 1. calculate the correlation of the original data
    # the current data shape is (seq_len, stock_num*feature_num)
    # we need to reshape it to (feature_num, seq_len) by stock to calculate the correlation
    # use args.feature_number_per_stock to reshape the data
    # all_in_one_generated_data is a (time_len, stock_num*feature_num) data
    # we need to reshape it to (stock_num, time_len, feature_num) to calculate the correlation
    # use args.feature_number_per_stock to reshape the data
    if args.hist_mode=="dim0":
        # Assuming all_in_one_generated_data has shape (time_len, stock_num * feature_num)
        time_len = all_in_one_generated_data.shape[0]
        stock_num = all_in_one_generated_data.shape[1] // args.feature_number_per_stock

        # Reshape the data to (time_len, stock_num, feature_num)
        all_in_one_generated_data_reshaped = all_in_one_generated_data.reshape(time_len, stock_num, args.feature_number_per_stock)

        # Transpose the data to get the desired shape (stock_num, time_len, feature_num)
        all_in_one_generated_data_reshaped = np.transpose(all_in_one_generated_data_reshaped, (1, 0, 2))

        # Do the same for the original data
        all_in_one_generated_orginal_data_reshaped = all_in_one_generated_orginal_data.reshape(time_len, stock_num, args.feature_number_per_stock)
        all_in_one_generated_orginal_data_reshaped = np.transpose(all_in_one_generated_orginal_data_reshaped, (1, 0, 2))
        print("all_in_one_generated_data_reshaped shape: ", all_in_one_generated_data_reshaped.shape)
        # calculate the correlation of the generated data
        # you need to calculate the correlation of each stock
        # for example you have a all_in_one_generated_data_reshaped with shape (3, time_len, 5)
        # the corrlation result should be a (3, 3, 5) matrix, with 3 stocks and 5 features
        # Initialize lists to store the correlation matrices
        
        generated_correlations = []
        original_correlations = []

        # Loop through each stock
        for stock_idx in range(all_in_one_generated_data_reshaped.shape[0]):
            # Extract the data for the current stock
            generated_stock_data = all_in_one_generated_data_reshaped[stock_idx]
            original_stock_data = all_in_one_generated_orginal_data_reshaped[stock_idx]
            
            # Calculate the correlation matrix for the generated and original data
            generated_correlation = np.corrcoef(generated_stock_data, rowvar=False)
            original_correlation = np.corrcoef(original_stock_data, rowvar=False)
            
            # Store the correlation matrices
            generated_correlations.append(generated_correlation)
            original_correlations.append(original_correlation)

        # Convert the list of matrices into a numpy array
        generated_correlations = np.array(generated_correlations)
        original_correlations = np.array(original_correlations)

        # Calculate the absolute difference between the correlation matrices
        correlation_diff = np.abs(generated_correlations - original_correlations)

        # Print the results
        print("Correlation matrix shape: ", generated_correlations.shape)
        print(f"Generated data correlation: {generated_correlations}")
        print(f"Original data correlation: {original_correlations}")
        print(f"Correlation difference: {correlation_diff}")

        import matplotlib.pyplot as plt

        # Initialize an empty list to store the non-diagonal values
        non_diag_values = []

        # Loop through each stock
        for stock_idx in range(correlation_diff.shape[0]):
            # Get the correlation difference matrix for the current stock
            diff_matrix = correlation_diff[stock_idx]
            
            # Extract the non-diagonal elements
            non_diag_values.extend(diff_matrix[np.triu_indices_from(diff_matrix, k=1)])
            non_diag_values.extend(diff_matrix[np.tril_indices_from(diff_matrix, k=-1)])

        # Convert the list to a numpy array for plotting
        non_diag_values = np.array(non_diag_values)

        # Plot the histogram excluding diagonal values
        plt.hist(non_diag_values, bins=100, range=(-1, 1))
        plt.xlabel("Correlation Difference")
        plt.ylabel("Frequency")
        plt.title("Correlation Difference Histogram (Excluding Diagonal)")
        # Save the histogram
        plt.savefig(f"{args.model_path}/correlation_diff_hist.png")

        # calculate the correlation difference of ic(orginial data, macro data) and ic(synthetic data, macro data)
    


    print(f"Model Runtime: {(end - start)/60} mins\n")

    #########################
    # Save train and generated data for visualization
    #########################
    
    # Save splitted data and generated data
    with open(f"{args.model_path}/train_data.pickle", "wb") as fb:
        pickle.dump(train_data, fb)
    with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
        pickle.dump(train_time, fb)
    with open(f"{args.model_path}/test_data.pickle", "wb") as fb:
        pickle.dump(test_data, fb)
    with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
        pickle.dump(test_time, fb)
    with open(f"{args.model_path}/fake_data.pickle", "wb") as fb:
        pickle.dump(generated_data, fb)
    with open(f"{args.model_path}/fake_data_denormalized.pickle", "wb") as fb:
        pickle.dump(generated_data_denormalized, fb)
    with open(f"{args.model_path}/fake_time.pickle", "wb") as fb:
        pickle.dump(generated_time, fb)
    # save the bbid_list in to a txt file with "[bbid1, bbid2, ...]"
    with open(f"{args.model_path}/bbid_list.txt", "w") as f:
        bbid_str=str(bbid_list)
        # replace the ' with "
        bbid_str=bbid_str.replace("'", "\"")
        f.write(bbid_str)


    #########################
    # Preprocess data for seeker
    #########################

    # # Define enlarge data and its labels
    # enlarge_data = np.concatenate((train_data, test_data), axis=0)
    # enlarge_time = np.concatenate((train_time, test_time), axis=0)
    # enlarge_data_label = np.concatenate((np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0)

    # # Mix the order
    # idx = np.random.permutation(enlarge_data.shape[0])
    # enlarge_data = enlarge_data[idx]
    # enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

    # # 1. Feature prediction
    # feat_idx = np.random.permutation(train_data.shape[2])[:args.feat_pred_no]
    # print("Running feature prediction using original data...")
    # ori_feat_pred_perf = feature_prediction(
    #     (train_data, train_time), 
    #     (test_data, test_time),
    #     feat_idx
    # )
    # print("Running feature prediction using generated data...")
    # new_feat_pred_perf = feature_prediction(
    #     (generated_data, generated_time),
    #     (test_data, test_time),
    #     feat_idx
    # )

    # feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

    # print('Feature prediction results:\n' +
    #       f'(1) Ori: {str(np.round(ori_feat_pred_perf, 4))}\n' +
    #       f'(2) New: {str(np.round(new_feat_pred_perf, 4))}\n')

    # # 2. One step ahead prediction
    # print("Running one step ahead prediction using original data...")
    # ori_step_ahead_pred_perf = one_step_ahead_prediction(
    #     (train_data[:,:,:args.Z_dim], train_time), 
    #     (test_data[:,:,:args.Z_dim], test_time)
    # )
    # print("Running one step ahead prediction using generated data...")
    # new_step_ahead_pred_perf = one_step_ahead_prediction(
    #     (generated_data, generated_time),
    #     (test_data[:,:,:args.Z_dim], test_time)
    # )

    # step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    # print('One step ahead prediction results:\n' +
    #       f'(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n' +
    #       f'(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n')

    # print(f"Total Runtime: {(time.time() - start)/60} mins\n")

    return None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        # choices=['cuda', 'cpu'],
        # default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--max_seq_len',
        default=100,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)
    parser.add_argument(
        "--data_path",
        default="",
        type=str)

    args = parser.parse_args()

    # Call main function
    main(args)

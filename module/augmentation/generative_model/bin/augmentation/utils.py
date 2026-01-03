import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import pandas as pd
from pathlib import Path


def get_attr(args, key=None, default_value=None):
    if isinstance(args, dict):
        return args[key] if key in args else default_value
    elif isinstance(args, object):
        return (
            getattr(args, key, default_value)
            if key is not None
            else default_value
        )


# splits for the cretaing augenmented train dataset
# The dataset genearte GlutonTS ListDataset with Slidding window of 390(360 context plus 30 prediction)
def split_gluonts_train_dataset(
    dataset,
    freq="1B",
    context_length=360,
    prediction_length=30,
):
    """Create a Sliding window of 390(360 context plus 30 prediction) for the train dataset"""
    train_data = []
    for entry in dataset:
        n = len(dataset[0][FieldName.TARGET])
        for window in range(0, n - context_length - prediction_length):
            # Splits the data into windows of length 'prediction_length' for testing
            train_entry = {
                FieldName.START: entry[FieldName.START],
                FieldName.TARGET: entry[FieldName.TARGET][
                    : window + context_length + prediction_length
                ],
                FieldName.ITEM_ID: entry[FieldName.ITEM_ID],
            }
            train_data.append(train_entry)

    train_dataset = ListDataset(train_data, freq=freq)
    return train_dataset


def get_forecast_df(forecasts):
    """Convert the forecasts GlutonTS to a pandas dataframe"""
    forecast_dfs = []
    for i in forecasts:
        forecast_dfs.append(pd.DataFrame(i.samples))
    return pd.concat(forecast_dfs, ignore_index=True)


def get_dataset_only_one(folder_path, ticker="AAPL", split_date="2000-01-01"):
    # TODO : Add support for non-contnuous data(currrently the Frequency is Business days but disregards hodilays)
    # Path to the folder containing Excel files
    path = Path(folder_path)

    # Container for all time series
    time_series_data = []
    # Iterate through each file in the folder
    for file in path.glob("*.csv"):
        if file.stem == ticker:
            # Read data from Excel file
            print("Reading CSV file", file)
            df = pd.read_csv(file)

            # Assuming the file has columns 'Date' and 'Price'
            # Convert 'Date' to datetime and ensure it's the index
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df = df.loc[split_date:]
            # Create a time series item for GluonTS
            time_series_data.append(
                {
                    "start": df.index[0],
                    "target": df[
                        "Close"
                    ].tolist(),  # Replace 'Price' with the appropriate column name
                    "item_id": file.stem,  # Use file name as item ID
                }
            )

    # Create GluonTS dataset
    gluonts_dataset = ListDataset(
        time_series_data, freq="1B"
    )  # 'B' for Business Day frequency

    return gluonts_dataset


def get_dji_dataset(folder_path, split_date="2000-01-01"):
    # TODO : Add support for non-contnuous data(currrently the Frequency is Business days but disregards hodilays)
    # Path to the folder containing Excel files
    path = Path(folder_path)

    # Container for all time series
    time_series_data = []
    # Iterate through each file in the folder
    for file in path.glob("*.csv"):
        # Read data from Excel file
        df = pd.read_csv(file)

        # Assuming the file has columns 'Date' and 'Price'
        # Convert 'Date' to datetime and ensure it's the index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.loc[split_date:]
        # Create a time series item for GluonTS
        time_series_data.append(
            {
                "start": df.index[0],
                "target": df[
                    "Close"
                ].tolist(),  # Replace 'Price' with the appropriate column name
                "item_id": file.stem,  # Use file name as item ID
            }
        )

    # Create GluonTS dataset
    gluonts_dataset = ListDataset(
        time_series_data, freq="1B"
    )  # 'B' for Business Day frequency

    return gluonts_dataset


def split_gluonts_dataset(
    dataset, prediction_length, freq="1B", num_test_windows=5
):
    train_data = []
    test_data = []

    for entry in dataset:
        # Splits the data by excluding the last 'prediction_length' points time the number of test windows
        # for training
        train_entry = {
            FieldName.START: entry[FieldName.START],
            FieldName.TARGET: entry[FieldName.TARGET][
                : -num_test_windows * prediction_length
            ],
            FieldName.ITEM_ID: entry[FieldName.ITEM_ID],
        }
        train_data.append(train_entry)
        for window in range(num_test_windows):
            # Splits the data into windows of length 'prediction_length' for testing
            if num_test_windows - window == 1:
                test_entry = {
                    FieldName.START: entry[FieldName.START],
                    FieldName.TARGET: entry[FieldName.TARGET][:],
                    FieldName.ITEM_ID: entry[FieldName.ITEM_ID],
                }
                # Last window of length prediction_length
                test_data.append(test_entry)
                break
            test_entry = {
                FieldName.START: entry[FieldName.START],
                FieldName.TARGET: entry[FieldName.TARGET][
                    : -((num_test_windows - window - 1) * prediction_length)
                ],
                FieldName.ITEM_ID: entry[FieldName.ITEM_ID],
            }
            # Full data for testing
            test_data.append(test_entry)

    train_dataset = ListDataset(train_data, freq=freq)
    test_dataset = ListDataset(test_data, freq=freq)

    return train_dataset, test_dataset


def split_data(stock, lookback, test_set_size=150, prediction_window=7):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index : index + lookback])

    data = np.array(data)
    test_set_size = test_set_size
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-prediction_window]
    y_train = data[:train_set_size, -prediction_window:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = y_train.reshape(-1, prediction_window)
    x_test = data[train_set_size:, :-prediction_window]
    y_test = data[train_set_size:, -prediction_window:]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = y_test.reshape(-1, prediction_window)

    return [x_train, y_train, x_test, y_test]


def split_aug_data(stock, lookback, prediction_window=7):

    x_train = stock[:, :-prediction_window, :]
    y_train = stock[:, -prediction_window:].reshape(-1, prediction_window)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # y_train = y_train.reshape(-1, prediction_window)

    return [x_train, y_train]


def split_data_plus_augmentation(
    stock, lookback, test_set_size=150, prediction_window=7, augmentor=None
):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index : index + lookback])

    data = np.array(data)
    test_set_size = test_set_size
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-prediction_window]
    y_train = data[:train_set_size, -prediction_window:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = y_train.reshape(-1, prediction_window)
    x_test = data[train_set_size:, :-prediction_window]
    y_test = data[train_set_size:, -prediction_window:]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = y_test.reshape(-1, prediction_window)

    return [x_train, y_train, x_test, y_test]

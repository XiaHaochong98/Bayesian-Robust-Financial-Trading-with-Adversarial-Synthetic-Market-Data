from gluonts.dataset.common import ListDataset
import pandas as pd
from pathlib import Path
from gluonts.dataset.field_names import FieldName


def split_gluonts_dataset(
    dataset, prediction_length, freq="B", num_test_windows=5
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


def get_dataset(folder_path):
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
        time_series_data, freq="B"
    )  # 'B' for Business Day frequency

    return gluonts_dataset


class MetaData:
    def __init__(self, freq, prediction_length):
        self.freq = freq
        self.prediction_length = prediction_length


class DJIDataset:
    def __init__(self, path):
        self.path = path
        self.dataset = get_dataset(path)
        self.train, self.test = split_gluonts_dataset(
            self.dataset, 30, num_test_windows=5
        )
        self.metadata = MetaData(freq="1B", prediction_length=30)


# dji_dataset = DJIDataset(path="/home/FYP/pratham001/Diffusion/test/hist")

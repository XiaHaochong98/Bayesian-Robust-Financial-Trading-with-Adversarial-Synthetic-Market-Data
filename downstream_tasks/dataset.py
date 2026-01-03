from typing import List

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
CURRENT = str(Path(__file__).resolve().parents[0])

from downstream_tasks.forecasting.data_provider.data_loader import Dataset_Stocks as BaseDatasetStocks

class AugmentatedDatasetStocks(BaseDatasetStocks):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='/datasets/processd_day_dj30/features',
                 target='ret1', scale=True, timeenc=0, freq='d', features='MS',
                 train_stock_ticker: str = None,
                 test_stock_ticker:  str = None,
                 features_name: List[str] = None,
                 temporals_name=[],
                 normalization_method: str = 'standard',
                 ):
        # Initialize the base class with all parameters
        super(AugmentatedDatasetStocks, self).__init__(
            root_path, flag, size, data_path, target, scale, timeenc, freq, features,
            train_stock_ticker, test_stock_ticker, features_name,temporals_name, normalization_method
        )

    def process_ticker(self, stocks_df):
        # Use super() to call the base class's process_ticker method

        # This assumes that a process_ticker method exists in the base class
        super_result = super(AugmentatedDatasetStocks, self).process_ticker(stocks_df)
        # If the base class method does some processing and returns a result, you can enhance it further here
        # Example of adding additional processing after calling the base method:
        enhanced_result = super_result
        # Here you can add whatever additional processing you need
        return enhanced_result

    def __read_data__(self):
        # Use super() to call the base class's __read_data__ method if it exists
        super_result = super(AugmentatedDatasetStocks, self).__read_data__()
        # Further processing or modifications can be done here
        return super_result

    # Add more methods or override existing ones as needed




# class Dataset():
#     def __init__(self,
#                  root: str = None,
#                  data_path: str = None,
#                  stocks_path: str = None,
#                  features_name: List[str] = None,
#                  temporals_name: List[str] = None,
#                  labels_name: List[str] = None, ):
#         super(Dataset, self).__init__()
#
#         self.root = root
#         self.data_path = data_path
#         self.stocks_path = stocks_path
#         self.features_name = features_name
#         self.temporals_name = temporals_name
#         self.labels_name = labels_name
#
#         self.data_path = os.path.join(root, self.data_path)
#         self.stocks_path = os.path.join(root, self.stocks_path)
#
#         self.stocks = self._init_stocks()
#
#         self.stocks2id = {stock: i for i, stock in enumerate(self.stocks)}
#         self.id2stocks = {i: stock for i, stock in enumerate(self.stocks)}
#
#         self.stocks_df = self._init_stocks_df()
#
#     def _init_stocks(self):
#         print("init stocks...")
#         stocks = []
#         with open(self.stocks_path) as op:
#             for line in op.readlines():
#                 line = line.strip()
#                 stocks.append(line)
#         print("init stocks success...")
#         return stocks
#
#     def _init_stocks_df(self):
#         print("init stocks dataframe...")
#         stocks_df = []
#         for stock in self.stocks:
#             path = os.path.join(self.data_path, f"{stock}.parquet")
#             df = pd.read_parquet(path)
#             df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d")
#             df = df.set_index("timestamp")
#             df = df[self.features_name + self.temporals_name + self.labels_name]
#             print(stock, df.shape)
#             stocks_df.append(df)
#         print("init stocks dataframe success...")
#         return stocks_df

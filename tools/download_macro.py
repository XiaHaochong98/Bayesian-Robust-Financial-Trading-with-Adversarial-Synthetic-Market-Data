from fredapi import Fred
import os

fred = Fred(api_key='5ce3e0cd1cb5b158b0c0f000d194ebd4')

ticker_list=["FEDFUNDS","CPIAUCSL","PAYEMS","GDP","NETFI"]

data_path="datasets/macro/"

# create the folder if it does not exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

for ticker in ticker_list:
    data = fred.get_series(ticker)
    print("ticker: ",ticker)
    print(data.head())
    data.to_csv(os.path.join(data_path,ticker+".csv"))
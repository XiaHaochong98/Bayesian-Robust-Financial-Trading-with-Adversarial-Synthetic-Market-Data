root = None
workdir = "workdir"
tag = "yahoofinance_day_prices_future_etfs"
batch_size = 20

downloader = dict(
    type = "YahooFinanceDayPriceDownloader",
    root = root,
    token = "",
    start_date = "2000-01-01",
    end_date = "2024-01-01",
    interval = "day",
    delay = 2,
    stocks_path = "configs/_asset_list_/future_etfs.txt",
    workdir = workdir,
    tag = tag
)
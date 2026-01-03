root = None
workdir = "workdir"
tag = "yahoofinance_day_prices_dj30"
batch_size = 5

downloader = dict(
    type = "YahooFinanceDayPriceDownloader",
    root = root,
    token = "",
    start_date = "2000-01-01",
    end_date = "2024-01-01",
    interval = "day",
    delay = 2,
    stocks_path = "configs/_asset_list_/dj30.txt",
    workdir = workdir,
    tag = tag
)
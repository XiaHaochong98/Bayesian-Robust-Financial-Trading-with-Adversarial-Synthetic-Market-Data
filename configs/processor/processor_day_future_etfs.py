root = None
workdir = "workdir"
tag = "processd_day_future_etfs"
batch_size = 5

processor = dict(
    type = "Processor",
    root = root,
    path_params = {
        "prices": [
            {
                "type": "yahoofinance",
                "path":"workdir/yahoofinance_day_prices_future_etfs",
            }
        ]
    },
    start_date = "2000-01-01",
    end_date = "2024-01-01",
    interval = "1d",
    stocks_path = "configs/_asset_list_/future_etfs.txt",
    workdir = workdir,
    tag = tag
)


root = None
workdir = "workdir"
tag = "processd_day_futures"
batch_size = 5


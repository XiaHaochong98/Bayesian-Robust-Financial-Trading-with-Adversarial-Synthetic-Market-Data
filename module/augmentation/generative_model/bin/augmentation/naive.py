import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vis import plot_series
from pytorch_lightning.loggers.csv_logs import CSVLogger


def main():
    logger = CSVLogger("./", name="lightning_logs")
    print(logger.log_dir)
    split_date = "2000-01-01"
    datos = pd.read_csv(
        "/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/hist/AAPL.csv"
    )
    datos["Date"] = pd.to_datetime(datos["Date"])
    datos.set_index("Date", inplace=True)
    datos = datos.loc[split_date:]

    series = datos["Close"]
    split_time = len(datos) - 150
    time_valid = datos.index[split_time:]
    x_valid = series[split_time:]

    naive_forecast = series[split_time - 1 : -1]

    errors = naive_forecast.to_numpy() - x_valid.to_numpy()
    abs_errors = np.abs(errors)
    mae = abs_errors.mean()
    logger.log_metrics({"mae": mae})
    logger.save()

    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, label="Series")
    plot_series(time_valid, naive_forecast, label="Forecast")
    plt.savefig(f"{logger.log_dir}/naive_forecast.png")


if __name__ == "__main__":
    main()

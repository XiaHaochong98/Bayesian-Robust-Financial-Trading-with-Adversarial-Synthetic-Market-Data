from downstream import (
    TSFDataModule,
    TSFModel,
)
from utils import split_data, split_aug_data
from metrics import get_metrics, error_distribution_plot, error_metrics
from vis import train_loss_plot, test_predictions_plot, test_prediction_plot
from models.LSTM import LSTM
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import argparse

seed_everything(101)


def main(model, stock, data_path, feature):
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LR = 1e-3
    EPOCHS = 100
    BATCH_SIZE = 64
    LOOKBACK_WINDOW = 30  # how many days to use for our predictions
    PREDICTION_WINDOW = 1  # how many days to predict in the future
    MIN_DELTA = 1e-9
    PATIENCE = 5

    SCALER = MinMaxScaler(feature_range=(0, 1))

    if model == "LSTM":
        model = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
    else:
        raise ValueError("Model not found")

    # getting augmentated data
    data = pd.read_csv(
        data_path,
        index_col=[0],
    )
    data = data.to_numpy().reshape(-1, 100, 30).mean(axis=1)
    data = data.flatten()
    data.shape

    # getting data
    split_date = "2000-01-01"
    # Not getting data from yfinance because it is unstable instead stored the in the CSV file
    # datos = yf.download("AAPL", start="2000-01-01", end="2024-02-01")
    datos = pd.read_csv(
        f"/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/hist/{stock}.csv"
    )
    datos["Date"] = pd.to_datetime(datos["Date"])
    datos.set_index("Date", inplace=True)
    datos = datos.loc[split_date:]
    # datos["Close"] = SCALER.fit_transform(datos["Close"].values.reshape(-1, 1))
    SCALER.fit_transform(
        np.hstack([data, datos.Close.to_numpy()[:-150]]).reshape(-1, 1)
    )  # fitting the scaler on the training + augmented data
    temp = SCALER.transform(
        np.hstack([data, datos.Close.to_numpy()]).reshape(-1, 1)
    )
    datos["Close"] = temp[data.shape[0] :]
    data = temp[: data.shape[0]].reshape(-1, 30, 1)

    # train test split
    X_train_real, y_train_real, X_test, y_test = split_data(
        datos.Close,
        LOOKBACK_WINDOW,
        test_set_size=150,
        prediction_window=PREDICTION_WINDOW,
    )
    X_train, y_train = split_aug_data(
        data, LOOKBACK_WINDOW, prediction_window=PREDICTION_WINDOW
    )

    X_train = np.concatenate((X_train, X_train_real))
    y_train = np.concatenate((y_train, y_train_real))

    print("x_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", X_test.shape)
    print("y_test.shape = ", y_test.shape)

    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # model and data loader
    model = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )

    data_module = TSFDataModule(
        X_train, y_train, X_test, y_test, BATCH_SIZE, 1
    )
    tsf_model = TSFModel(model, LR)

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        mode="min",
    )

    logger = CSVLogger(
        "./", name=f"lightning_logs/downstream/{stock}", version=feature
    )
    logger.log_hyperparams(
        {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lookback_window": LOOKBACK_WINDOW,
            "prediction_window": PREDICTION_WINDOW,
            "augmentation": "TSDiff",
        }
    )
    # training
    trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback],
        logger=logger,
    )

    trainer.fit(
        model=tsf_model,
        datamodule=data_module,
    )

    train_loss_plot(logger.log_dir)

    # test
    test_loss = trainer.test(model=tsf_model, datamodule=data_module)
    print("Test Loss: ", test_loss)
    test_predictions = trainer.predict(
        model=tsf_model,
        datamodule=data_module,
    )
    test_unified = torch.cat(test_predictions)[:, 0].squeeze().numpy()

    # reverse scaling transform pred data
    test_unified_unscaled = SCALER.inverse_transform(
        test_unified.reshape(-1, 1)
    ).flatten()
    # reverse scaling transform real data
    y_test_unscaled = SCALER.inverse_transform(
        y_test[:, 0].detach().numpy().reshape(-1, 1)
    )
    error_distribution_plot(
        y_test_unscaled.reshape(
            -1,
        ),
        test_unified_unscaled,
        logger.log_dir,
    )
    (
        std_errors,
        mean_errors,
        median_errors,
        max_error,
        min_error,
        iqr_errors,
    ) = error_metrics(
        y_test_unscaled.reshape(
            -1,
        ),
        test_unified_unscaled,
    )
    mae, mse, rmse, r2, mape = get_metrics(
        y_test_unscaled.reshape(
            -1,
        ),
        test_unified_unscaled,
    )
    logger.log_hyperparams(
        {
            "metrics": {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
                "mape": float(mape),
                "std_errors": float(std_errors),
                "mean_errors": float(mean_errors),
                "median_errors": float(median_errors),
                "max_error": float(max_error),
                "min_error": float(min_error),
                "iqr_errors": float(iqr_errors),
            }
        }
    )
    logger.save()
    # test predictions plot
    test_prediction_plot(
        f"Real + Augmented Data Prediction using {model.__class__.__name__} Model",
        logger.log_dir,
        y_test_unscaled,
        test_unified_unscaled,
    )

    # TODO Prediction pipepline
    # lookback_series = datos[-LOOKBACK_WINDOW:]
    # dataset_pred = lookback_series.Close

    # dataset_pred = np.reshape(dataset_pred, (-1, 1))

    # # scaling transformation
    # dataset_pred = SCALER.fit_transform(dataset_pred)

    # # Reshaping 1D to 2D array
    # dataset_pred = np.reshape(dataset_pred, (1, -1, 1))

    # dataset_pred = torch.tensor(dataset_pred, dtype=torch.float32)

    # y_pred = torch.zeros((1, PREDICTION_WINDOW, 1))
    # print(dataset_pred.shape, y_pred.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="LSTM",
        help="Model name",
    )
    args = parser.parse_args()
    args = vars(args)
    stocks = ["PG", "GS", "NKE", "DIS", "AXP"]
    features = ["no_lag", "lagged", "alpha158"]
    for stock in stocks:
        for feature in features:
            print("Running for", stock, feature)
            data_path = f"/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/augmented_data/{stock}_{feature}_360daysContext.csv"
            main(
                model=args["model"],
                stock=stock,
                data_path=data_path,
                feature=feature,
            )

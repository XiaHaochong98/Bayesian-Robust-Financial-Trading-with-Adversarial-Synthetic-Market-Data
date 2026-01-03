from downstream import (
    TSFDataModule,
    TSFModel,
)
from utils import split_data
from vis import train_loss_plot, test_predictions_plot, test_prediction_plot
from metrics import get_metrics, error_distribution_plot, error_metrics
from models.LSTM import LSTM
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import argparse
import tsaug as ts
import numpy as np

seed_everything(101)

from benchmarks.basic_augmenation import (
    magnitude_warp,
    time_warp,
    window_warp,
    jitter,
    scaling,
)


da_methods_mapping = {
    "convolve": ts.Convolve(window="hann"),
    "pool": ts.Pool(size=3),
    "jitter": jitter,
    "quantize": ts.Quantize(n_levels=17),
    "reverse": ts.Reverse(),
    "timewarp": ts.TimeWarp(n_speed_change=4, max_speed_ratio=1.5),
    "scaling": scaling,
    "magnitude_warp": magnitude_warp,
    "window_warp": window_warp,
}


def run_augmentation(aug_method, X_train, y_train, augment_times=1):
    augment_times = 1
    if aug_method in ["convolve", "pool", "quantize", "reverse", "timewarp"]:
        X_train = np.concatenate(
            [
                X_train,
                *[
                    da_methods_mapping[aug_method].augment(X_train)
                    for i in range(augment_times)
                ],
            ]
        )
        y_train = np.concatenate(
            [y_train, *[y_train for i in range(augment_times)]]
        )
    elif aug_method in ["magnitude_warp", "window_warp", "scaling", "jitter"]:
        X_train = np.concatenate(
            [
                X_train,
                *[
                    da_methods_mapping[aug_method](X_train)
                    for i in range(augment_times)
                ],
            ]
        )
        y_train = np.concatenate(
            [y_train, *[y_train for i in range(augment_times)]]
        )
    return X_train, y_train


def main(model_name):
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

    # getting data
    split_date = "2000-01-01"
    # Not getting data from yfinance because it is unstable instead stored the in the CSV file
    # datos = yf.download("AAPL", start="2000-01-01", end="2024-02-01")
    datos = pd.read_csv(
        "/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/hist/AAPL.csv"
    )
    datos["Date"] = pd.to_datetime(datos["Date"])
    datos.set_index("Date", inplace=True)
    datos = datos.loc[split_date:]
    SCALER.fit_transform(
        datos["Close"].values[:-150].reshape(-1, 1)
    )  # fitting the scaler on the training data
    datos["Close"] = SCALER.transform(datos["Close"].values.reshape(-1, 1))
    # train test split
    X_train, y_train, X_test, y_test = split_data(
        datos.Close,
        LOOKBACK_WINDOW,
        test_set_size=150,
        prediction_window=PREDICTION_WINDOW,
    )

    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    main_logger = CSVLogger("./", name="lightning_logs/benchmarks")
    main_logger.log_hyperparams(
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
        }
    )
    # data loader
    for i, aug_model in enumerate(da_methods_mapping):
        print(f"Augmentation method {aug_model}")
        X_train_aug, y_train_aug = run_augmentation(
            aug_model, X_train, y_train
        )
        print("x_train.shape = ", X_train.shape)
        print("y_train.shape = ", y_train.shape)
        print("x_test.shape = ", X_test.shape)
        print("y_test.shape = ", y_test.shape)
        X_train_aug = torch.from_numpy(X_train).type(torch.Tensor)
        y_train_aug = torch.from_numpy(y_train).type(torch.Tensor)
        if model_name == "LSTM":
            model = LSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
            )
        else:
            raise ValueError("Model not found")

        data_module = TSFDataModule(
            X_train_aug, y_train_aug, X_test, y_test, BATCH_SIZE, 1
        )
        tsf_model = TSFModel(model, LR)

        early_stop_callback = EarlyStopping(
            monitor="train_loss",
            min_delta=MIN_DELTA,
            patience=PATIENCE,
            mode="min",
        )

        logger = CSVLogger("./", name="lightning_logs/benchmarks")
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
                "augmentation_method": aug_model,
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
        main_logger.log_metrics(
            {
                "test_loss": test_loss[0]["test_loss"],
                "augmentation_method": aug_model,
            }
        )
        main_logger.log_hyperparams(
            {
                aug_model: {
                    "test_loss": test_loss[0]["test_loss"],
                    "augmentation_method": aug_model,
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
            },
        )
        logger.save()
        # test predictions plot
        test_prediction_plot(
            f"Real Data Prediction using {model.__class__.__name__} Model and {aug_model} Augmentation Method",
            logger.log_dir,
            y_test_unscaled,
            test_unified_unscaled,
        )
    main_logger.save()

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
    # Setup argparse
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

    main(model_name=args["model"])

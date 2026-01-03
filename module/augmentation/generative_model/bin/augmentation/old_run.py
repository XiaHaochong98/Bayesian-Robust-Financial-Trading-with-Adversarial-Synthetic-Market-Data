from downstream import (
    create_test_sequence,
    create_train_sequence,
    LSTMModel,
    NLinear,
    TSFDataModule,
    TSFModel,
)
from vis import train_loss_plot, test_predictions_plot
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

seed_everything(101)


def main():
    # getting data
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LR = 1e-3
    EPOCHS = 100
    BATCH_SIZE = 64
    LOOKBACK_WINDOW = 23  # how many days to use for our predictions
    PREDICTION_WINDOW = 7  # how many days to predict in the future
    INPUT_SIZE = 1  # one value per day
    NUM_LAYERS = 3
    HIDDEN_SIZE = 64
    CHANNELS = 1

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

    # getting augmentated data
    data = pd.read_csv(
        "/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/augmented_aapl.csv",
        index_col=[0],
    )

    # train test split
    dataset_train = data.to_numpy().flatten()
    training_cutoff = int(len(datos) - 150)
    test_df = datos[training_cutoff:]
    print(test_df.shape)
    dataset_test = test_df.Close
    dataset_test = np.reshape(dataset_test, (-1, 1))
    dataset_train = np.reshape(dataset_train, (-1, 1))
    dataset_train_scaled = SCALER.fit_transform(dataset_train)
    dataset_train_scaled = dataset_train_scaled.reshape(-1, 30)
    dataset_test_scaled = SCALER.fit_transform(dataset_test)
    print(dataset_train_scaled.shape, dataset_test_scaled.shape)

    X_train, y_train = create_train_sequence(
        dataset_train_scaled, LOOKBACK_WINDOW, PREDICTION_WINDOW
    )

    X_test, y_test = create_test_sequence(
        dataset_test_scaled, LOOKBACK_WINDOW, PREDICTION_WINDOW
    )

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # model and data loader
    model = NLinear(LOOKBACK_WINDOW, PREDICTION_WINDOW, CHANNELS)
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

    logger = CSVLogger("./", name="lightning_logs")

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
    test_unified = torch.cat(test_predictions)[:, 0, :].squeeze().numpy()

    # reverse scaling transform
    test_unified_unscaled = SCALER.inverse_transform(
        test_unified.reshape(-1, 1)
    ).flatten()

    observation_dates = test_df[: len(test_df)].index
    test_prediction_dates = observation_dates[
        LOOKBACK_WINDOW : LOOKBACK_WINDOW + len(test_unified_unscaled)
    ]
    # test predictions plot
    test_predictions_plot(
        logger.log_dir,
        observation_dates,
        dataset_test,
        test_prediction_dates,
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
    main()

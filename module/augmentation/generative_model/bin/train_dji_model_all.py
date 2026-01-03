import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName
import pandas as pd
import os
from gluonts.dataset.common import ListDataset
from pathlib import Path

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.model.callback import EvaluateCallback
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_transforms_indicator,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}
MIN_DELTA = 1e-9
PATIENCE = 5
feature_map = {
    "lagged": 8,
    "basic": 0,
    "alpha158": 88,
}


def create_model(config):
    model = TSDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        num_feat_dynamic_real=feature_map[config["indicator"]],
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
    )
    model.to(config["device"])
    return model


def evaluate_guidance(
    config, model, test_dataset, transformation, num_samples=100
):
    logger.info(f"Evaluating with {num_samples} samples.")
    results = []
    if config["setup"] == "forecasting":
        missing_data_kwargs_list = [
            {
                "missing_scenario": "none",
                "missing_values": 0,
            }
        ]
        config["missing_data_configs"] = missing_data_kwargs_list
    elif config["setup"] == "missing_values":
        missing_data_kwargs_list = config["missing_data_configs"]
    else:
        raise ValueError(f"Unknown setup {config['setup']}")

    Guidance = guidance_map[config["sampler"]]
    sampler_kwargs = config["sampler_params"]
    for missing_data_kwargs in missing_data_kwargs_list:
        logger.info(
            f"Evaluating scenario '{missing_data_kwargs['missing_scenario']}' "
            f"with {missing_data_kwargs['missing_values']:.1f} missing_values."
        )
        sampler = Guidance(
            model=model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_kwargs,
            num_feat_dynamic_real=feature_map[config["indicator"]],
        )

        transformed_testdata = transformation.apply(
            test_dataset, is_train=False
        )
        test_splitter = create_splitter(
            past_length=config["context_length"] + max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="test",
            num_feat_dynamic_real=feature_map[config["indicator"]],
        )

        masking_transform = MaskInput(
            FieldName.TARGET,
            FieldName.OBSERVED_VALUES,
            config["context_length"],
            missing_data_kwargs["missing_scenario"],
            missing_data_kwargs["missing_values"],
        )
        test_transform = test_splitter + masking_transform

        predictor = sampler.get_predictor(
            test_transform,
            batch_size=1280 // num_samples,
            device=config["device"],
        )
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=transformed_testdata,
            predictor=predictor,
            num_samples=num_samples,
        )
        forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
        tss = list(ts_it)
        evaluator = Evaluator()
        metrics, _ = evaluator(tss, forecasts)
        metrics = filter_metrics(metrics)
        results.append(dict(**missing_data_kwargs, **metrics))

    return results


def get_dataset(folder_path, stock="DJI30"):
    # Path to the folder containing Excel files
    path = Path(folder_path)
    stocks = (
        [
            "AAPL",
            "MSFT",
            "JPM",
            "V",
            "RTX",
            "PG",
            "GS",
            "NKE",
            "DIS",
            "AXP",
            "HD",
            "INTC",
            "WMT",
            "IBM",
            "MRK",
            "UNH",
            "KO",
            "CAT",
            "TRV",
            "JNJ",
            "CVX",
            "MCD",
            "VZ",
            "CSCO",
            "XOM",
            "BA",
            "MMM",
            "PFE",
            "WBA",
            "DD",
        ]
        if stock == "DJI30"
        else [stock]
    )

    # Container for all time series
    time_series_data = []
    # Iterate through each file in the folder
    for file in path.glob("*.csv"):
        if file.stem in stocks:
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


# Now you can use 'dataset' for further time series analysis or modeling
# ## train-test split
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


def main(config, log_dir):
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    total_length = context_length + prediction_length

    # Create model
    model = create_model(config)

    # Setup dataset and data loading
    folder_path = "/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/hist"
    dataset = get_dataset(folder_path)
    logger.info(f"Loaded dataset with {len(dataset)} time series.")
    train_dataset, test_dataset = split_gluonts_dataset(
        dataset, prediction_length
    )
    logger.info(f"Training dataset: {len(train_dataset)}")
    logger.info(f"Test dataset: {len(test_dataset)}")
    # assert dataset.metadata.freq == freq
    # assert dataset.metadata.prediction_length == prediction_length

    if config["setup"] == "forecasting":
        training_data = train_dataset
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-total_length)
        training_data, _ = missing_values_splitter.split(train_dataset)

    num_rolling_evals = int(len(test_dataset) / len(train_dataset))

    transformation = create_transforms_indicator(
        num_feat_dynamic_real=feature_map[config["indicator"]],
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=config["prediction_length"],
        indicator=config["indicator"],
    )

    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="train",
        num_feat_dynamic_real=feature_map[config["indicator"]],
    )
    val_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="val",
        num_feat_dynamic_real=feature_map[config["indicator"]],
    )

    callbacks = []
    if config["use_validation_set"]:
        transformed_data = transformation.apply(training_data, is_train=True)
        train_val_splitter = OffsetSplitter(
            offset=-config["prediction_length"] * num_rolling_evals
        )
        _, val_gen = train_val_splitter.split(training_data)
        val_data = val_gen.generate_instances(
            config["prediction_length"], num_rolling_evals
        )

        callbacks = [
            EvaluateCallback(
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                sampler=config["sampler"],
                sampler_kwargs=config["sampler_params"],
                num_samples=config["num_samples"],
                model=model,
                transformation=transformation,
                test_dataset=test_dataset,
                val_dataset=val_data,
                eval_every=config["eval_every"],
                num_feat_dynamic_real=feature_map[config["indicator"]],
            ),
            # Early Stopping Callback
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="train_loss",
                patience=PATIENCE,
                mode="min",
                min_delta=MIN_DELTA,
            ),
        ]
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    log_monitor = "train_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"
    transformed_val_data = transformation.apply(test_dataset, is_train=False)
    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )
    val_loader = ValidationDataLoader(
        Cached(transformed_val_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=val_splitter,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)
    callbacks.append(RichProgressBar())
    csv_logger = CSVLogger("./", name=f"new_lightning_logs/{config['trial']}")
    trainer = pl.Trainer(
        logger=csv_logger,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1,
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
    )
    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(
        model, train_dataloaders=data_loader, val_dataloaders=val_loader
    )
    logger.info("Training completed.")

    best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

    if not best_ckpt_path.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt_path,
        )
    logger.info(f"Loading {best_ckpt_path}.")
    best_state_dict = torch.load(best_ckpt_path)
    model.load_state_dict(best_state_dict, strict=True)

    metrics = (
        evaluate_guidance(config, model, test_dataset, transformation)
        if config.get("do_final_eval", True)
        else "Final eval not performed"
    )
    with open(Path(trainer.logger.log_dir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": trainer.logger.version,
                "metrics": metrics,
            },
            fp,
        )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)
    features = ["basic"]
    stocks = [
        "DJI30",
        # "AAPL",
        # "MSFT",
        # "JPM",
        # "V",
        # "RTX",
        # "PG",
        # "GS",
        # "NKE",
        # "DIS",
        # "AXP",
        # "HD",
        # "INTC",
        # "WMT",
        # "IBM",
        # "MRK",
        # "UNH",
        # "KO",
        # "CAT",
        # "TRV",
        # "JNJ",
        # "CVX",
        # "MCD",
        # "VZ",
        # "CSCO",
        # "XOM",
        # "BA",
        # "MMM",
        # "PFE",
        # "WBA",
        # "DD",
    ]
    for feature in features:
        for stock in stocks:
            experiment_name = f"{stock}_{feature}_360daysContext"
            if feature == "no_lag":  # uses no lags and no additional features
                config["use_lags"] = False
                feature = "basic"
            else:
                config["use_lags"] = True
            config["indicator"] = feature
            config["trial"] = experiment_name
            print("Running experiment: ", experiment_name)
            main(config=config, log_dir=args.out_dir)
            print(f"Training Complete for {experiment_name}")

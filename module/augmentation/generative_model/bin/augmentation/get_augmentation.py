import json
import copy
import logging
import argparse
from pathlib import Path
import glob
import yaml
import torch
import numpy as np
from tqdm.auto import tqdm
from gluonts.mx import DeepAREstimator, TransformerEstimator
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import matplotlib.pyplot as plt
from uncond_ts_diff.utils import (
    create_transforms,
    create_transforms_indicator,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    filter_metrics,
)
from uncond_ts_diff.model import TSDiff, LinearEstimator
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.sampler import (
    MostLikelyRefiner,
    MCMCRefiner,
    DDPMGuidance,
    DDIMGuidance,
)
import pandas as pd
import uncond_ts_diff.configs as diffusion_configs

from utils import (
    get_dataset_only_one,
    get_dji_dataset,
    split_gluonts_dataset,
    split_gluonts_train_dataset,
)

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}
refiner_map = {"most_likely": MostLikelyRefiner, "mcmc": MCMCRefiner}
feature_map = {
    "lagged": 8,
    "basic": 0,
    "alpha158": 88,
}


def load_loadel(
    model_path,
    context_length,
    prediction_length,
    device="cuda:0",
    num_feat_dynamic_real=0,
    use_lags=True,
):
    # Load dataset and model
    logger.info("Loading model")
    model = TSDiff(
        **getattr(
            diffusion_configs,
            "diffusion_small_config",
        ),
        num_feat_dynamic_real=num_feat_dynamic_real,
        freq="1B",
        use_features=True,
        use_lags=use_lags,
        normalization="mean",
        context_length=context_length,
        prediction_length=prediction_length,
        init_skip=True,
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
        ),
        strict=True,
    )
    model = model.to(device)
    return model


def get_best_diffusion_step(model: TSDiff, data_loader, device):
    """what does this mean?"""
    losses = np.zeros(model.timesteps)
    batch = {
        k: v.to(device)
        for k, v in next(iter(data_loader)).items()
        if isinstance(v, torch.Tensor)
    }
    x, features, scale = model._extract_features(batch)
    for t in range(model.timesteps):
        loss, _, _ = model.p_losses(
            x.to(device), torch.tensor([t], device=device)
        )
        losses[t] = loss

    best_t = ((losses - losses.mean()) ** 2).argmin()
    return best_t


def main(
    model_path, file_name, stock="AAPL", indicator="alpha158", use_lags=True
):
    dataset_name = stock
    freq = "1B"
    context_length = 360
    prediction_length = 30
    total_length = context_length + prediction_length
    device = "cuda:0"
    base_model_name = "linear"
    num_samples = 100
    csv_path = "/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/hist"

    logger.info("Loading dataset")
    dataset = get_dataset_only_one(csv_path, ticker=dataset_name)
    print("Length of dataset", len(dataset))
    if len(dataset) == 0:
        return
    logger.info("Splitting dataset")
    train_dataset, test_dataset = split_gluonts_dataset(
        dataset, prediction_length
    )
    logger.info("Loading model")
    model = load_loadel(
        model_path,
        context_length,
        prediction_length,
        device,
        feature_map[indicator],
        use_lags,
    )

    # Setup data transformation and splitter(from source code)
    transformation = create_transforms_indicator(
        num_feat_dynamic_real=feature_map[indicator],
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=prediction_length,
        indicator=indicator,
    )
    transformed_data = transformation.apply(list(train_dataset), is_train=True)

    training_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="train",
        num_feat_dynamic_real=feature_map[indicator],
    )

    test_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="test",
        num_feat_dynamic_real=feature_map[indicator],
    )

    train_dataloader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=1024,
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=2048,
    )

    best_t = get_best_diffusion_step(model, train_dataloader, device)
    logger.info(f"Best diffusion step: {best_t}")

    sliding_train = split_gluonts_train_dataset(dataset)
    print("Length of sliding train", len(sliding_train))
    transformed_sliding = transformation.apply(
        list(sliding_train), is_train=False
    )
    logger.info("Running Refinement pipeline")
    # same as the source code
    iterations = 20
    refiner_config = {
        "guidance": "quantile",
        "method": "lmc",
        "method_kwargs": {"noise_scale": 0.1},
        "refiner_name": "mcmc",
        "step_size": 0.01,
    }  # using the best performing config
    refiner_name = refiner_config.pop("refiner_name")
    Refiner = refiner_map[refiner_name]
    refiner = Refiner(
        model,
        prediction_length,
        num_samples=num_samples,
        fixed_t=best_t,
        iterations=iterations,
        **refiner_config,
        num_feat_dynamic_real=feature_map[indicator],
    )
    refiner_predictor = refiner.get_predictor(
        test_splitter, batch_size=1024 // num_samples, device=device
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_sliding,
        predictor=refiner_predictor,
        num_samples=num_samples,
    )
    y = list(tqdm(forecast_it, total=len(transformed_sliding)))
    x = list(ts_it)

    # plotting on sample window
    ts_entry = x[0][-150:]
    forecast_entry = y[0]
    plt.figure(figsize=(12, 8))
    plt.plot(ts_entry[-60:].to_timestamp())
    forecast_entry.plot(color="green", prediction_intervals=(0, 90))
    plt.legend()
    plt.savefig(f"forecast.png")

    # storing the agmented Prediction windows in a csv file
    forecast_dfs = []
    for i in y:
        forecast_dfs.append(pd.DataFrame(i.samples))
    pd.concat(forecast_dfs, ignore_index=True).to_csv(
        f"/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/data/augmented_data/{file_name}.csv"
    )


def get_model_path(experiment_name):
    path = glob.glob(
        f"/home/FYP/pratham001/Diffusion/unconditional-time-series-diffusion/new_lightning_logs/{experiment_name}/*"
    )[-1]
    return f"{path}/best_checkpoint.ckpt"


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    features = ["alpha158", "basic"]
    stocks = [
        # "AAPL",
        # "MSFT",
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

    for feature in features:
        change_feature = False
        for stock in stocks:
            experiment_name = f"DJI30_{feature}_360daysContext"
            file_name = f"{stock}_{feature}_360daysContext"
            if feature == "no_lag":  # uses no lags and no additional features
                use_lags = False
                feature = "basic"
                change_feature = True
            else:
                use_lags = True
            print("Running experiment: ", file_name)
            main(
                get_model_path(experiment_name),
                file_name,
                stock=stock,
                indicator=feature,
                use_lags=use_lags,
            )
            print(f"Training Complete for {file_name}")
            if change_feature:
                feature = "no_lag"
                change_feature = False

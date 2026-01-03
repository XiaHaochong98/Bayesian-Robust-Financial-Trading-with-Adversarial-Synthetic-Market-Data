import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from module.plots.qualitative_analysis import plot_tsne_time_series, plot_pca_time_series, plot_dynamics

data_dir="../datasets"
workdir="workdir"
plot_dir=os.path.join(workdir,"qualitative_analysis")
import logging

# Configure logging to write to a file, including the timestamp.
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# for each .parquet file and .csv file in the data_dir, load the data draw a plot of it and save it in the workdir
seed=42
mode = "binary"  # "binary" or "continuous"
def main():
    # set random seed
    np.random.seed(seed)
    # set search range
    perplexities = [30]
    iterations = [500]
    # TSNE for data
    TSNE_dir = os.path.join(plot_dir,mode,"TSNE")
    if not os.path.exists(TSNE_dir):
        os.makedirs(TSNE_dir)
    # do recursive search for .parquet and .csv files in the data_dir
    for root, dirs, files in tqdm(os.walk(data_dir), desc="Plotting TSNE"):
        for file in files:
            full_path = os.path.join(root, file)
            file_parent = os.path.basename(os.path.dirname(full_path))
            file_name = os.path.basename(full_path)
            plot_name = f"{file_parent}_{file_name}.png"
            save_path = os.path.join(TSNE_dir, plot_name)
            try:
                for perplexity in perplexities:
                    for n_iter in iterations:
                        save_path = save_path[:-4] + f"_perp{perplexity}_iter{n_iter}.png"
                        if os.path.exists(save_path):
                            print("plot already exists, skipping...")
                            continue
                        if file.endswith(".parquet"):
                            df = pd.read_parquet(full_path)
                        elif file.endswith(".csv"):
                            df = pd.read_csv(full_path)
                        else:
                            print(f"Skipping {file}")
                            continue
                        print(f"Running t-SNE with perplexity = {perplexity} and n_iter = {n_iter}")
                        plot_tsne_time_series(data=df,perplexity=perplexity, n_iter=n_iter, random_state=seed, save_path=save_path,color_scheme=mode)
            except Exception as e:
                logging.error(f"Error plotting {file}: {e}", exc_info=True)

    # TSNE for context/label
    MarketDynamics_dir = os.path.join(plot_dir, mode, "MarketDynamics")
    if not os.path.exists(MarketDynamics_dir):
        os.makedirs(MarketDynamics_dir)
    # do recursive search for .parquet and .csv files in the data_dir
    for root, dirs, files in tqdm(os.walk(data_dir), desc="Plotting TSNE"):
        for file in files:
            full_path = os.path.join(root, file)
            file_parent = os.path.basename(os.path.dirname(full_path))
            file_name = os.path.basename(full_path)
            plot_name = f"{file_parent}_{file_name}.png"
            save_path = os.path.join(MarketDynamics_dir, plot_name)
            try:
                for perplexity in perplexities:
                    for n_iter in iterations:
                        save_path = save_path[:-4] + f"_perp{perplexity}_iter{n_iter}.png"
                        print("plotting {}".format(save_path))
                        if os.path.exists(save_path):
                            print("plot already exists, skipping...")
                            continue
                        if file.endswith(".parquet"):
                            df = pd.read_parquet(full_path)
                            # drop the timestamp column
                        elif file.endswith(".csv"):
                            df = pd.read_csv(full_path)
                        else:
                            print(f"Skipping {file}")
                            continue
                        print(f"Running t-SNE with perplexity = {perplexity} and n_iter = {n_iter}")
                        plot_dynamics(data=df, perplexity=perplexity, n_iter=n_iter, random_state=seed,
                                            save_path=save_path, color_scheme=mode)
            except Exception as e:
                logging.error(f"Error plotting {file}: {e}", exc_info=True)

    # PCA
    PCA_dir = os.path.join(plot_dir,mode,"PCA")
    if not os.path.exists(PCA_dir):
        os.makedirs(PCA_dir)
    for root, dirs, files in tqdm(os.walk(data_dir), desc="Plotting PCA"):
        for file in files:
            full_path = os.path.join(root, file)
            # get plot_name == {parent_foldername}_{filename}.png, there could be .parquet or .csv files
            file_parent = os.path.basename(os.path.dirname(full_path))
            file_name = os.path.basename(full_path)
            plot_name = f"{file_parent}_{file_name}.png"
            save_path=os.path.join(PCA_dir, plot_name)
            try:
                save_path = save_path[:-4] + f"_perp{perplexity}_iter{n_iter}.png"
                if os.path.exists(save_path):
                    print("plot already exists, skipping...")
                    continue
                if file.endswith(".parquet"):
                    df = pd.read_parquet(full_path)
                elif file.endswith(".csv"):
                    df = pd.read_csv(full_path)
                else:
                    print(f"Skipping {file}")
                    continue
                plot_pca_time_series(data=df, random_state=seed,
                              save_path=save_path,color_scheme=mode)
            except Exception as e:
                logging.error(f"Error plotting {file}: {e}", exc_info=True)

if __name__ == '__main__':
    main()
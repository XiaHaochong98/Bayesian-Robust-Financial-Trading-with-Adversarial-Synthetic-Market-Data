This directory contains multiple Python script files.

- `run.py`: This script is used to train and test a downstream forecasting model using PyTorch Lightning. The model is trained on Augmentated stock price data and is used to predict future stock prices.The logs contains the Train Loss Graph and Test Prediction Graph.

- `base_run.py`: This script is used to train and test a downstream forecasting model using PyTorch Lightning. The model is trained on Real stock price data and is used to predict future stock prices. The logs contains the Train Loss Graph and Test Prediction Graph.

- `get_augmentation.py`: This script loads the trained TSDiff model from a checkpoint,and uses the Refiner Pipeline to generate samples for the prediction window. Using sliding window generates complete augmentated time-series data. Saves it to a csv file for trainnig downstream model.

- `optuna.py` : This script performs hyperparametr tuning using Optuna and Pytroch Lightning

- `vis.py` : This script contains code to plot various graph using matplotlib. TODO : Add T-SNE plot to show how representative the Data Generated from Diffusion model for the Real Data.

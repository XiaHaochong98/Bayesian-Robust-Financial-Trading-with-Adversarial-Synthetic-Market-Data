import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def calculate_log_returns(data):
    """Calculate log returns for each column in the DataFrame."""
    data=np.log(data / data.shift(1))
    return data.dropna()
def plot_tsne_time_series(data, window_size=21, perplexity=50, n_iter=2000, random_state=42, save_path=None,color_scheme="continuous"):
    """
    Plots a t-SNE visualization for a multivariate time-series dataset.
    """

    # Remove non-numeric columns
    data = data.select_dtypes(include=[np.number])

    # Data cleaning
    data = data.dropna()

    # Display the size of the data
    print("data shape", data.shape)
    print("data columns", data.columns)
    print("")

    # Normalize the data with the historical mean and standard deviation in a rolling window
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Windowing the data
    windows = [scaled_data[i:i + window_size] for i in range(len(scaled_data) - window_size + 1)]

    # Flatten the windows for t-SNE
    flattened_windows = np.array([window.flatten() for window in windows])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(flattened_windows)

    # # Convert to log returns
    # data = calculate_log_returns(data)

    # # Windowing and normalization
    # windowed_data = []
    # for start in range(len(data) - window_size + 1):
    #     window = data.iloc[start:start+window_size]
    #     scaler = StandardScaler()
    #     normalized_window = scaler.fit_transform(window)
    #     windowed_data.append(normalized_window.flatten())
    # windowed_data = np.array(windowed_data)
    # # Apply t-SNE
    # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    # tsne_results = tsne.fit_transform(windowed_data)

    # Create a figure and axes object for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Mapping each index to a color in the gradient
    num_points = len(tsne_results)
    index_range = pd.Series(np.arange(len(data))[-num_points:])  # Adjusted this line
    min_val, max_val = index_range.min(), index_range.max()
    norm = plt.Normalize(min_val, max_val)
    if color_scheme=="continuous":
        colors = plt.cm.viridis(norm(index_range.values))
    elif color_scheme=="binary":
        # we set the first 0.8 as train and the rest as test, marked by different colors
        colors = np.where(index_range.values < 0.8*max_val, 'r', 'b')

    # Adjust the size and alpha of the scatter plot points
    point_size = 10  # Adjust point size here
    point_alpha = 0.6  # Adjust point opacity here

    # Plotting with adjustments
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=point_size, alpha=point_alpha)

    # Adding a colorbar and its label
    colorbar = plt.colorbar(scatter, ax=ax, label='Index')

    # Setting the title and labels
    ax.set_title('T-SNE Visualization')

    # Saving the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Plot saved at", save_path)


def plot_pca_time_series(data, window_size=30, n_components=2, random_state=42, save_path=None,color_scheme="continuous"):
    """
    Plots a PCA visualization for a multivariate time-series dataset.

    Parameters:
    data (pd.DataFrame): Multivariate time-series data.
    window_size (int): The size of each time window.
    n_components (int): Number of components for PCA.
    random_state (int): Random state for PCA.

    Returns:
    Matplotlib figure: A plot of the PCA results with time evolution.
    """

    # Remove non-numeric columns
    data = data.select_dtypes(include=[np.number])

    # Data clearing
    data = data.dropna()

    # Display the size of the data
    # print("data shape", data.shape)
    # print("")

    # # Normalize the data
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data)
    #
    # # Windowing the data
    # windows = [scaled_data[i:i+window_size] for i in range(len(scaled_data) - window_size + 1)]
    #
    # # Flatten the windows for PCA
    # flattened_windows = np.array([window.flatten() for window in windows])


    # # Windowing and normalization
    # # Convert to log returns
    # data = calculate_log_returns(data)

    # # Windowing and normalization
    # windowed_data = []
    # for start in range(len(data) - window_size + 1):
    #     window = data.iloc[start:start+window_size]
    #     scaler = StandardScaler()
    #     normalized_window = scaler.fit_transform(window)
    #     windowed_data.append(normalized_window.flatten())
    # windowed_data = np.array(windowed_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Windowing the data
    windows = [scaled_data[i:i + window_size] for i in range(len(scaled_data) - window_size + 1)]

    # Flatten the windows for t-SNE
    windowed_data = np.array([window.flatten() for window in windows])

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_results = pca.fit_transform(windowed_data)


    # Create a figure and axes object for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Mapping each index to a color in the gradient
    num_points = len(pca_results)
    index_range = pd.Series(np.arange(len(data))[-num_points:])  # Adjusted this line
    min_val, max_val = index_range.min(), index_range.max()
    norm = plt.Normalize(min_val, max_val)
    if color_scheme=="continuous":
        colors = plt.cm.viridis(norm(index_range.values))
    elif color_scheme=="binary":
        # we set the first 0.8 as train and the rest as test, marked by different colors
        colors = np.where(index_range.values < 0.8*max_val, 'r', 'b')

    # Plotting
    scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], c=colors)

    # Explicitly specify the axes for the colorbar
    plt.colorbar(scatter, ax=ax, label='Index')

    # Setting the title and labels
    ax.set_title('PCA Visualization')

    # Saving the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Plot saved at", save_path)


def plot_dynamics(data, window_size=21, perplexity=50, n_iter=3000, random_state=42, save_path=None,
                  color_scheme="continuous"):
    """
    Plots a t-SNE visualization for a multivariate time-series dataset to show the relationship between
    all points (X) and the last point return of the previous day (Y) and how this relationship changes over time.
    """

    # Remove non-numeric columns and clean data
    data = data.select_dtypes(include=[np.number]).dropna()

    target_columns = ["OT", "close", "ret1", "mov1"]
    for target_column in target_columns:
        try:
            Y = data.loc[:, target_column]
            X = data.drop(columns=[target_column])
            column_name = target_column
            break
        except:
            continue

    target_should_be_normalized = ["OT", "close"]
    target_should_not_be_normalized = ["ret1", "mov1"]

    # Check if the target column needs to be normalized
    if column_name in target_should_be_normalized:
        # Combine X and Y for normalization
        combined = pd.concat([X, Y.to_frame()], axis=1)

        # Normalize
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)

        # Separate back into X and Y
        X = pd.DataFrame(combined_scaled[:, :-1], columns=X.columns)
        Y = pd.Series(combined_scaled[:, -1], name=column_name)
    else:
        # Only normalize X
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Windowing the data
    windowed_X = np.array([X[i:i + window_size].values.flatten() for i in range(len(X) - window_size + 1)])
    windowed_Y = Y[window_size - 1:].values

    # Apply t-SNE to X and Y
    tsne_X = TSNE(n_components=1, perplexity=perplexity, n_iter=n_iter, random_state=random_state).fit_transform(windowed_X)
    tsne_Y = TSNE(n_components=1, perplexity=perplexity, n_iter=n_iter, random_state=random_state).fit_transform(
        windowed_Y.reshape(-1, 1))
    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Map each index to a color
    num_points = len(tsne_X)
    index_range = pd.Series(np.arange(num_points))
    norm = plt.Normalize(index_range.min(), index_range.max())
    colors = plt.cm.viridis(norm(index_range.values)) if color_scheme == "continuous" else np.where(index_range.values < 0.8 * num_points, 'r', 'b')

    # Plotting
    scatter = ax.scatter(tsne_X, tsne_Y, c=colors)

    # Colorbar
    plt.colorbar(scatter, ax=ax, label='Index')

    # Title and labels
    ax.set_title('t-SNE Visualization of X and Y over Time')
    ax.set_xlabel('t-SNE Component of X')
    ax.set_ylabel('t-SNE Component of Y')

    # Save the plot if specified
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved at {save_path}")

    plt.show()



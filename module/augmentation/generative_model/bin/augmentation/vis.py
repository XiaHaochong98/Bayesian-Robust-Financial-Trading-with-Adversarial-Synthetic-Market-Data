# Custom Visualizations for the Augmentation Experiment
import matplotlib.pyplot as plt
import pandas as pd


def train_loss_plot(dir):
    loss_df = pd.read_csv(f"{dir}/metrics.csv")
    # set the size of the plot
    plt.rcParams["figure.figsize"] = [14, 4]

    # Test data
    plt.plot(loss_df.train_loss, label="Training Loss", color="b")
    plt.grid()
    plt.legend()
    plt.savefig(f"{dir}/train_loss.png")
    plt.close()


def test_predictions_plot(
    dir,
    observation_dates,
    dataset_test,
    test_prediction_dates,
    test_unified_unscaled,
):
    plt.plot(
        observation_dates,
        dataset_test,
        label="Observation",
        color="r",
        alpha=1,
        linewidth=0.5,
    )
    plt.plot(
        test_prediction_dates,
        test_unified_unscaled,
        label="Prediction",
        color="b",
        alpha=1,
        linewidth=1.0,
    )
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.grid()
    plt.legend()
    plt.savefig(f"{dir}/test_prediction.png")
    plt.close()


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def test_prediction_plot(title, log_dir, y_test, y_test_pred):
    """
    Function to plot the test prediction
    y_test: The actual test data of shape (n, 1)
    y_test_pred: The predicted test data of shape (n, 1)
    """
    figure, axes = plt.subplots(figsize=(15, 6))
    # axes.xaxis_date()

    axes.plot(y_test, color="red", label="Real Stock Price")
    axes.plot(y_test_pred, color="blue", label="Predicted Stock Price")
    # axes.xticks(np.arange(0,394,50))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig(f"{log_dir}/pred.png")
    plt.show()

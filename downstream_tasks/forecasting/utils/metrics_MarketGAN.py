# Necessary packages
import os
import random
import sys
from argparse import Namespace
from collections import namedtuple

sys.path.append(os.path.abspath('../'))
from downstream_tasks.forecasting.utils.dataset import *
from downstream_tasks.forecasting.models.TCN import TCN
from downstream_tasks.forecasting.models.RNN import RNN
from downstream_tasks.forecasting.models.LSTM import LSTM
from downstream_tasks.forecasting.models.GRU import GRU
from downstream_tasks.forecasting.models.TimesNet_modified import TimesNet
from downstream_tasks.forecasting.models.S4 import S4Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import trange
from utils.util import *

TimesNet_arg = Namespace(
    c_out=args_.feature_dim,
    learning_rate=1e-3,
    batch_size=512,
    embedding_size=64,
    hidden_size=64,
    num_filters=64,
    filter_sizes=[2, 3, 4],
    num_layers=2,
    num_channels=[32],
    kernel_size=3,
    dropout=0.1,
    task_name='short_term_forecast',
    seq_len=args_.max_seq_len - 1,
    pred_len=1,  # one step ahead prediction
    e_layers=3,
    enc_in=args_.feature_dim,
    hidden_dim=32,
    embed='timeF',
    freq='d',
    num_class=args_.label_dim,
    epochs=epochs,
    device=args_.device
)
TCN_args = Namespace(
    input_size=args_.feature_dim,
    output_size=args_.feature_dim,
    num_channels=[32, 64, 64],
    kernel_size=3,
    dropout=0.1
)
RNN_args = Namespace(
    input_dim=args_.feature_dim,
    hidden_dim=32,
    num_layers=3,
    output_dim=args_.feature_dim,
    dropout=0.1
)
LSTM_args = Namespace(
    input_dim=args_.feature_dim,
    hidden_dim=32,
    num_layers=3,
    output_dim=args_.feature_dim,
    dropout=0.1
)
GRU_args = Namespace(
    input_dim=args_.feature_dim,
    hidden_dim=32,
    num_layers=3,
    output_dim=args_.feature_dim,
    dropout=0.1
)

S4_args = S4Model(
    d_input=args_.feature_dim,
    d_output=args_.feature_dim,
    d_model=128,
    n_layers=4,
    dropout=0.1,
    prenorm=True,
)


def forecasting_regression(train_data, test_data, args, model='TimesNet'):
    """Use the previous time-series to predict one-step ahead feature values.

    Args:
    - train_data: training time-series
    - test_data: testing time-series

    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # # normalization the train data
    # scaler = StandardScaler()
    # # train_data shape (no,seq_len,dim)
    # scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
    # # transform the train data and test data
    # train_data = scaler.transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    # test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
    # Define the window size
    window_size = 30  # Example window size, adjust according to your dataset

    # Apply rolling window normalization
    normalized_train_data, normalized_test_data = rolling_window_normalization(train_data, test_data, window_size)
    # Set training features and labels
    train_dataset = RegressionDataset(normalized_train_data, mode="direct", prediction_len=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    # print('train_set shuflle True')

    # Set testing features and labels
    test_dataset = RegressionDataset(normalized_test_data, mode="direct", prediction_len=1)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    # Initialize model
    print('prediction model is: ', model)
    if model == 'TimesNet':
        model = TimesNet(TimesNet_arg)
    elif model == 'TCN':
        model = TCN(TCN_args)
    elif model == 'RNN':
        model = RNN(RNN_args)
    elif model == 'LSTM':
        model = LSTM(LSTM_args)
    elif model == 'GRU':
        model = GRU(GRU_args)
    elif model == "s4":
        model = S4Model(
            d_input=S4_args.d_input,
            d_output=S4_args.d_output,
            d_model=S4_args.d_model,
            n_layers=S4_args.n_layers,
            dropout=S4_args.dropout,
            prenorm=S4_args.prenorm,
        )
    else:
        raise ValueError('model is not supported')
    model.to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TimesNet_arg.learning_rate
    )

    # Train the predictive model
    logger = trange(TimesNet_arg.epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0
        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args.device)
            train_y = train_y.to(args.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            # print(train_p.shape,train_y.shape)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()
        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.6f}")
    # record the last epoch running_loss as training loss
    train_loss = running_loss

    # Evaluate the trained model
    with torch.no_grad():
        running_metric = 0
        running_metric_rescale = 0
        total_metric = 0
        total_metric_rescale = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args.device)
            test_p = model(test_x, test_t).cpu()
            # copy test_p and test_y as test_p_norm and test_y_norm
            test_p_norm = test_p.clone()
            test_y_norm = test_y.clone()
            # reshape test_p_norm and test_y_norm
            test_p_norm = np.reshape(test_p_norm.numpy(), [-1])
            test_y_norm = np.reshape(test_y_norm.numpy(), [-1])
            running_metric += SMAPE(test_y_norm, test_p_norm)
            total_metric += running_metric * test_x.shape[0]

            test_p = scaler.inverse_transform(test_p.reshape(-1, test_p.shape[-1])).reshape(test_p.shape)
            test_y = scaler.inverse_transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)
            test_p = np.reshape(test_p, [-1])
            test_y = np.reshape(test_y, [-1])
            running_metric_rescale += SMAPE(test_y, test_p)
            total_metric_rescale += running_metric_rescale * test_x.shape[0]

    return (100 * total_metric) / len(test_dataset), (100 * total_metric_rescale) / len(test_dataset), train_loss


def forecasting_classification(train_data, test_data, args, model='TimesNet'):
    """Use the previous time-series to predict one-step ahead feature values.

    Args:
    - train_data: training time-series
    - test_data: testing time-series

    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # # normalization the train data
    # scaler = StandardScaler()
    # # train_data shape (no,seq_len,dim)
    # scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
    # # transform the train data and test data
    # train_data = scaler.transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    # test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
    # Define the window size
    window_size = 30  # Example window size, adjust according to your dataset

    # Apply rolling window normalization
    normalized_train_data, normalized_test_data = rolling_window_normalization(train_data, test_data, window_size)
    # Set training features and labels
    train_dataset = ClassificationDataset(normalized_train_data, mode="direct", prediction_len=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    # print('train_set shuflle True')

    # Set testing features and labels
    test_dataset = ClassificationDataset(normalized_test_data, mode="direct", prediction_len=1)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    # Initialize model
    print('prediction model is: ', model)
    if model == 'TimesNet':
        model = TimesNet(TimesNet_arg)
    elif model == 'TCN':
        model = TCN(TCN_args)
    elif model == 'RNN':
        model = RNN(RNN_args)
    elif model == 'LSTM':
        model = LSTM(LSTM_args)
    elif model == 'GRU':
        model = GRU(GRU_args)
    elif model == "s4":
        model = S4Model(
            d_input=S4_args.d_input,
            d_output=S4_args.d_output,
            d_model=S4_args.d_model,
            n_layers=S4_args.n_layers,
            dropout=S4_args.dropout,
            prenorm=S4_args.prenorm,
        )
    else:
        raise ValueError('model is not supported')
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TimesNet_arg.learning_rate
    )

    # Train the predictive model
    logger = trange(TimesNet_arg.epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0
        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args.device)
            train_y = train_y.to(args.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            # print(train_p.shape,train_y.shape)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()
        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.6f}")
    # record the last epoch running_loss as training loss
    train_loss = running_loss

    # Evaluate the trained model
    with torch.no_grad():
        total_accuracy = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args.device)
            test_y = test_y.to(args.device)
            test_p = model(test_x, test_t)

            # Compute accuracy or other relevant metric
            predictions = torch.argmax(test_p, dim=1)
            correct_predictions = (predictions == test_y).sum().item()
            total_accuracy += correct_predictions

        accuracy = total_accuracy / len(test_dataset)

    return accuracy, train_loss


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.input_size = args.input_size
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        classification = torch.sigmoid(logits)
        return logits, classification


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, args_):
        super(DiscriminatorNetwork, self).__init__()
        TimesNet_arg = Namespace(
            batch_size=args_.batch_size,
            input_size=args_.input_size,
            output_size=1,
            train_rate=0.8,
            embedding_size=64,
            hidden_size=64,
            num_layers=2,
            num_filters=64,
            filter_sizes=[2, 3, 4],
            num_channels=[32],
            kernel_size=3,
            dropout=0.1,
            task_name='classification',
            seq_len=args_.max_seq_len,
            pred_len=0,
            e_layers=3,
            enc_in=args_.input_size,
            hidden_dim=32,
            embed='timeF',
            freq='d',
            num_class=1,
        )
        self.model = TimesNet(TimesNet_arg)

    def forward(self, x, t):
        try:
            x = self.model.forward(x)
        except:
            print('x shape is', x.shape)
            print('x', x)
        logit = x
        # get softmax of x
        x = torch.nn.functional.sigmoid(x)
        return logit, x


def post_hoc_discriminator(ori_data, generated_data, args_, epoch=0):
    args = {}
    args["device"] = args_.device
    args["model_type"] = "gru"
    args["epochs"] = epoch
    args["batch_size"] = args_.batch_size
    print(f'batch size is {args["batch_size"]}')
    args["num_layers"] = 6
    args["padding_value"] = -1.0
    args["max_seq_len"] = args_.max_seq_len
    args["train_rate"] = 0.8
    args["learning_rate"] = 1e-3
    args['weight_decay'] = 1
    print('seed')
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)

    ori_data, ori_time = ori_data
    generated_data, generated_time = generated_data
    ori_train_data, ori_test_data, ori_train_time, ori_test_time = train_test_split(
        ori_data, ori_time, test_size=1 - args['train_rate'], random_state=args_.seed
    )
    generated_train_data, generated_test_data, generated_train_time, generated_test_time = train_test_split(
        generated_data, generated_time, test_size=1 - args['train_rate'], random_state=args_.seed
    )
    no, seq_len, dim = ori_data.shape
    args["input_size"] = dim
    args["hidden_dim"] = int(int(dim) / 2)
    args_tuple = namedtuple('GenericDict', args.keys())(**args)

    # normalize the data
    scale = True
    if scale:
        # normalization the train data using MinMaxScaler
        scaler = MinMaxScaler()
        # concatenate ori_train_data and generated_train_data
        full_data = np.concatenate((ori_train_data, generated_train_data), axis=0)
        scaler.fit(full_data.reshape(-1, dim))
        # transform the ori_train_data and generated_train_data
        ori_train_data = scaler.transform(ori_train_data.reshape(-1, dim)).reshape(ori_train_data.shape)
        generated_train_data = scaler.transform(generated_train_data.reshape(-1, dim)).reshape(
            generated_train_data.shape)
        # transform the ori_test_data and generated_test_data
        ori_test_data = scaler.transform(ori_test_data.reshape(-1, dim)).reshape(ori_test_data.shape)
        generated_test_data = scaler.transform(generated_test_data.reshape(-1, dim)).reshape(generated_test_data.shape)

    train_dataset = DiscriminatorDataset(ori_data=ori_train_data, generated_data=generated_train_data,
                                         ori_time=ori_train_time, generated_time=generated_train_time)
    test_dataset = DiscriminatorDataset(ori_data=ori_test_data, generated_data=generated_test_data,
                                        ori_time=ori_test_time, generated_time=generated_test_time)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False
    )

    # Train the post-host discriminator
    discriminator = DiscriminatorNetwork(args_tuple)
    discriminator.to(args["device"])
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=args['learning_rate'],
                                 weight_decay=args['weight_decay'])
    logger = trange(args["epochs"], desc=f"Epoch: 0,loss: 0, real_loss: 0, fake_loss: 0")
    for epoch in logger:
        running_real_loss = 0.0
        running_fake_loss = 0.0
        running_loss = 0.0
        for generated_data, generated_time, ori_data, ori_time in train_dataloader:
            generated_data = generated_data.to(args["device"])
            ori_data = ori_data.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            generated_logits, generated_label = discriminator(generated_data, generated_time)
            ori_logits, ori_label = discriminator(ori_data, ori_time)
            D_loss_real = torch.nn.functional.binary_cross_entropy(ori_label, torch.ones_like(ori_label))
            D_loss_fake = torch.nn.functional.binary_cross_entropy(generated_label, torch.zeros_like(generated_label))
            D_loss = D_loss_real + D_loss_fake
            # backward
            D_loss.backward()
            # optimize
            optimizer.step()
            running_real_loss += D_loss_real.item()
            running_fake_loss += D_loss_fake.item()
            running_loss += D_loss.item()

        logger.set_description(
            f"batchnum: {len(train_dataloader)}, Epoch: {epoch},loss: {running_loss / len(train_dataloader):.4f}, real_loss: {running_real_loss / len(train_dataloader):.4f}, fake_loss: {running_fake_loss / len(train_dataloader):.4f}")
    # Evaluate the discriminator on the test set
    with torch.no_grad():
        discriminative_score = []
        running_loss = 0
        for generated_data, generated_time, ori_data, ori_time in test_dataloader:
            generated_data = generated_data.to(args["device"])
            # generated_time = generated_time.to(configs["device"])
            ori_data = ori_data.to(args["device"])
            # ori_time = ori_time.to(configs["device"])

            generated_logits, generated_label = discriminator(generated_data, generated_time)
            generated_logits = generated_logits.cpu()
            generated_label = generated_label.cpu()
            ori_logits, ori_label = discriminator(ori_data, ori_time)
            ori_logits = ori_logits.cpu()
            ori_label = ori_label.cpu()
            y_pred_final = torch.squeeze(torch.concat((ori_label, generated_label), axis=0))
            y_label_final = torch.concat((torch.ones_like(ori_label), torch.zeros_like(generated_label)),
                                         axis=0)
            acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
            discriminative_score.append(acc)
            y_pred_final = torch.squeeze(y_pred_final)
            y_label_final = torch.squeeze(y_label_final)
            running_loss += torch.nn.functional.binary_cross_entropy(y_pred_final, y_label_final)

    return (sum(discriminative_score) * 100) / len(discriminative_score), running_loss / len(test_dataloader)


def predictive_score(ori_data, generated_data, args_):
    args = {}
    args["device"] = args_.device
    args["learning_rate"] = 1e-3
    # configs["grad_clip_norm"] = 5.0
    #
    ori_data, ori_time = ori_data
    generated_data, generated_time = generated_data
    no, seq_len, dim = ori_data.shape

    TimesNet_arg = Namespace(
        model=args.pretrain_model,
        input_size=args.feature_dim,
        output_size=args.label_dim,
        embedding_size=64,
        hidden_size=64,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='classification',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.label_dim,
    )

    args_tuple = namedtuple('GenericDict', args.keys())(**args)

    # Set training features and labels
    train_dataset = OneStepPredictionDataset(ori_data, ori_time)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(generated_data, generated_time)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False
    )
    # Initialize model
    model = TimesNet(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["learning_rate"]
    )

    # Train the predictive model
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args["device"])
            train_y = train_y.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

    # Evaluate the trained model
    with torch.no_grad():
        perf = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x, test_t).cpu()

            test_p = np.reshape(test_p.numpy(), [-1])
            test_y = np.reshape(test_y.numpy(), [-1])

            perf += SMAPE(test_y, test_p)

    return perf / len(test_dataset)


def feature_constraint_evaluaton(data, features):
    high_low_diff = data[:, :, features.index('high')] - data[:, :, features.index('low')]
    high_open_diff = data[:, :, features.index('high')] - data[:, :,
                                                          features.index('open')]
    high_close_diff = data[:, :, features.index('high')] - data[:, :,
                                                           features.index('close')]
    low_open_diff = data[:, :, features.index('low')] - data[:, :, features.index('open')]
    low_close_diff = data[:, :, features.index('low')] - data[:, :, features.index('close')]
    # get the percentage of the data that doesn't satisfy the logic constraints
    high_low_ratio_loss = np.sum(high_low_diff < 0) / (data.shape[0] * data.shape[1])
    high_open_ratio_loss = np.sum(high_open_diff < 0) / (data.shape[0] * data.shape[1])
    high_close_ratio_loss = np.sum(high_close_diff < 0) / (data.shape[0] * data.shape[1])
    low_open_ratio_loss = np.sum(low_open_diff > 0) / (data.shape[0] * data.shape[1])
    low_close_ratio_loss = np.sum(low_close_diff > 0) / (data.shape[0] * data.shape[1])
    print(
        f"Percentage of data that doesn't satisfy the logic constraints: {round(high_low_ratio_loss, 4)}, {round(high_open_ratio_loss, 4)}, {round(high_close_ratio_loss, 4)}, {round(low_open_ratio_loss, 4)}, {round(low_close_ratio_loss, 4)}")
    avg_ratio = (
                        high_low_ratio_loss + high_open_ratio_loss + high_close_ratio_loss + low_open_ratio_loss + low_close_ratio_loss) / 5
    return avg_ratio

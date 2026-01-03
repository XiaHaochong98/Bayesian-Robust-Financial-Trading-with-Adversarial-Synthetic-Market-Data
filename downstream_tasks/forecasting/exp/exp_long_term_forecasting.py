import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from torch import optim
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # change the model to classification mode if the target is 'mov1'
        # if self.configs.target == 'mov1':
        #     self.configs.task_name = 'classification'
        #     self.configs.num_class = 2
        model = self.model_dict[self.args.model].Model(self.args).float()
        # if self.configs.target == 'mov1':
        #     self.configs.task_name = 'long_term_forecast'

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        print('loss function:', loss_name)
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        elif loss_name == 'BCE':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'CE':
            return nn.CrossEntropyLoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        else:
            Exception(f"The loss function {loss_name} is not defined")

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        train_acc_count = 0
        total_acc = None
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                try:
                    f_dim = vali_data.target_dim * -1
                    # print("f_dim:", f_dim)
                except:
                    pass
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.loss == 'CE':

                    outputs_ = outputs.squeeze(1)
                    batch_y_ = batch_y.squeeze(1)

                    # Convert the one-hot encoded target to class indices
                    target_indices = torch.argmax(batch_y_, dim=1)
                    loss = criterion(outputs_, target_indices)
                else:
                    loss = criterion(outputs, batch_y)
                # loss = criterion(pred, true)

                total_loss.append(loss)
                if self.args.data == 'DJ30':
                    # calculate the accuracy
                    pred = outputs
                    true = batch_y
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    if self.args.loss == 'CE':
                        true = np.argmax(true, axis=-1)[..., np.newaxis]
                        pred = np.argmax(pred, axis=-1)[..., np.newaxis]
                        correct = np.sum(pred == true)
                    else:
                        # if true and pred have the same sign, then the prediction is correct
                        correct = np.sum(np.sign(pred) == np.sign(true))
                    train_acc_count += correct
                    total_samples += pred.shape[0] * pred.shape[1] * pred.shape[2]
        # # drop the last nan loss to avoid nan loss, the loss type is a pytorch tensor
        # # if the loss is a nan value tensor
        # total_loss =
        # total_loss = np.average(total_loss)
        # # report the number of nan loss
        # print('nan loss count:',  )
        # Step 1: Check for NaN values
        # Assuming total_loss is a list of tensors
        # Convert the list of tensors to a single tensor
        total_loss_tensor = torch.cat([loss.unsqueeze(0) for loss in total_loss])
        total_acc = train_acc_count / total_samples
        average_loss = total_loss_tensor.mean().item()
        # # Now check for NaN values in the tensor
        # nan_mask = torch.isnan(total_loss_tensor)
        #
        # # Filter out NaN values
        # filtered_losses = total_loss_tensor[~nan_mask]
        #
        # # Calculate the average of the remaining values
        # if len(filtered_losses) > 0:
        #     average_loss = filtered_losses.mean().item()  # Using PyTorch's mean() for tensor
        # else:
        #     average_loss = float('nan')
        #
        # # Count and report the number of NaN values
        # nan_count = nan_mask.sum().item()
        # print('nan loss count:', nan_count)
        #
        # # Use the calculated average_loss as needed
        # print('Average Loss (excluding NaNs):', average_loss)

        self.model.train()
        return average_loss, total_acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in tqdm(range(self.args.train_epochs), desc='Epochs'):
            iter_count = 0
            train_loss = []
            train_acc_count = 0
            train_samples = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), desc='Batches',
                                                                          total=train_steps):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        try:
                            f_dim = train_data.target_dim * -1
                            # print("f_dim:", f_dim)
                        except:
                            pass
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # print('outputs:', outputs[:5,:,:], 'batch_y:', batch_y[:5,:,:])
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        if self.args.data == 'DJ30':
                            # calculate the accuracy
                            pred = outputs
                            true = batch_y
                            pred = pred.detach().cpu().numpy()
                            true = true.detach().cpu().numpy()
                            if self.args.loss == 'CE':
                                true = np.argmax(true, axis=-1)[..., np.newaxis]
                                pred = np.argmax(pred, axis=-1)[..., np.newaxis]
                                correct = np.sum(pred == true)
                            else:
                                # if true and pred have the same sign, then the prediction is correct
                                correct = np.sum(np.sign(pred) == np.sign(true))
                            train_acc_count += correct
                            train_samples += pred.shape[0] * pred.shape[1] * pred.shape[2]

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    try:
                        f_dim = train_data.target_dim * -1
                        # print("f_dim:", f_dim)
                    except:
                        pass
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # print('outputs:', outputs.shape, 'batch_y:', batch_y.shape)
                    if self.args.loss == 'CE':

                        outputs_ = outputs.squeeze(1)
                        batch_y_ = batch_y.squeeze(1)

                        # Convert the one-hot encoded target to class indices
                        target_indices = torch.argmax(batch_y_, dim=1)
                        loss = criterion(outputs_, target_indices)
                    else:
                        loss = criterion(outputs, batch_y)
                    # print('outputs:', outputs[:5, :, :], 'batch_y:', batch_y[:5, :, :])
                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    if self.args.data == 'DJ30':
                        # calculate the accuracy
                        pred = outputs
                        true = batch_y
                        pred = pred.detach().cpu().numpy()
                        true = true.detach().cpu().numpy()
                        if self.args.loss == 'CE':
                            true = np.argmax(true, axis=-1)[..., np.newaxis]
                            pred = np.argmax(pred, axis=-1)[..., np.newaxis]
                            correct = np.sum(pred == true)
                        else:
                            # if true and pred have the same sign, then the prediction is correct
                            correct = np.sum(np.sign(pred) == np.sign(true))
                        train_acc_count += correct
                        train_samples += pred.shape[0] * pred.shape[1] * pred.shape[2]

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_acc = train_acc_count / train_samples
            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)
            # print('epoch:', epoch + 1, 'train_loss:', train_loss, 'vali_loss:', vali_loss, 'test_loss:', test_loss)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f},  Train Acc: {5:.7f}, Vali Acc: {6:.7f}, Test Acc: {7:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss, train_acc, vali_acc, test_acc))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, data_flag='test'):
        test_data, test_loader = self._get_data(flag=data_flag)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' + data_flag + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # if loss is BCE or CE, the output is the probability of each class
                # so we need to convert the output to the class label
                # print('output shape:', outputs.shape, 'output:', outputs[0, :5, :])
                # output is a size of (batch_size, pred_len, num_class), we need to convert it to (batch_size, pred_len,1)
                # if self.configs.loss == 'BCE' or self.configs.loss == 'CE':
                #     outputs = torch.argmax(outputs, dim=-1).unsqueeze(-1)
                # print('output shape:', outputs.shape, 'output:', outputs[0, :5, :])

                # print('batch_x:', batch_x.shape, 'batch_y:', batch_y.shape, 'outputs:', outputs.shape)


                f_dim = -1 if self.args.features == 'MS' else 0
                try:
                    f_dim = test_data.target_dim * -1
                    # print("f_dim:", f_dim)
                except:
                    pass
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    # print('batch_y:', batch_y.shape)
                    if self.args.data != 'DJ30':
                        # for DJ30, the output is return, so no need to inverse
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    else:
                        # batch_y  = np.concatenate((test_data.inverse_transform(batch_y[:, :, :-1*test_data.target_dim]), batch_y[:, :, -1*test_data.target_dim:]), axis=-1) # do not inverse the target
                        pass


                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse and self.args.data != 'DJ30':
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # print('preds preivew:', preds[:50, :, :, :])
        if self.args.loss == 'CE':
            trues = np.argmax(trues, axis=-1)[..., np.newaxis]
            preds = np.argmax(preds, axis=-1)[..., np.newaxis]
        # plot the all-in-one result
        if test_data.scale and self.args.inverse and self.args.data != 'DJ30':
            shape = preds.shape
            all_in_one_trues = test_data.inverse_transform(trues.squeeze(0)).reshape(shape)
            if self.args.data != 'DJ30':
                # for DJ30, the output is return, so no need to inverse
                all_in_one_preds = test_data.inverse_transform(preds.squeeze(0)).reshape(shape)
        else:
            all_in_one_preds = preds
            all_in_one_trues = trues
        # print('true shape:', trues.shape, 'pred shape:', preds.shape)
        # print("all_in_one_trues shape:", all_in_one_trues[0, :, -1].shape, "all_in_one_preds shape:", all_in_one_preds[0, :, -1].shape)
        cutdown = 100
        for i in range(all_in_one_trues.shape[0] // cutdown):
            visual(all_in_one_trues[i * cutdown: (i + 1) * cutdown, :, -1].reshape(-1),
                   all_in_one_preds[i * cutdown: (i + 1) * cutdown, :, -1].reshape(-1),
                   os.path.join(folder_path, 'all_in_one_' + str(i) + '.pdf'), line_width=1)
        visual(all_in_one_trues.reshape(-1), all_in_one_preds.reshape(-1), os.path.join(folder_path, 'all_in_one.pdf'),
               line_width=1)



        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)
        # print('preds preivew:', preds[:50, :, -1])

        # plot the predicted price recovered from the return prediction if the data is DJ30
        if self.args.data == 'DJ30' and self.args.target == 'ret1':
            # revse the return to price
            recoverd_close_price = test_data.recover_close_price_from_return(preds)
            # plot the recoverd close price by segment
            cutdown = 100
            for i in range(recoverd_close_price.shape[0] // cutdown):
                visual(recoverd_close_price[i * cutdown: (i + 1) * cutdown],
                       test_data.close_price.values[i * cutdown: (i + 1) * cutdown],
                       os.path.join(folder_path, 'close_price_' + str(i) + '.pdf'), line_width=1)
            visual(recoverd_close_price, test_data.close_price.values, os.path.join(folder_path, 'close_price.pdf'))
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # check if there is nan in the prediction
        nan_mask = np.isnan(preds)
        nan_count = np.sum(nan_mask)
        print('nan count:', nan_count)
        # print the nan index
        if nan_count > 0:
            nan_index = np.argwhere(nan_mask)
            print('nan index:', nan_index)
            # print the input of the nan prediction
            print('nan input:', test_data.data[-nan_index[0][0] - 1, -nan_index[0][1] - 1, :])
        # print('preview of the prediction:', preds[0, :5, -1])
        # print('preview of the true:', trues[0, :5, -1]
        if self.args.features == 'MS':
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        print('true shape:', trues.shape, 'pred shape:', preds.shape)
        mae, mse, rmse, mape, mspe, acc = metric(preds.reshape(-1), trues.reshape(-1))
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, acc:{}'.format(mse, mae, rmse, mape, mspe, acc))
        if data_flag == 'test':
            f = open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, acc:{}\n'.format(mse, mae, rmse, mape, mspe, acc))
            f.write('\n')
            f.write('\n')
            f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

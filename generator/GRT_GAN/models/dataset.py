# -*- coding: UTF-8 -*-
import numpy as np
import torch

class TimeGANDataset(torch.utils.data.Dataset):
    """TimeGAN Dataset for sampling data with their respective time

    Args:
        - data (numpy.ndarray): the padded dataset to be fitted (D x S x F)
        - time (numpy.ndarray): the length of each data (D)
    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data 
    """
    def __init__(self, data, time=None, mask=None, padding_value=None):
        # sanity check
        if len(data) != len(time):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )

        if isinstance(time, type(None)):
            time = [len(x) for x in data]

        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)
        # mask is get from 
        # mask = np.isnan(data)
        self.mask = torch.BoolTensor(mask) if mask is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.mask[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]
        
        # The actual length of each data
        T_mb = [T for T in batch[1]]

        # Masking
        if self.mask is not None:
            M_mb = [M for M in batch[2]]
            return X_mb, T_mb, M_mb
        else:
            return X_mb, T_mb


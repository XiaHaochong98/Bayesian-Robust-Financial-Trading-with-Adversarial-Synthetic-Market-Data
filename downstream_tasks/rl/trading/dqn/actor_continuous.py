import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from einops import rearrange
from torch.distributions import Normal

ROOT = str(Path(__file__).resolve().parents[1])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)
from embed import TimesEmbed, TokenEmbedding, TemporalEmbedding

class ActorContinuous(nn.Module):
    def __init__(self,
                 *args,
                 input_dim=156,
                 timestamps=10,
                 embed_method="mean",
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 output_dim=3,
                 temporals_name=["day", "weekday", "month"],
                 device=torch.device("cuda"),
                 **kwargs
                 ):
        super(ActorContinuous, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.output_dim = output_dim
        self.temporal_dim = len(temporals_name)

        self.value_embedding = TokenEmbedding(input_dim=input_dim - self.temporal_dim,
                                              embed_dim=embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim,
                                                    temporals_name=temporals_name,
                                                    embed_type='fixed')

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, 
                            num_layers=depth, batch_first=True)

        self.norm = norm_layer(embed_dim)

        self.fc_mean = nn.Linear(
            embed_dim,
            self.output_dim,
            bias=True,
        )
        
        self.fc_std = nn.Linear(
            embed_dim,
            self.output_dim,
            bias=True,
        )

        self.initialize_weights()

        self.to(device)
        self.device = device

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.device)
        x_shape = x.shape
        if len(x_shape) > 3:
            x = x.reshape(-1, *x.shape[-2:])
        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]
        x = self.value_embedding(feature) + self.temporal_embedding(temporal)

        # apply Transformer blocks
        x, (h0, c0) = self.lstm(x)
        x = self.norm(x)

        mean = torch.tanh(self.fc_mean(x))
        std = torch.sigmoid(self.fc_std(x))
        dist = Normal(mean, std)
        
        action = dist.sample()
        action = torch.tanh(action)
        action = action.reshape(*x_shape[:-2], *action.shape[-2:])
        return action
    
    def evaluate_actions(self, x, actions):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions).to(self.device)
        x_shape = x.shape
        if len(x.shape) > 3:
            x = x.reshape(-1, *x.shape[-2:])
        if len(actions.shape) > 3:
            actions = actions.reshape(-1, *actions.shape[-2:])
        actions = torch.atanh(actions)
        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]
        x = self.value_embedding(feature) + self.temporal_embedding(temporal)

        # apply Transformer blocks
        x, (h0, c0) = self.lstm(x)
        x = self.norm(x)

        mean = torch.tanh(self.fc_mean(x))
        std = torch.sigmoid(self.fc_std(x))
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions) - torch.log(1 - torch.tanh(actions).pow(2) + 1e-6)
        log_prob = log_prob.reshape(*x_shape[:-2], *log_prob.shape[-2:])
        
        return log_prob.sum(dim=-1).sum(dim=-1)
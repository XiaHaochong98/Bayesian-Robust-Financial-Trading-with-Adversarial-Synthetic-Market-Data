import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.layers import Mlp
from einops import rearrange

ROOT = str(Path(__file__).resolve().parents[1])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)
from embed import TimesEmbed, TokenEmbedding, TemporalEmbedding

class QuantileBelief(nn.Module):
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
        super(QuantileBelief, self).__init__()
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
            raise NotImplementedError

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, 
                            num_layers=depth, batch_first=True)

        self.norm = norm_layer(embed_dim)

        self.decoder_pred = nn.Linear(
            embed_dim,
            self.output_dim,
            bias=True,
        )

        self.initialize_weights()

        self.to(device)

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

    def forward_encoder(self, x):

        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]
        x = self.value_embedding(feature) + self.temporal_embedding(temporal)

        # apply Transformer blocks
        x, (h0, c0) = self.lstm(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_pred(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        return pred



class QNet(nn.Module):
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
                 use_quantile_belief=False,
                 quantile_heads_num=0,
                 device=torch.device("cuda"),
                 **kwargs
                 ):
        super(QNet, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.output_dim = output_dim
        self.use_quantile_belief = use_quantile_belief
        self.quantile_heads_num = quantile_heads_num

        self.patch_embed = TimesEmbed(
            timestamps=timestamps,
            input_dim=input_dim,
            embed_dim=embed_dim,
            embed_method=embed_method,
            temporals_name=temporals_name,
        )

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim,
                    hidden_features=embed_dim,
                    act_layer=nn.Tanh,
                    out_features=embed_dim)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_pred = nn.Linear(
            embed_dim,
            self.output_dim,
            bias=True,
        )
        
        if self.use_quantile_belief:
            self.belief_embedding = nn.Embedding(self.quantile_heads_num, embed_dim)

        self.initialize_weights()

        self.to(device)

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

    def forward_encoder(self, x):
        # b, c, d, f = x.shape  # num envs, 1, timestamps, features
        x = self.patch_embed(x)

        b, c, f = x.shape  # num envs, 1, features

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_pred(x)
        return x

    def forward(self, x, quantile_belief=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        latent = self.forward_encoder(x)
        if self.use_quantile_belief:
            belief_logits = self.belief_embedding(quantile_belief).unsqueeze(1)
            latent = latent + belief_logits
        pred = self.forward_decoder(latent).squeeze(-1)
        pred = pred.squeeze(1)
        return pred


if __name__ == '__main__':
    device = torch.device("cpu")

    model = QNet(
        input_dim=156,
        timestamps=10,
        embed_dim=128,
        depth=2,
        cls_embed=False,
        output_dim=3,
        device=device,
    )

    feature = torch.randn(4, 4, 10, 153)
    temporal = torch.zeros(4, 4, 10, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    pred = model(batch)  # torch.Size([4, 4, 3])
    print(pred.shape)

    feature = torch.randn(4, 10, 153)
    temporal = torch.zeros(4, 10, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    pred = model(batch)
    print(pred.shape)  # torch.Size([4, 3])

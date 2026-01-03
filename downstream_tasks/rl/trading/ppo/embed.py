import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
import torch
import math
import torch.nn as nn
from einops import rearrange
from torch.nn import ModuleDict


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=input_dim,
                                   out_channels=embed_dim,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(input_dim, embed_dim).float()
        w.require_grad = False

        position = torch.arange(0, input_dim).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(input_dim, embed_dim)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type='timeF'):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(3, embed_dim, bias=False)

    def forward(self, x):
        return self.embed(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type='fixed', temporals_name=["day", "weekday", "month"]):
        super(TemporalEmbedding, self).__init__()

        self.temporals_name = temporals_name
        self.size_maps = {
            'day': 32,
            'weekday': 7,
            'month': 13,
            'hour': 25,
            'minute': 61
        }
        self.embed = ModuleDict()

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        for item in temporals_name:
            assert item in self.size_maps.keys(), f"Temporal name {item} not in {self.size_maps.keys()}"
            self.embed[item] = Embed(self.size_maps[item], embed_dim)

    def forward(self, x):
        x = x.long()

        embeds = []
        for index, item in enumerate(self.temporals_name):
            embeds.append(self.embed[item](x[:, :, index]))

        return torch.stack(embeds, dim=-2).sum(dim=-2)


class TimesEmbed(nn.Module):
    def __init__(self,
                 *args,
                 timestamps: int = 10,
                 input_dim: int = 156,
                 embed_dim: int = 128,
                 embed_type='fixed',
                 temporals_name=["day", "weekday", "month"],
                 **kwargs
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.temporal_dim = len(temporals_name)
        self.embed_dim = embed_dim
        self.timestamps = timestamps

        self.feature_dim = self.input_dim - self.temporal_dim

        self.value_embedding = TokenEmbedding(input_dim=self.feature_dim,
                                              embed_dim=embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim,
                                                    temporals_name=temporals_name,
                                                    embed_type=embed_type) if embed_type != 'timeF' else \
            TimeFeatureEmbedding(embed_dim=embed_dim, embed_type=embed_type)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        b, c, d, f = x.shape  # batch size, num envs, timestamps, features

        x = rearrange(x, "b c d f -> (b c) d f", b=b, c=c)

        feature = x[..., :-self.temporal_dim]
        temporal = x[..., -self.temporal_dim:]

        x = self.value_embedding(feature) + self.temporal_embedding(temporal) + self.position_embedding(feature)

        x = rearrange(x, "(b c) d f -> b c d f", b=b, c=c)

        x = x.mean(dim=-2)

        return x


if __name__ == '__main__':
    device = torch.device("cpu")

    model = TimesEmbed(
        timestamps=10,
        input_dim=156,
        embed_dim=128,
        embed_type='fixed',
        temporals_name=["day", "weekday", "month"]
    ).to(device)

    feature = torch.randn(4, 4, 10, 153)
    temporal = torch.zeros(4, 4, 10, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    res = model(batch)
    print(res.shape)

    feature = torch.randn(4, 10, 153)
    temporal = torch.zeros(4, 10, 3)
    batch = torch.cat([feature, temporal], dim=-1)
    res = model(batch)
    print(res.shape)

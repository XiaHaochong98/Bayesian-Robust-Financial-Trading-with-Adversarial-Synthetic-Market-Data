import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(CURRENT)

from qnet import QNet
from actor import Actor


class Agent(nn.Module):
    def __init__(self,
                 *args,
                 input_dim=156,
                 timestamps=10,
                 embed_method="mean",
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 action_dim=3,
                 temporals_name=["day", "weekday", "month"],
                 device=torch.device("cuda"),
                 **kwargs
                 ):
        super(Agent, self).__init__()

        self.q1 = QNet(
            input_dim=input_dim,
            timestamps=timestamps,
            embed_method=embed_method,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
            temporals_name=temporals_name,
            device=device,
        ).to(device)

        self.q1_target = QNet(
            input_dim=input_dim,
            timestamps=timestamps,
            embed_method=embed_method,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
            temporals_name=temporals_name,
            device=device,
        ).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())

        self.q2 = QNet(
            input_dim=input_dim,
            timestamps=timestamps,
            embed_method=embed_method,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
            temporals_name=temporals_name,
            device=device,
        ).to(device)

        self.q2_target = QNet(
            input_dim=input_dim,
            timestamps=timestamps,
            embed_method=embed_method,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
            temporals_name=temporals_name,
            device=device,
        ).to(device)

        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor = Actor(
            input_dim=input_dim,
            timestamps=timestamps,
            embed_method=embed_method,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
            temporals_name=temporals_name,
            device=device,
        ).to(device)

        self.device = device

    def forward(self, *input, **kwargs):
        pass


if __name__ == '__main__':
    device = torch.device("cpu")

    model = Agent(
        input_dim=156,
        timestamps=10,
        embed_dim=128,
        depth=1,
        cls_embed=False,
        action_dim=3,
        device=device,
    )

    feature = torch.randn(4, 4, 10, 153)
    temporal = torch.zeros(4, 4, 10, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    action = model.actor(batch)
    print(action.shape)

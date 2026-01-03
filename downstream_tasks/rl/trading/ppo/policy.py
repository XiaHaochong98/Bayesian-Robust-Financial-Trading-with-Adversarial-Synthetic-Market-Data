import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.categorical import Categorical

CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(CURRENT)

from actor import Actor
from critic import Critic


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

        self.critic = Critic(
            input_dim=input_dim,
            timestamps=timestamps,
            embed_method=embed_method,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=1,
            temporals_name=temporals_name,
            device=device,
        ).to(device)

        self.device = device

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)

        if len(logits.shape) == 2:
            logits = logits.unsqueeze(1)

        b, c, n = logits.shape

        logits = rearrange(logits, "b c n -> (b c) n", b=b, c=c, n=n)

        dis = Categorical(logits=logits)

        if action is None:
            action = dis.sample()

        probs = dis.log_prob(action)
        entropy = dis.entropy()
        value = self.critic(x)

        action = rearrange(action, "(b c) -> b c", b=b, c=c).squeeze(1)
        probs = rearrange(probs, "(b c) -> b c", b=b, c=c).squeeze(1)
        entropy = rearrange(entropy, "(b c) -> b c", b=b, c=c).squeeze(1)

        return action, probs, entropy, value

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
    action, probs, entropy, value = model.get_action_and_value(batch)
    print(action.shape, probs.shape, entropy.shape,
          value.shape)  # torch.Size([4, 4, 30]) torch.Size([4, 4]) torch.Size([4, 4]) torch.Size([4, 4])
    print(action)

    feature = torch.randn(4, 10, 153)
    temporal = torch.zeros(4, 10, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    action, probs, entropy, value = model.get_action_and_value(batch)
    print(action.shape, probs.shape, entropy.shape,
          value.shape)  # torch.Size([4, 30]) torch.Size([4]) torch.Size([4]) torch.Size([4])

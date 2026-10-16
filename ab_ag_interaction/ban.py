"""Low-rank BAN interaction-map generator, matching the referenced BANLayer."""

import torch
from torch import Tensor, nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    def __init__(self, dims, act: str = "ReLU", dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            if act:
                layers.append(getattr(nn, act)())
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class BANLayer(nn.Module):
    """Factorized bilinear antigen--antibody residue-pair score map."""

    def __init__(self, v_dim: int, q_dim: int, h_dim: int = 256,
                 h_out: int = 1, act: str = "ReLU", dropout: float = 0.2, k: int = 3):
        super().__init__()
        self.h_out, self.k, self.c = h_out, k, 32
        self.v_net = FCNet([v_dim, h_dim * k], act, dropout)
        self.q_net = FCNet([q_dim, h_dim * k], act, dropout)
        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.empty(1, h_out, 1, h_dim * k).normal_())
            self.h_bias = nn.Parameter(torch.empty(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * k, h_out), dim=None)

    def forward(self, v: Tensor, q: Tensor, return_features: bool = False):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        if self.h_out <= self.c:
            maps = torch.einsum("xhyk,bvk,bqk->bhvq", self.h_mat, v_, q_) + self.h_bias
        else:
            d_ = v_.transpose(1, 2).unsqueeze(3) * q_.transpose(1, 2).unsqueeze(2)
            maps = self.h_net(d_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (maps, v_, q_) if return_features else maps

    def bilinear_pool(self, maps: Tensor, v_features: Tensor, q_features: Tensor) -> Tensor:
        """Original BAN low-rank pooling, one output vector per attention glimpse."""
        joint = torch.einsum("bhvq,bvk,bqk->bhk", maps, v_features, q_features)
        return joint.view(*joint.shape[:2], -1, self.k).sum(dim=-1)

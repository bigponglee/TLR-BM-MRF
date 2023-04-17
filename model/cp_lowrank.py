import torch
import torch.nn as nn
from model.base import conv1x1
from configs import TLR_BM_configs as cfg


class CPC(nn.Module):
    '''
    CP component modules (CPC)
    '''

    def __init__(self, T=cfg.input_dim, Nx=cfg.data_shape[0], Ny=cfg.data_shape[1]) -> None:
        super().__init__()
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv1x1(T, T),
            nn.Softsign()
        )  # channel-wise
        self.b = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv1x1(Nx, Nx),
            nn.Softsign()
        )  # Nx
        self.c = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv1x1(Ny, Ny),
            nn.Softsign()
        )  # Ny

    def forward(self, x):
        a = self.a(x)  # [batch, 2T, 1, 1]
        b = self.b(x.permute(0, 2, 1, 3))  # [batch, Nx, 1, 1]
        b = b.permute(0, 2, 1, 3)  # [batch, 1, Nx, 1]
        c = self.c(x.permute(0, 3, 1, 2))  # [batch, Ny, 1, 1]
        c = c.permute(0, 2, 3, 1)  # [batch, 1, 1, Ny]

        ab = torch.einsum('btij,bixj->btxj', a, b)  # [batch, 2T, Nx, 1]
        abc = torch.einsum('btxj,bijy->btxy', ab, c)  # [batch, 2T, Nx, Ny]
        return abc


class CP_lowrank(nn.Module):
    '''
    CP low-rank tensor
    '''

    def __init__(self, R=cfg.rank, T=cfg.input_dim) -> None:
        super().__init__()
        self.R = R
        self.cp_frac = nn.ModuleList([nn.Sequential(
            CPC(),
            nn.BatchNorm2d(T)
        ) for _ in range(R)])
        self.tensor_weight = nn.Parameter(torch.ones(R)).type(
            cfg.dtype_float).to(cfg.device)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        '''
        Args:
            x: [batch_size, 2*T, Nx, Ny]
        Returns:
            x_cp: [batch_size, 2*T, Nx, Ny]
        '''
        x_cp = x*1.0  # init
        out_low_rank_tensor = []
        for cp_frac in self.cp_frac:
            x_out = cp_frac(x_cp)
            x_cp = x_cp - x_out
            out_low_rank_tensor.append(x_out)
        out_low_rank_tensor = torch.stack(
            out_low_rank_tensor, dim=1)  # [batch_size, R, 2*T, Nx, Ny]
        weight = self.softmax(self.tensor_weight)  # [R]
        weight = weight.view(1, self.R, 1, 1, 1)  # [1, R, 1, 1, 1]
        out_low_rank_tensor = out_low_rank_tensor * \
            weight  # [batch_size, R, 2*T, Nx, Ny]
        out_low_rank_tensor = torch.sum(
            out_low_rank_tensor, dim=1)  # [batch_size, 2*T, Nx, Ny]
        return out_low_rank_tensor + x

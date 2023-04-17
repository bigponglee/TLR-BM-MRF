from model.decoder import Decoder
from model.encoder import Encoder
import torch
from configs import TLR_BM_configs as cfg
from model.cp_lowrank import CP_lowrank
from torch.nn.init import trunc_normal_, constant_
from model.base import conv1x1


class BM_module(torch.nn.Module):
    '''
    Bloch manifold module: encoder + decoder
    '''

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(inchannel=cfg.input_dim,
                               outchannel=cfg.para_maps, channels_list=[128, 64, 16, 16, 8, 8, 8, 4, 4])
        self.decoder = Decoder(inchannel=cfg.para_maps,
                               outchannel=cfg.input_dim, channels_list=[4, 4, 8, 8, 8, 16, 16, 64, 128])

    def forward(self, atb):
        '''
        Args:
            atb: [batch_size, 2*T, Nx, Ny]
        Returns:
            x: [batch_size, 2*T, Nx, Ny]
        '''
        x_map = self.encoder(atb)
        x = self.decoder(x_map)
        return x, x_map


class TLR_BM_NET(torch.nn.Module):

    def __init__(self, depth=cfg.depth, dc=None, ed_coder=None) -> None:
        super().__init__()
        self.ed_coder = torch.nn.ModuleList([ed_coder() for _ in range(depth)])
        self.depth = depth
        self.dc_list = torch.nn.ModuleList([dc() for _ in range(depth)])
        self.cp_list = torch.nn.ModuleList(
            [CP_lowrank(R=cfg.rank, T=cfg.input_dim) for _ in range(depth)])
        self.final_map = torch.nn.Sequential(
            conv1x1(cfg.para_maps, cfg.para_maps, groups=cfg.para_maps),
            torch.nn.BatchNorm2d(cfg.para_maps),
            torch.nn.ELU(inplace=True),
            conv1x1(cfg.para_maps, cfg.para_maps, groups=cfg.para_maps),
            torch.nn.BatchNorm2d(cfg.para_maps),
            torch.nn.ELU(inplace=True),
            conv1x1(cfg.para_maps, cfg.para_maps, groups=cfg.para_maps),
            torch.nn.Softmax(dim=0)
        )

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, atb):
        '''
        Args:
            atb: [batch_size, 2*T, Nx, Ny]
        Returns:
            x_recon: [x] [batch_size, 2*T, Nx, Ny]
            x_map: parameter maps [batch_size, 2, Nx, Ny]
            ata_x: ata(X) [batch_size, 2*T, Nx, Ny]
            x_sym: 变换算子正交性 [batch_size, 2*T, Nx, Ny]
            svd_flag: svd是否成功执行，0表示成功，1表示失败 
        '''
        out_map = []
        ata_x = []
        x = atb * 1.0
        for i in range(self.depth):
            x, x_m = self.ed_coder[i](x)
            out_map.append(x_m)
            x = self.cp_list[i](x)
            x, x_ata = self.dc_list[i](x, atb)
            ata_x.append(x_ata)
        x_map = torch.cat(out_map, dim=0)  # [depth, 2, Nx, Ny]
        weight_map = self.final_map(x_map)  # [depth, 2, Nx, Ny]
        x_map = torch.sum(x_map * weight_map, dim=0,
                          keepdim=True)  # [1, 2*T, Nx, Ny]
        return x, x_map, ata_x

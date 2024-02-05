'''
data consistency
'''
import torch
import torchkbnufft as tkbn
from configs import TLR_BM_configs as cfg
import os
import scipy.io as sio
from model.base import complex2real, real2complex
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def toep_op_init(data_shape=cfg.data_shape, csm_path=cfg.csm_path,
                 ktraj_path=cfg.ktraj_path, grid_factor=2,
                 numpoints=4):
    '''
        Args:
            data_shape: [Nx, Ny, T]
            csm_path: coil sensitivity map path
            ktraj_path: k-space trajectory path
            grid_factor: grid factor, default 2
            numpoints: number of points, default 4, 越小NUFFT计算速度越快
            norm: nufft norm 方法, default 'ortho'
        '''
    im_size = (data_shape[0], data_shape[1])
    grid_size = (data_shape[0]*grid_factor, data_shape[1]*grid_factor)
    T = data_shape[2]
    csm = torch.from_numpy(sio.loadmat(csm_path)['csm_maps']).type(
        cfg.dtype_complex).to(cfg.device)
    csm = csm.expand(T, -1, -1, -1)  # [T, ncoil, Nx, Ny]
    ktraj = torch.from_numpy(sio.loadmat(ktraj_path)['ktraj_nufft']).type(
        cfg.dtype_float).to(cfg.device)
    ktraj = ktraj[:T, :, :]
    norm = 'ortho'
    dcomp = tkbn.calc_density_compensation_function(
        ktraj=ktraj, im_size=im_size, grid_size=grid_size, numpoints=numpoints).to(cfg.device)
    kernel = tkbn.calc_toeplitz_kernel(
        ktraj, im_size, weights=dcomp, norm=norm, grid_size=grid_size, numpoints=numpoints).to(cfg.device)
    toep_op = tkbn.ToepNufft().to(cfg.device)
    return toep_op, kernel, csm, norm, T


toep_op, kernel, csm, norm, T = toep_op_init(data_shape=cfg.data_shape, csm_path=cfg.csm_path,
                                             ktraj_path=cfg.ktraj_path, grid_factor=2,
                                             numpoints=4)


class Data_Consistency(torch.nn.Module):
    '''
    data consistency
    '''

    def __init__(self) -> None:
        super().__init__()
        self.mu_max = cfg.data_shape[0] * \
            cfg.data_shape[1]/cfg.k_sample_points  # N/M
        self.mu_parameter = torch.nn.Parameter(
            torch.Tensor([0.01]))

    def forward(self, x, atb):
        '''
        Args:
            x: [batch_size, 2*T, Nx, Ny]
            ata_x: [batch_size, 2*T, Nx, Ny] 用于计算atb
        '''
        x_complex = real2complex(x, T)
        x_complex = toep_op(x_complex, kernel, smaps=csm, norm=norm)  # ata x
        ata_x = complex2real(x_complex)
        x = x + self.mu_parameter*(atb - ata_x)
        return x, ata_x


def gen_atb(x):
    '''
    Args:
        x: [batch_size, 2*T, Nx, Ny]
    Returns:
        atb: [batch_size, 2*T, Nx, Ny]
    '''
    x_complex = real2complex(x, T)
    x_complex = toep_op(x_complex, kernel, smaps=csm, norm=norm)
    atb = complex2real(x_complex)
    return atb

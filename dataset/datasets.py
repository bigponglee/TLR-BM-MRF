import scipy.io as sio
import torch
from configs import TLR_BM_configs as cfg
from random import shuffle


def load_data(data_path=cfg.data_path, data_index=None, data_name=None):
    '''
    Args:
    data_path: 数据路径 cfg.dataset_path = '/data/MRF_dataset/'
    data_index: 数据集索引 1-380
    data_name: 数据名称 'X'  
    Returns:
    X: loaded数据 Tensor [1,2*T,Kx,Ky]
    para_maps: loaded参数图 Tensor [1,para_maps,Kx,Ky]
    '''
    X = sio.loadmat(
        data_path+'/{}/{}_{}.mat'.format(data_name, data_name, data_index))[data_name]  # [Kx,Ky,T]
    X = torch.from_numpy(X).type(cfg.dtype_complex).to(cfg.device)  # [Kx,Ky,T]
    X = torch.permute(X, (2, 0, 1))  # [T,Kx,Ky]
    max_X = torch.max(torch.abs(X))
    X = X/max_X
    X = torch.cat((X.real, X.imag), dim=0)  # [2*T,Kx,Ky]
    X = X.unsqueeze(0)  # [1,2*T,Kx,Ky]

    para_maps = sio.loadmat(
        cfg.data_path+'/para_maps/para_map_{}.mat'.format(data_index))['para_maps']  # [Kx,Ky,3]
    para_maps = para_maps[:, :, :cfg.para_maps]  # [Kx,Ky,para_maps]
    para_maps[:, :, 0] = para_maps[:, :, 0] / \
        cfg.T1_max  # [Kx,Ky,para_maps] 归一化
    para_maps[:, :, 1] = para_maps[:, :, 1] / \
        cfg.T2_max  # [Kx,Ky,para_maps] 归一化
    para_maps = torch.from_numpy(para_maps).type(
        cfg.dtype_float).to(cfg.device)  # [Kx,Ky,para_maps]
    para_maps = torch.permute(para_maps, (2, 0, 1))  # [para_maps,Kx,Ky]
    para_maps = para_maps.unsqueeze(0)  # [1,para_maps,Kx,Ky]
    return X, para_maps, max_X


def train_dataloader(data_index_list, data_path, data_name, batch_size_default=cfg.batch_size):
    '''
    由加载的数据构造训练数据集
    Args:
    data_index_list: 数据集索引列表
    data_path: 数据路径 cfg.dataset_path = '/media/deep/D/MRF_dataset/'
    data_path = cfg.dataset_path + 'X/X_'
    data_name: 数据名称 'X'
    batch_size_default: 1
    Returns:
    X: loaded数据 Tensor [batch_size,2*T,Kx,Ky]
    para_maps: loaded参数图 Tensor [batch_size,para_maps,Kx,Ky]
    '''
    shuffle(data_index_list)  # 打乱数据集
    if batch_size_default != 1:
        raise ValueError('batch_size_default must be 1')
    for data_index in data_index_list:
        X, para_maps, max_X = load_data(data_path, data_index, data_name)
        yield (X, para_maps, max_X)


def test_dataloader(data_index_list, data_path, data_name, batch_size_default=cfg.batch_size):
    '''
    由加载的数据构造测试数据集
    Args:
    data_index_list: 数据集索引列表
    data_path: 数据路径 cfg.dataset_path = '/media/deep/D/MRF_dataset/'
    data_path = cfg.dataset_path + 'X/X_'
    data_name: 数据名称 'X'
    batch_size_default: 1
    Returns:
    X: loaded数据 Tensor [batch_size,2*T,Kx,Ky]
    para_maps: loaded参数图 Tensor [batch_size,para_maps,Kx,Ky]
    '''
    if batch_size_default != 1:
        raise ValueError('batch_size_default must be 1')
    for data_index in data_index_list:
        X, para_maps, max_X = load_data(data_path, data_index, data_name)
        yield (X, para_maps, max_X, data_index)


def rebuild_paramap(para_est):
    '''
    rebuild parameter maps
    '''
    para_est = para_est.permute(0, 2, 3, 1)  # [batch_size,Kx,Ky,para_maps]
    para_est = torch.squeeze(para_est, 0)  # [Kx,Ky,para_maps]
    para_est = para_est.detach().cpu().numpy()
    para_est[:, :, 0] = para_est[:, :, 0]*cfg.T1_max
    para_est[:, :, 1] = para_est[:, :, 1]*cfg.T2_max
    return para_est

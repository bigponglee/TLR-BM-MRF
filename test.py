'''
train
'''
import torch
import os
from utils.func import def_logger
from model.TLR_BM_Net import TLR_BM_NET, BM_module
from model.dc import Data_Consistency, gen_atb
from dataset.datasets import test_dataloader, rebuild_paramap
from configs import TLR_BM_configs as cfg
import time
import scipy.io as sio


def test(data_index_list):
    #####################文件路径#########################################
    try:
        os.mkdir(cfg.OUT_dir)
    except:
        logger.info('文件夹已经存在！！！')
    OUT_dir = cfg.OUT_dir+'test/'
    try:
        os.mkdir(OUT_dir)
        logger = def_logger(OUT_dir+'/log.txt')
    except:
        logger = def_logger(OUT_dir+'/log.txt')
        logger.info('文件夹已经存在！！！')
    ####################构建网络#########################################
    Net = TLR_BM_NET(depth=cfg.depth, dc=Data_Consistency,
                 ed_coder=BM_module).to(cfg.device)
    loaded_params = torch.load(cfg.model_saved_path, map_location='cuda:{}'.format(cfg.cuda_device))
    Net.load_state_dict(loaded_params)
    logger.info('load decoder from: {}'.format(cfg.model_saved_path))
    logger.info(
        "===================================Net Build======================================")
    ####################损失函数#########################################
    Loss_para_maps = torch.nn.MSELoss().to(cfg.device)
    Loss_X = torch.nn.MSELoss().to(cfg.device)
    logger.info('using Loss Function: MSE Loss')
    ####################dataset#########################################
    data_path = cfg.data_path
    data_name = cfg.data_name
    logger.info(
        "===================================Testing START=====================================")
    iter_i = 0
    Net.train()
    for e in range(1):
        data_loader = test_dataloader(
            data_index_list, data_path, data_name, cfg.batch_size)
        for X, para_maps, max_X, data_index in data_loader:
            with torch.no_grad():
                atb = gen_atb(X)
                start_tim = time.time()
                X_recon, para_est, _ = Net(atb)
                end_tim = time.time()
            # reconstruction loss
            loss_X_recon= Loss_X(X_recon, X)
            loss_para_maps_final = Loss_para_maps(para_est, para_maps)
            X_recon = torch.complex(X_recon[:, :cfg.data_shape[2], :, :], X_recon[:, cfg.data_shape[2]:, :, :]).squeeze()*max_X
            X_recon = X_recon.permute(1, 2, 0).detach().cpu().numpy()
            para_map = rebuild_paramap(para_est)
            sio.savemat(OUT_dir+'X_recon_{}.mat'.format(data_index), {'X_recon': X_recon})
            sio.savemat(OUT_dir+'para_est_{}.mat'.format(data_index),
                        {'para_maps': para_map})
            logger.info('data_index: {}, X: {:.6f} Para: {:.6f}; time(s): {:.6f}'.format(
                data_index, loss_X_recon.item(), loss_para_maps_final.item(), end_tim-start_tim))
            torch.cuda.empty_cache()
            iter_i += 1
    logger.info(
        "===================================END======================================")


if __name__ == "__main__":
    test_index_list = [50]
    test(test_index_list)

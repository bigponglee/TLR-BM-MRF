'''
train
'''
import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from utils.func import def_logger
from model.TLR_BM_Net import TLR_BM_NET, BM_module
from model.dc import Data_Consistency, gen_atb
from dataset.datasets import train_dataloader
from configs import TLR_BM_configs as cfg
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast


def train(data_index_list):
    #####################文件路径##########################################
    OUT_dir = cfg.OUT_dir
    Save_NET_root = OUT_dir+'/Model_save'
    try:
        os.mkdir(OUT_dir)
        logger = def_logger(OUT_dir+'/log.txt')
    except:
        logger = def_logger(OUT_dir+'/log.txt')
        logger.info('文件夹已经存在！！！')
    try:
        os.mkdir(Save_NET_root)
    except:
        logger.info('文件夹已经存在！！！')
    shutil.copy('/configs/TLR_BM_configs.py', OUT_dir)  # 复制配置文件
    ####################可视化###########################################
    writer_train = SummaryWriter(log_dir=OUT_dir+'/runs')  # tensorboard
    logger.info('tensorboard saved at: {}'.format(OUT_dir+'/runs'))
    ####################构建网络#########################################
    Net = TLR_BM_NET(depth=cfg.depth, dc=Data_Consistency,
                     ed_coder=BM_module).to(cfg.device)
    if cfg.load_model: #load checkpoint
        Net.load_state_dict(torch.load(cfg.model_saved_path))
        logger.info('load decoder from: {}'.format(cfg.model_saved_path))
    if cfg.projection_op_saved_model != None: # load pre-trained BM module
        loaded_params_ed = torch.load(
            cfg.projection_op_saved_model, map_location='cuda:{}'.format(cfg.cuda_device))
        for i in range(cfg.load_depth):
            Net.ed_coder[i].load_state_dict(loaded_params_ed)
    logger.info(
        "===================================Net Build======================================")
    ####################优化器#########################################
    Optimizer = torch.optim.Adam([
        {'params': Net.parameters(), 'lr': cfg.LearningRate},
    ])
    if cfg.use_scheduler:  # 学习率衰减
        Scheduler_decoder = torch.optim.lr_scheduler.StepLR(
            Optimizer, step_size=1, gamma=0.95)
    logger.info('Optimizer: Adam, scheduler: {}'.format(False))
    ####################损失函数#########################################
    Loss_para_maps = torch.nn.MSELoss().to(cfg.device)
    Loss_X = torch.nn.MSELoss().to(cfg.device)
    Loss_X_res = torch.nn.MSELoss().to(cfg.device)
    logger.info('using Loss Function: MSE Loss')
    scaler = GradScaler()  # 训练前实例化一个GradScaler对象
    ####################dataset#########################################
    data_path = cfg.data_path
    data_name = cfg.data_name
    logger.info(
        "===================================Training START=====================================")
    loss_all_avg = 0.0
    loss_X_final_avg = 0.0
    loss_para_maps_final_avg = 0.0
    loss_X_res_final_avg = 0.0
    atb_error_avg = 0.0
    iter_i = 0
    oc_sign = 1
    Net.train()
    for e in range(cfg.iter_epoch):
        data_loader = train_dataloader(
            data_index_list, data_path, data_name, cfg.batch_size)
        for X, para_maps, max_X in data_loader:
            Optimizer.zero_grad()
            with autocast():
                # input
                with torch.no_grad():
                    atb = gen_atb(X)
                    atb_error = torch.mean(torch.pow(atb-X, 2))
                # forward
                X_recon, para_est, X_res_list = Net(atb)
                # reconstruction loss
                loss_X_recon = Loss_X(X_recon, X)
                # At(AX-b) data consistency loss
                loss_X_res_list = [Loss_X_res(X_res, atb)
                                   for X_res in X_res_list]
                loss_X_res_all = sum(loss_X_res_list)
                loss_X_res_final = loss_X_res_list[-1]
                # parameter maps estimation loss
                loss_para_maps_final = Loss_para_maps(para_est, para_maps)
                # total loss
                loss = loss_para_maps_final + loss_X_recon
            scaler.scale(loss).backward()
            scaler.step(Optimizer)
            scaler.update()  # 更新缩放器

            writer_train.add_scalar('loss', loss.item(), iter_i)
            for i in range(cfg.depth):
                writer_train.add_scalar(
                    'loss_X_res_{}'.format(i), loss_X_res_list[i].item(), iter_i)
            writer_train.add_scalar(
                'loss_para_maps_{}'.format(cfg.depth), loss_para_maps_final.item(), iter_i)
            writer_train.add_scalar(
                'X res all', loss_X_res_all.item(), global_step=iter_i)
            writer_train.add_scalar(
                'atb error', atb_error.item(), global_step=iter_i)
            loss_all_avg += loss.item()
            loss_X_res_final_avg += loss_X_res_final.item()
            loss_para_maps_final_avg += loss_para_maps_final.item()
            loss_X_final_avg += loss_X_recon.item()
            atb_error_avg += atb_error.item()

            if cfg.use_scheduler and iter_i % cfg.scheduler_update == 0:
                Scheduler_decoder.step()

            if iter_i % cfg.print_every == cfg.print_every-1:
                logger.info('Epoch: {}, iter: {}, Avg: loss: {:.6f}; Atb: {:.6f}; X: {:.6f}; Para: {:.6f}; Res: {:.6f}'.format(
                    e, iter_i, loss_all_avg / cfg.print_every, atb_error_avg /
                    cfg.print_every, loss_X_final_avg / cfg.print_every,
                    loss_para_maps_final_avg / cfg.print_every, loss_X_res_final_avg / cfg.print_every))
                writer_train.add_scalar(
                    'Avg-loss', loss_all_avg / cfg.print_every, global_step=iter_i)
                writer_train.add_scalar(
                    'Avg-X', loss_X_final_avg / cfg.print_every, global_step=iter_i)
                writer_train.add_scalar(
                    'Avg-X-res', loss_X_res_final_avg / cfg.print_every, global_step=iter_i)
                writer_train.add_scalar(
                    'Avg-para', loss_para_maps_final_avg / cfg.print_every, global_step=iter_i)
                writer_train.add_scalar(
                    'Avg-atb', atb_error_avg / cfg.print_every, global_step=iter_i)
                for i in range(cfg.depth):
                    writer_train.add_scalar('mu_{}'.format(
                        i), Net.dc_list[i].mu_parameter.item(), global_step=iter_i)

                # reset to zero
                loss_all_avg = 0.0
                loss_X_final_avg = 0.0
                loss_X_res_final_avg = 0.0
                loss_para_maps_final_avg = 0.0
                atb_error_avg = 0.0

            writer_train.add_scalar('learning_rate',
                                    Optimizer.param_groups[-1]['lr'],
                                    global_step=iter_i)
            iter_i += 1
        if e % cfg.save_every == cfg.save_every-1:
            torch.save(Net.state_dict(), Save_NET_root+'/Net_{}.pth'.format(e))
            logger.info('save model at: {}'.format(
                Save_NET_root+'/Net_{}.pth'.format(e)))
    torch.save(Net.state_dict(), Save_NET_root+'/Net_final.pth')
    logger.info('save model at: {}'.format(Save_NET_root+'/Net_final.pth'))
    writer_train.close()
    # 修改文件权限
    if cfg.train_in_docker:
        os.system('chown -R {} {}'.format(cfg.user_id, OUT_dir))
        logger.info(
            'All file permissions are given to users: {}'.format(cfg.user_id))
    logger.info(
        "===================================END======================================")


if __name__ == "__main__":
    data_index_list = list(range(1,381))
    train(data_index_list)

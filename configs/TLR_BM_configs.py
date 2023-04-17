"""
常量声明，设置需要的参数
"""
import torch
import time
now_time = time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))

# 关键网络参数
data_length = 500
depth = 4  # depth of network
rank = 5  # rank of CP tensor
batch_size = 1

# 模型加载
load_model = False  # 是否加载已存储的模型
model_saved_path = './data/output/Model_save/Net_final.pth'  # 模型存储路径
projection_op_saved_model = './output/Model_save/BM_final.pth'  # 模型存储路径
load_depth = 4  # 加载BM模型的深度

# 存储设置
data_path = './data/MRF_dataset/'  # 数据集路径
data_name = 'X'
OUT_dir = './data/output/'+'/TLR_BM_{}_{}_{}_{}/'.format(
    data_name, depth, rank, now_time)

# 序列参数范围，归一化以方便网路训练
FA_min = 10.0  # degree
FA_max = 70.0
TR_min = 12.0  # ms
TR_max = 15.0
TE = 3.0
# 参数范围 ms
T1_max = 5000.0
T2_max = 2500.0

# 默认数据类型，统一网络及代码中的数据类型，避免类型不匹配
dtype_float = torch.float32
dtype_complex = torch.complex64
cuda_device = '0'
device = torch.device("cuda:{}".format(cuda_device)
                      if torch.cuda.is_available() else "cpu")

# data
para_maps = 2  # 参数图数量 2=T1, T2; 3=T1, T2, PD
data_shape = (220, 220, 500)  # 数据形状[Kx, Ky, T]
dataset_len = 380  # 数据集长度
k_sample_points = 2880  # k采样点数

# 其他网络参数
input_dim = 2*data_shape[2]  # input channel: complex --> [real, imag]
output_dim = para_maps
csm_path = data_path + 'csm_maps/csm_maps.mat'  # coil sensitivity map path
ktraj_path = data_path + 'ktraj_nufft.mat'  # k-space trajectory path

# 训练参数
LearningRate = 3e-4  # decoder学习率
scheduler_update = 2000
use_scheduler = True  # 是否使用学习率调整器
iter_epoch = 50
print_every = 10    # 每隔多少个batch打印一次loss
save_every = 10     # 每隔多少个epoch保存一次模型
dc_loss_weight = 0.01  # dc loss weight

# 使用docker
train_in_docker = True
user_id = 1000  # 更改文件所有者为 user_id

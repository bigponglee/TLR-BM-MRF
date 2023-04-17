import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 数据类型
dtype_float = torch.float64  # float64下与matlab仿真结果一直，32会有精度损失
dtype_complex = torch.complex128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 参数范围
FA_min = 10.0
FA_max = 70.0
TR_min = 12.0
TR_max = 15.0
TE = 2.94
'''
序列Bloch仿真pytorch实现
epg_ir_fisp_signal：Bloch仿真，生成信号，序列FISP
'''

N_states = 20  # Number of states to simulate
phi = 90.  # degrees, this function assumes phi = 90 for all real states, but can be any number
# create a matrix that is size 3 x N_states, for shifting without wrap, put ones in off diagonals
_mask_F_plus = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_mask_F_plus[0, :] = 1.0
_mask_F_minus = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_mask_F_minus[1, :] = 1.
_mask_Z = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_mask_Z[2, :] = 1.
_F0_plus_mask = torch.zeros(3, N_states, dtype=dtype_complex).to(device)
_F0_plus_mask[0, 0] = 1.
_shift_right_mat = torch.roll(
    torch.eye(N_states, dtype=dtype_complex), 1, 1).to(device)
_shift_right_mat[:, 0] = 0
_shift_left_mat = torch.roll(
    torch.eye(N_states, dtype=dtype_complex), -1, 1).to(device)
_shift_left_mat[:, -1] = 0


def get_F_plus_states(FZ):
    return _mask_F_plus.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def get_F_minus_states(FZ):
    return _mask_F_minus.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def get_Z_states(FZ):
    # elementwise multiplication
    return _mask_Z.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def get_F0_plus(FZ):
    return _F0_plus_mask.view([1, 3, N_states]).expand(FZ.shape[0], 3, N_states) * FZ


def grad_FZ(FZ):
    # shift the states
    out1 = torch.matmul(get_F_plus_states(FZ), _shift_right_mat.view([1, N_states, N_states]).expand(FZ.shape[0], N_states, N_states)) + \
        torch.matmul(get_F_minus_states(FZ), _shift_left_mat.view(
            [1, N_states, N_states]).expand(FZ.shape[0], N_states, N_states)) + get_Z_states(FZ)
    # fill in F0+ using a mask
    out2 = out1 + get_F0_plus(torch.conj(torch.roll(out1, -1, 1)))
    return out2


def rf_epg(alpha, phi):
    '''
    Bloch模拟射频脉冲RF激活，旋转矩阵R计算
    输入:
    alpha：翻转角（弧度制）.
    phi = angle of rotation axis from Mz (radians).
    输出:
    R = RF rotation matrix (3x3).
    '''
    phi = torch.as_tensor(phi, dtype=dtype_float)
    R = torch.zeros(alpha.shape[0], 3, 3, dtype=dtype_complex).to(device)
    R[:, 0, 0] = torch.pow(torch.cos(alpha/2.0), 2)
    R[:, 0, 1] = torch.exp(2*1j*phi)*torch.pow(torch.sin(alpha/2.0), 2)
    R[:, 0, 2] = -1j*torch.exp(1j*phi)*torch.sin(alpha)
    R[:, 1, 0] = torch.exp(-2j*phi)*torch.pow(torch.sin(alpha/2.0), 2)
    R[:, 1, 1] = torch.pow(torch.cos(alpha/2.0), 2)
    R[:, 1, 2] = 1j*torch.exp(-1j*phi)*torch.sin(alpha)
    R[:, 2, 0] = -1j/2.0*torch.exp(-1j*phi)*torch.sin(alpha)
    R[:, 2, 1] = 1j/2.0*torch.exp(1j*phi)*torch.sin(alpha)
    R[:, 2, 2] = torch.cos(alpha)
    return R


def rf_FZ(FZ, alpha, phi):
    '''
    Bloch模拟初始反转脉冲激活
    输入:
    FZ：3xN vector of F+, F- and Z states.
    alpha：翻转角（度）.
    phi = angle of rotation axis from Mz (度).
    输出:
    Updated FpFmZ state.
    '''
    R = rf_epg(alpha*torch.pi/180., phi*torch.pi/180.)
    return torch.matmul(R, FZ)


def relax_epg(M0, T, T1, T2):
    '''
    Bloch模拟磁化弛豫
    输入：
    M0:稳态磁化强度/弛豫前一刻磁化强度
    T:弛豫时间
    T1:纵向弛豫
    T2:横向弛豫
    输出：
    A: spin relaxation
    B:input matrix
    '''
    E1 = torch.exp(-T/T1)
    E2 = torch.exp(-T/T2)
    # decay of states due to relaxation
    A = torch.zeros(T1.shape[0], 3, 3, dtype=dtype_complex).to(device)
    A[:, 0, 0] = E2
    A[:, 1, 1] = E2
    A[:, 2, 2] = E1
    B = torch.zeros(T1.shape[0], 3, N_states, dtype=dtype_complex).to(device)
    B[:, 2, 0] = M0*(1.0-E1)
    return A, B


def relax_FZ(FZ, M0, T, T1, T2):
    '''
    Bloch模拟初始反转脉冲后磁化弛豫
    输入：
    FZ：初始磁化矩阵
    M0:稳态磁化强度/弛豫前一刻磁化强度
    T:弛豫时间
    T1:纵向弛豫
    T2:横向弛豫
    输出：
    弛豫后磁化矩阵
    '''
    A, B = relax_epg(M0, T, T1, T2)
    return torch.matmul(A, FZ)+B


def init_ir_relax(M0, inversion_delay, T1, T2):
    '''
    180°翻转脉冲施加后弛豫
    输入：
    M0：稳态磁化强度（质子密度）
    inversion_delay: 180°翻转脉冲施加后回复时间
    T1:纵向弛豫时间
    T2:横向弛豫时间
    输出：
    下次施加脉冲前信号初始态
    '''
    m1 = torch.as_tensor([[0.], [0.], [M0]], dtype=dtype_complex).to(
        device).view([1, 3, 1]).expand(T1.shape[0], 3, 1)
    m2 = torch.zeros(T1.shape[0], 3, N_states-1,
                     dtype=dtype_complex).to(device)
    FZ_init = torch.cat((m1, m2), 2)
    init_FA = torch.as_tensor(
        np.ones(T1.shape[0]) * 180.0, dtype=dtype_float).to(device)
    FZ_flip = relax_FZ(rf_FZ(FZ_init, init_FA, phi),
                       M0, inversion_delay, T1, T2)
    return FZ_flip


def get_rf_epg(FZ, FA):
    '''
    模拟射频脉冲激活
    输入：
    FZ：3xN vector of F+, F- and Z states.
    FA：翻转角（°）
    输出：
    Updated FpFmZ state.
    '''
    R = rf_epg(FA*torch.pi/180.0, phi*torch.pi/180.0)
    return torch.matmul(R, FZ)


def get_relax_epg(FZ, M0, T, T1, T2):
    '''
    当前脉冲激发后产生的信号
    输入：
    M0：稳态磁化强度（质子密度）
    T: 脉冲施加后回复时间
    T1:纵向弛豫时间
    T2:横向弛豫时间
    输出：
    施加脉冲激发信号
    '''
    FZ_out = relax_FZ(FZ, M0, T, T1, T2)
    return FZ_out


def next_rf_relax(FZ, M0, T, T1, T2):
    '''
    下一次脉冲前，信号的初始状态
    输入：
    M0：稳态磁化强度（质子密度）
    T: 恢复时间
    T1:纵向弛豫时间
    T2:横向弛豫时间
    输出：
    下次脉冲激发前，信号
    '''
    FZ_spoiled = grad_FZ(FZ)
    return get_relax_epg(FZ_spoiled, M0, T, T1, T2)


def epg_ir_fisp_signal_batch(FAs_TRs, T1, T2):
    '''
    Bloch仿真，生成信号，序列IR_FISP
    输入：
    M0：稳态磁化强度（质子密度）
    FAs:翻转角 °度
    TEs: Echo Time  ms
    TRs: 脉冲序列重复周期 ms
    inversion_delay: 180°翻转脉冲施加后回复时间 ms
    T1:纵向弛豫时间 ms
    T2:横向弛豫时间 ms
    输出：
    signal：响应信号
    '''
    M0 = torch.as_tensor(1.0).to(device)
    TEs = torch.as_tensor(
        np.ones(FAs_TRs.shape[1]) * TE, dtype=dtype_float).to(device)
    LEN = FAs_TRs.shape[1]
    signal = torch.zeros(T1.shape[0], LEN, dtype=dtype_complex).to(device)
    W = torch.zeros(T1.shape[0], 3, N_states, dtype=dtype_complex).to(device)
    W[:, 2, 0] = -1.0
    for i in range(LEN):
        U = get_rf_epg(
            W, FAs_TRs[:, i, 0])
        V = get_relax_epg(U, M0, TEs[i], T1, T2)
        W = next_rf_relax(
            V, M0, FAs_TRs[:, i, 1]-TEs[i], T1, T2)
        signal[:, i] = V[:, 0, 0]
    return signal


def get_LUT_full():
    T1 = list(range(10, 2000, 10))+list(range(2000, 5001, 100))
    T2 = list(range(10, 100, 5))+list(range(100, 200, 10)) + \
        list(range(200, 2501, 100))

    T1.append(2569.0)
    T1.append(833.0)
    T1.append(752.0)
    T2.append(329.0)
    T2.append(83.0)
    T2.append(47.0)
    T2.append(237.0)
    T1.sort()
    T2.sort()
    T1 = np.float64(T1)
    T2 = np.float64(T2)
    LUT = np.zeros((len(T1)*len(T2), 2), dtype=np.float64)
    k = 0
    for tmp_T1 in T1:
        for tmp_T2 in T2:
            if tmp_T1 < tmp_T2:  # 组织的T1值大于于T2值 差距在5-10倍，此处去除T1小于T2的情况
                continue
            LUT[k, 0] = tmp_T1
            LUT[k, 1] = tmp_T2
            k = k+1
    # 去除没有曲线的全为零的多余项
    LUT = LUT[0:k, :]
    return torch.from_numpy(LUT).type(dtype_float).to(device)


def get_LUT_all():
    T1 = list(range(10, 1000, 5)) + list(range(1000, 2500, 10)) + \
        list(range(2500, 5001, 20))
    T2 = list(range(10, 100, 2))+list(range(100, 500, 5)) + \
        list(range(500, 1500, 10))+list(range(1500, 2500, 20))
    T1.append(2569.0)
    T1.append(833.0)
    T1.append(752.0)
    T2.append(329.0)
    T2.append(83.0)
    T2.append(47.0)
    T2.append(237.0)
    T1.sort()
    T2.sort()
    T1 = np.float64(T1)
    T2 = np.float64(T2)
    LUT = np.zeros((len(T1)*len(T2), 2), dtype=np.float64)
    k = 0
    for tmp_T1 in T1:
        for tmp_T2 in T2:
            if tmp_T1 < tmp_T2:  # 组织的T1值大于于T2值 差距在5-10倍，此处去除T1小于T2的情况
                continue
            LUT[k, 0] = tmp_T1
            LUT[k, 1] = tmp_T2
            k = k+1
    # 去除没有曲线的全为零的多余项
    LUT = LUT[0:k, :]
    return torch.from_numpy(LUT).type(dtype_float).to(device)


def get_LUT_original():
    T1 = np.float64(list(range(100, 2001, 20))+list(range(2300, 5001, 300)))
    T2 = np.float64(list(range(20, 101, 5))+list(range(110, 201, 10)) +
                    list(range(300, 1901, 200)))
    LUT = np.zeros((len(T1)*len(T2), 2), dtype=np.float64)
    k = 0
    for tmp_T1 in T1:
        for tmp_T2 in T2:
            if tmp_T1 < tmp_T2:  # 组织的T1值大于于T2值 差距在5-10倍，此处去除T1小于T2的情况
                continue
            LUT[k, 0] = tmp_T1
            LUT[k, 1] = tmp_T2
            k = k+1
    # 去除没有曲线的全为零的多余项
    LUT = LUT[0:k, :]
    return torch.from_numpy(LUT).type(dtype_float).to(device)


def get_batch_index(LEN, batch_size):
    if LEN <= batch_size:
        raise ValueError(
            'The LEN should be longer than batch_size.')
    num = LEN//batch_size
    yu = LEN % batch_size
    out = []
    for i in range(num):
        out.append([i*batch_size, (i+1)*batch_size])
    if yu != 0:
        out.append([num*batch_size, LEN])
    return out


def guiyi(in_array, min1=None, max1=None):
    '''
    矩阵归一化
    '''
    if min1 == None:
        min1 = np.min(in_array)
    in_array = in_array-min1
    if max1 == None:
        max1 = np.max(in_array)
    if max1 == 0.0:
        pass
    else:
        in_array = in_array/(max1-min1)
    return in_array


def build_Dictionary(FAs, TRs):
    '''
    Bloch仿真，生成字典，序列IR_FISP
    输入：
    FAs:翻转角序列 度°
    TE: Echo Time ms
    TRs: 脉冲序列重复周期序列 ms 
    其他说明：
    M0：稳态磁化强度（质子密度）（生成字典默时认为1）
    T1: 纵向弛豫时间 字典指定范围 ms
    T2: 横向弛豫时间 字典指定范围 ms
    输出：
    D：包含所有信号条目的字典 Entries*L
    LUT: 字典对应的参照表Entries*2
    '''
    # batch_size = 92011
    LEN = FAs.shape[0]
    # LUT = get_LUT_full()
    # LUT = get_LUT_full()
    LUT = get_LUT_original()
    # batch_index = get_batch_index(LUT.shape[0], batch_size)
    FA_init = torch.from_numpy(FAs.astype(np.float64).reshape([LEN, 1]))
    TR_init = torch.from_numpy(TRs.astype(np.float64).reshape([LEN, 1]))
    FAs_TRs = torch.cat([FA_init, TR_init], 1).type(dtype_float).to(device)
    FAs_TRs = FAs_TRs.view([1, LEN, 2])

    dictionary = epg_ir_fisp_signal_batch(
        FAs_TRs, LUT[:, 0], LUT[:, 1])
    return dictionary.detach().cpu().numpy(), LUT.detach().cpu().numpy()


def build_TemplateMatrix(FAs, TRs, T1_maps, T2_maps, M0_maps):
    '''
    Bloch仿真，生成时域矩阵X，序列IR_FISP
    输入：
    FAs:翻转角序列 度°
    TEs: Echo Time ms
    TRs: 脉冲序列重复周期序列 ms
    inversion_delay: 180°翻转脉冲施加后回复时间 ms
    M0_maps：稳态磁化强度（质子密度）（生成字典默时认为1）
    T1_maps: 纵向弛豫时间  ms
    T2_maps: 横向弛豫时间  ms
    输出：
    X：仿真生成的时域矩阵
    '''
    m = T1_maps.shape[0]
    n = T1_maps.shape[1]
    LEN = FAs.shape[0]
    LUT = np.zeros([m*n, 2])
    LUT[:, 0] = T1_maps.reshape([m*n, ])
    LUT[:, 1] = T2_maps.reshape([m*n, ])
    LUT = torch.from_numpy(LUT).type(dtype_float).to(device)
    M0_maps = torch.from_numpy(M0_maps.reshape(
        [m*n, 1])).type(dtype_float).to(device)
    FA_init = torch.from_numpy(FAs.astype(np.float64).reshape([LEN, 1]))
    TR_init = torch.from_numpy(TRs.astype(np.float64).reshape([LEN, 1]))
    FAs_TRs = torch.cat([FA_init, TR_init], 1).type(dtype_float).to(device)
    FAs_TRs = FAs_TRs.view([1, LEN, 2])
    X = epg_ir_fisp_signal_batch(
        FAs_TRs, LUT[:, 0], LUT[:, 1])
    X = X*M0_maps.expand(m*n, LEN)
    return X.detach().cpu().numpy().reshape([m, n, LEN])


def build_signal(FAs, TRs):
    '''
    Bloch仿真，生成字典，序列IR_FISP
    输入：
    FAs:翻转角序列 度°
    TE: Echo Time ms
    TRs: 脉冲序列重复周期序列 ms 
    其他说明：
    M0：稳态磁化强度（质子密度）（生成字典默时认为1）
    T1: 纵向弛豫时间 字典指定范围 ms
    T2: 横向弛豫时间 字典指定范围 ms
    输出：
    D：包含所有信号条目的字典 Entries*L
    LUT: 字典对应的参照表Entries*2
    '''
    # batch_size = 92011
    LEN = FAs.shape[0]
    LUT = get_LUT_full()
    # LUT = get_LUT_all()
    LUT = np.zeros((7, 2), dtype=np.float64)
    LUT[0, :] = [2569, 329]  # CSF
    LUT[1, :] = [833, 83]  # GREY MATTER
    LUT[2, :] = [500, 70]  # WHITE MATTER
    LUT[3, :] = [350, 70]  # GREY MATTER
    LUT[4, :] = [900, 47]  # MUSCLE SKIN
    LUT[5, :] = [2569, 329]  # SKIN
    LUT[6, :] = [752, 237]  # MS LESION
    PD = np.array([1.0, 0.86, 0.77, 1.0, 1.0, 1.0, 0.76])
    LUT = torch.from_numpy(LUT).type(dtype_float).to(device)
    M0_maps = torch.from_numpy(PD.reshape(
        [7, 1])).type(dtype_float).to(device)
    FA_init = guiyi(FAs, FA_min, FA_max)
    TR_init = guiyi(TRs, TR_min, TR_max)
    FA_init = torch.from_numpy(FA_init.astype(np.float64).reshape([LEN, 1]))
    TR_init = torch.from_numpy(TR_init.astype(np.float64).reshape([LEN, 1]))
    FAs_TRs = torch.cat([FA_init, TR_init], 1).type(dtype_float).to(device)
    FAs_TRs = FAs_TRs.view([1, LEN, 2])
    dictionary = epg_ir_fisp_signal_batch(
        FAs_TRs, LUT[:, 0], LUT[:, 1])
    return dictionary.detach().cpu().numpy()


def build_dictionary_mat(FAs, TRs):
    '''
    Bloch仿真，生成字典，序列IR_FISP
    输入：
    FAs:翻转角序列 度° [L, ]
    TRs: 脉冲序列重复周期序列 ms  [L, ]
    输出：
    D：包含所有信号条目的字典 Entries*L
    LUT: 字典对应的参照表Entries*2
    '''
    # FAs = np.array(FAs)
    # TRs = np.array(TRs)
    Dictionary, LUT = build_Dictionary(FAs, TRs)
    # Dictionary = array.array('d', Dictionary)
    # LUT = array.array('d', LUT)
    return {'dic': Dictionary, 'lut': LUT}


def build_TemplateMatrix_mat(FAs, TRs, T1_maps, T2_maps, M0_maps):
    '''
    Bloch仿真，生成时域矩阵X，序列IR_FISP
    输入：
    FAs:翻转角序列 度° [L, ]
    TRs: 脉冲序列重复周期序列 ms[L, ]
    M0_maps：稳态磁化强度（质子密度）（生成字典默时认为1） [Nx,Ny]
    T1_maps: 纵向弛豫时间  ms [Nx,Ny]
    T2_maps: 横向弛豫时间  ms [Nx,Ny]
    输出：
    X：仿真生成的时域矩阵 [Nx,Ny, L]
    '''
    # FAs = np.array(FAs)
    # TRs = np.array(TRs)
    # T1_maps = np.array(T1_maps)
    # T2_maps = np.array(T2_maps)
    # M0_maps = np.array(M0_maps)
    X = build_TemplateMatrix(FAs, TRs, T1_maps, T2_maps, M0_maps)
    # X = array.array('d', X)
    return X

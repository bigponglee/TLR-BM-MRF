import logging
from matplotlib import pyplot as plt
import numpy as np
import math


def def_logger(path):
    """

    Args:
        path: logger存储路径

    Returns:

    """
    LOGGING_LEVEL = logging.DEBUG
    logger = logging.getLogger(__name__)
    logger.setLevel(level=LOGGING_LEVEL)

    handler = logging.FileHandler(path)
    handler.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s--->>>%(message)s', "%Y/%m/%d--%H:%M:%S")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(LOGGING_LEVEL)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def show_map(para_map, para_map_est):
    T1_range = [0, 2500]
    T2_range = [0, 500]
    T1_error_range = [0, 300]
    T2_error_range = [0, 200]
    show_aspect = 1.0
    plt.figure()
    ax = plt.subplot(161)
    a = plt.pcolor(para_map[:,:,0], cmap='hot', vmin=T1_range[0], vmax=T1_range[1])
    ax.set_aspect(show_aspect)
    ax.set_title('T1 GT')
    plt.axis('off')
    bx = plt.subplot(162)
    b = plt.pcolor(para_map_est[:,:,0], cmap='hot', vmin=T1_range[0], vmax=T1_range[1])
    plt.axis('off')
    bx.set_aspect(show_aspect)
    bx.set_title('T1 Est')
    abx = plt.subplot(163)
    ab = plt.pcolor(np.abs(para_map[:,:,0]-para_map_est[:,:,0]), cmap='hot', vmin=T1_error_range[0], vmax=T1_error_range[1])
    plt.axis('off')
    abx.set_aspect(show_aspect)
    abx.set_title('T1 Error')

    cx = plt.subplot(164)
    c = plt.pcolor(para_map[:,:,1], cmap='hot', vmin=T2_range[0], vmax=T2_range[1])
    cx.set_aspect(show_aspect)
    cx.set_title('T2 GT')
    plt.axis('off')
    dx = plt.subplot(165)
    d = plt.pcolor(para_map_est[:,:,1], cmap='hot', vmin=T2_range[0], vmax=T2_range[1])
    plt.axis('off')
    dx.set_aspect(show_aspect)
    dx.set_title('T2 Est')
    cdx = plt.subplot(166)
    cd = plt.pcolor(np.abs(para_map[:,:,1]-para_map_est[:,:,1]), cmap='hot', vmin=T2_error_range[0], vmax=T2_error_range[1])
    plt.axis('off')
    cdx.set_aspect(show_aspect)
    cdx.set_title('T2 Error')
    plt.show()
 
 
def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    scale = np.max(np.abs(ref))

    target_data = target/scale
    ref_data = ref/scale
    mse = np.mean( np.abs(target_data - ref_data) ** 2 )
    psnr = 10 * math.log10(1.0/mse)

    return psnr
import torch
import numpy as np
import cv2
import math

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def PSNR(img1, img2):
    """
    计算PSNR
    :param img1/img2: 原图和恢复图
    :return: PSNR值
    """
    mse = np.mean((img1/1. - img2/1.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 255.
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))




def save_img(filepath, img):
    # cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(filepath, img)

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

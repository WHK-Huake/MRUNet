
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
import cv2
import time

from data_RGB import get_test_data
from models.MRUNet import MRUNet
from utils.dir_utils import imread_CS_py

parser = argparse.ArgumentParser(description='MRUNet')

parser.add_argument('--input_dir', default='G:/low-light_image_enhancement/Our/dataset/test_datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/unpaired/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/LOL.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MRUNet()
utils.load_checkpoint(model_restoration,args.weights)

print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = ['DICM', 'VV', 'LIME', 'NPE', 'Fusion', 'MEF']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    result_dir  = os.path.join(args.result_dir, dataset)
    utils.mkdirs(result_dir)
    psnr = []
    times_list = []

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_ = data_test[0]
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(input_, 32)
            filenames = data_test[1]

            start_time = time.time()
            restored = model_restoration(Ipad.cuda())
            end_time = time.time()
            t = end_time - start_time

            stage4 = restored[0].permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(0)
            restored4 = stage4[:row, :col, :]
            times_list.append(t)

            cv2.imwrite((os.path.join(result_dir, filenames[0] + '_4.png')), stage4)

        times_ave = np.mean(times_list)
        print("Ave_time: %.4f" % ( times_ave))
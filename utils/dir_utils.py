import os
from natsort import natsorted
from glob import glob
import numpy as np
import torch

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_last_path(path, session):

	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]

	return x


def imread_CS_py(Iorg, block_size):
    # block_size = args.block_size
    [batch, channel, row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = torch.cat((Iorg, torch.zeros([batch, channel, row, col_pad])), dim=3)
    Ipad = torch.cat((Ipad, torch.zeros([batch, channel, row_pad, col + col_pad])), dim=2)
    [batch, channel, row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]
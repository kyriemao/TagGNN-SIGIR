import pynvml
from IPython import embed
import torch
import torch.nn as nn
import os
from os.path import join as oj
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import pickle


'''
automatically decide which gpu to use 
'''
def auto_gpu_setting(candidates = [0,1,2,3]):
    device = torch.device("cpu")
    if not torch.cuda.is_available():
        return device
    pynvml.nvmlInit()
    min_used = 9e15
    res_gpu = -1

    for gpu in candidates:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if min_used > meminfo.used:
            min_used = meminfo.used
            res_gpu = gpu

    if res_gpu != -1:
        device = torch.device("cuda:{}".format(res_gpu))
    
    return device


'''
time stamp
'''
def now():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%Y.%m.%d-%H:%M:%S')


def pstore(x, loc):
    with open(loc, 'wb') as f:
        pickle.dump(x, f)
    print('store {} ok!'.format(loc))

def pload(loc):
    with open(loc, 'rb') as f:
        x = pickle.load(f)
    print('load {} ok!'.format(loc))
    return x
import sys
sys.path.append('..')

import numpy as np
import os

data_dir = '/home/wuiron/Joint/data/cifar'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar10():
    for k in range(1,6):
        train = unpickle(data_dir+'/data_batch_{}'.format(k))
        train_raw = train[b'data']
        image = train_raw.reshape(10000,3,32,32)
        if k==1:
            trX = image
        else:
            trX = np.vstack([trX, image])

    trX.reshape([-1, 3, 32, 32])
    trX = trX.transpose(0,2,3,1)

    return trX

def cifar100():
    train = unpickle(data_dir+'/train')
    train_raw = train[b'data']
    trX = train_raw.reshape(-1,3,32,32)
    trX = trX.transpose(0,2,3,1)
    
    return trX


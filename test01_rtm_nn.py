#!/usr/bin/env python
# coding: utf-8

import os
import time
import itertools
import datetime
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

import netCDF4
import numpy as np
from pandas import DataFrame

from sklearn.model_selection import train_test_split
import input_test01_rtm_nn as input_params



from ds_rtm_nn import RTM
from ds_rtm_nn import MLP
from ds_rtm_nn import msre
from ds_rtm_nn import test_plot300
from ds_rtm_nn import load_LUT
from ds_rtm_nn import features_maker
from ds_rtm_nn import XY_data_loader
from ds_rtm_nn import search_LUT_files


if __name__ == '__main__':
    print('start')
    LUT_file = input_params.LUT_file
    lat = LUT_file[17]
    toz = LUT_file[18:21]
    lr = input_params.lr

    data = netCDF4.Dataset(LUT_file, mode='r')

    (sza, vza, raa, alb, pre, wav, ps_in, zs_in, ts_in, os_in, taudp_in, nl_in,
        rad, albwf, o3wf) = load_LUT(LUT_file)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[]})

    pre_list = []
    alb_list = []
    raa_list = []
    vza_list = []
    sza_list = []
    wav_list = []
    wav_l = list(wav)
    #wav_num = len(wav)
    wav300 = wav[660:]
    wav_num = len(wav300)

    (pre_list, alb_list, raa_list, vza_list, sza_list) = features_maker(pre, alb, raa, vza, sza)    

    train_loader, valid_loader, test_loader = XY_data_loader(pre_list, alb_list,
            raa_list, vza_list, sza_list, rad, albwf, o3wf)

    pre_list = None
    alb_list = None
    raa_list = None
    vza_list = None
    sza_list = None
    rad = None
    albwf = None
    o3wf = None


# neural network model

    model = MLP()
    print(model)

#import weight_init
#weight_init.weight_init(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss()
#loss_fn = msre()


    mean_train_losses = []
    mean_valid_losses = []
    mean_test_losses = []
    valid_acc_list = []

    epochs = 5

    _epoch_list = []

    statefiles = []
    for (path, dir, files) in os.walk('./states_01/'):
        
        for filename in files:
            basename = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[-1]
            if basename[0:14] == '01_rtm_nn_' + lat + toz and ext == '.pth':
                _epoch = filename[21:26]
                print(_epoch)
                _epoch_list.append(_epoch)
                statefiles.append(filename)

    if len(statefiles) >= 1:
        torchstatefile = './states_01/' + sorted(statefiles)[-1]
        model.load_state_dict(torch.load(torchstatefile))
        real_epoch = int(sorted(_epoch_list)[-1]) + 1
        print('real_epoch = ', real_epoch)
    else:
        real_epoch = 0

    for epoch in range(epochs):

# training
        model.train()
        
        train_losses = []
        valid_losses = []
        test_losses = []
        timestamp = time.time()
        for i, (features, radiances) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = msre(outputs, radiances)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            #print(time.time() - timestamp)
            if (i * 128) % (128 * 10) == 0:
                print(f'{i * 128} / ', len(train_loader)*128, time.time() - timestamp,
                        datetime.datetime.now())
                print(loss.item())
                timestamp = time.time()
                
# validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (features, radiances) in enumerate(valid_loader):
                outputs = model(features)
                loss = msre(outputs, radiances)
                valid_losses.append(loss.item())
                if (i * 128) % (128 * 10) == 0:
                    print(f'{i * 128} / ', len(valid_loader)*128, time.time() - timestamp,
                            datetime.datetime.now(), 'valid')
                    print(loss.item())
                    timestamp = time.time()
                #_, predicted = torch.max(outputs.data, 1)
                #correct += (predicted == radiances).sum().item()
                #total += radiances.size(0)
                
        print('asdjfaijdfaidsf')
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))

# test
# plot radiances
        with torch.no_grad():
            for i, (features, radiances) in enumerate(test_loader):
                outputs = model(features)
                loss = msre(outputs, radiances)
                test_losses.append(loss.item())
                if (i * 128) % (128 * 10) == 0:
                    print(f'{i * 128} / ', len(test_loader)*128, time.time() - timestamp,
                            datetime.datetime.now(), 'test')
                    print(loss.item())
                    timestamp = time.time()

        batch_idx = 0
        filename = ('./plot/01_rtm_nn_' + lat + toz + '_epoch_' + str(epoch).zfill(5) +
            '_index_' + str(batch_idx).zfill(5))
        test_plot300(real_epoch, batch_idx, features, wav300, radiances,
                outputs, filename, lr)

        print('test plotting done')
        lossesfile = './result/01_rtm_nn_' + lat + toz + '_mean_losses.txt'
        if os.path.exists(lossesfile):
            with open(lossesfile, 'a') as f:
                f.write(str(real_epoch).zfill(5) + ',' + 
                        str(np.mean(train_losses)) + ',' + 
                        str(np.mean(valid_losses)) + ',' +
                        str(np.mean(test_losses)) + '\n')
        else:
            with open(lossesfile, 'w') as f:
                f.write('index,mean_train_losses,mean_valid_losses,mean_test_losses' + '\n')
                f.write(str(real_epoch).zfill(5) + ',' + 
                        str(np.mean(train_losses)) + ',' + 
                        str(np.mean(valid_losses)) + ',' + 
                        str(np.mean(test_losses)) + '\n')

        real_epoch = real_epoch + 1
        print('whole process done')


    for batch_idx, (features, radiances) in enumerate(test_loader):
        filename = ('./plot/01_rtm_nn_' + lat + toz + '_epoch_' + str(epoch).zfill(5) +
            '_index_' + str(batch_idx).zfill(5))
        test_plot300(real_epoch, batch_idx, features, wav300, radiances, outputs,
                filename, lr)
# save state
    torchstatefile = './states_01/01_rtm_nn_' + lat + toz + '_epoch_' + str(real_epoch).zfill(5) + '.pth'
    torch.save(model.state_dict(), torchstatefile)
    print('torch save')

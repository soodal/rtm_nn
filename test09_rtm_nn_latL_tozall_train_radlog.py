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
#import input_test01_rtm_nn as input_params

from ds_rtm_nn import RTM
from ds_rtm_nn import MLPv01_6_800
from ds_rtm_nn import msre
from ds_rtm_nn import test_plot300
from ds_rtm_nn import load_LUT
from ds_rtm_nn import features_maker_toz
from ds_rtm_nn import XY_data_loader_toz_800_train_radlog
from ds_rtm_nn import search_lat_LUT_files
#from ds_rtm_nn import rad_custom_normalize
from ds_rtm_nn import input_custom_normalize
from ds_rtm_nn import wav_custom_normalize


if __name__ == '__main__':
# set up the neural network

    LUT_filelist = search_lat_LUT_files('L')

    model = MLPv01_6_800()
    print(model)
    lr = 0.0000001 # changed from 0.0001 after 223 epoch
# changed from 0.00001 after 526 epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()

    mean_train_losses = []
    mean_valid_losses = []
    mean_test_losses = []
    valid_acc_list = []

    epochs = 1000

    _epoch_list = []
    for (path, dir, files) in os.walk('./states_09/'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                _epoch = filename[-12:-4]
                _epoch_list.append(_epoch)

    if len(files) >= 1:
        torchstatefile = './states_09/' + sorted(files)[-1]
        real_epoch = int(sorted(_epoch_list)[-1]) + 1
        print('real_epoch = ', real_epoch)
        if real_epoch == 621:
            print(torchstatefile)
            model.load_state_dict(torch.load(torchstatefile))
        else:
            checkpoint = torch.load(torchstatefile)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_local = checkpoint['epoch']
            loss = checkpoint['loss']
            if real_epoch -1 !=  epoch_local:
                print('epoch_local is different with real_epoch')
                quit()

        print(_epoch_list)
        print(sorted(_epoch_list)[-1])
    else:
        real_epoch = 0

#for epoch in range(epochs):
    epoch = real_epoch
    print('real_epoch is ', real_epoch)
    epoch_local = real_epoch
    model.train()
    train_losses = np.array([])
    valid_losses = np.array([])
    timestamp = time.time()

    #with open('./LUT/read_lut_check.txt', 'r') as f:
        #read_lut_list = f.read().splitlines()

    #train_losses = np.load('./losses/test09_train_losses.npy')
    #valid_losses = np.load('./losses/test09_valid_losses.npy')

    #read_count = 0

    for LUT_file in LUT_filelist:
            print('real_epoch check' , real_epoch)
            lat = LUT_file[17]
            toz = list(np.array([int(LUT_file[18:21])]))
            print(toz)
            
        #if read_count == 4:
            #break

        #if not LUT_file in read_lut_list:
            (sza, vza, raa, alb, pre, wav, ps_in, zs_in, ts_in, o3_in, taudp_in, nl_in,
                rad, albwf, o3wf) = load_LUT(LUT_file)
            timestamp = time.time()

            wav_l = list(wav)
            #wav_num = len(wav)
            wav300 = wav[660:]
            wav_num = len(wav300)
            wav_list = list(wav300)

            (pre_list, alb_list, raa_list, vza_list, sza_list, toz_list) = features_maker_toz(
                    pre, alb, raa, vza, sza, toz)    


            train_loader = XY_data_loader_toz_800_train_radlog(pre_list, alb_list,
                    raa_list, vza_list, sza_list, toz_list, rad, albwf, o3wf)

            pre_list = None
            alb_list = None
            raa_list = None
            vza_list = None
            sza_list = None
            toz_list = None
            rad = None
            albwf = None
            o3wf = None


            for i, (features, radiances) in enumerate(train_loader):
                #if real_epoch == 0:
                optimizer.zero_grad()
                outputs = model(features)
                loss = msre(outputs, radiances)
                loss.backward()
                optimizer.step()
                #train_losses.append(loss.item())
                train_losses = np.append(train_losses, loss.item())
                #train_losses.append(loss.data)
                if (i * 128) % (128 * 10) == 0:
                    print(f'{i * 128} / ', len(train_loader)*128, time.time() - timestamp,
                            datetime.datetime.now())
                    print(loss.item())
                    timestamp = time.time()
            if epoch_local % 100 == 0:
                filename = ('./plot/09_rtm_nn_' + lat + '_alltoz_epoch_' + str(epoch).zfill(8) +
                    '_index_' + str(i).zfill(8))
                test_plot300(epoch_local, i, features, wav300, radiances,
                        outputs, filename, lr)



            print('model.eval()')
            model.eval()
            correct = 0
            total = 0

            #with torch.no_grad():
                #for i, (features, radiances) in enumerate(valid_loader):
                    #outputs = model(features)
                    #loss = msre(outputs, radiances)
                    
                    ##valid_losses.append(loss.item())
                    #valid_losses = np.append(valid_losses, loss.item())
                    ##valid_losses.append(loss.data)
                   
                    ##_, predicted = torch.max(outputs.data, 1)
                    ##correct += (predicted == radiances).sum().item()
                    ##total += radiances.size(0)
                    #outputs = None
                    #loss = None
                    #del outputs, loss

            #with torch.no_grad():
                #for i, (features, radiances) in enumerate(test_loader):
                    #outputs = model(features)
                    #loss = msre(outputs, radiances)
                    
                    ##valid_losses.append(loss.item())
                    #test_losses = np.append(test_losses, loss.item())
                    ##valid_losses.append(loss.data)
                   
                    ##_, predicted = torch.max(outputs.data, 1)
                    ##correct += (predicted == radiances).sum().item()
                    ##total += radiances.size(0)
                    #outputs = None
                    #loss = None
                    #del outputs, loss


            #with open('./LUT/read_lut_list', 'a') as f:
                #f.write(LUT_file + '\n')

            #np.save('./train_losses/test09_train_losses.npy',
                    #np.array(train_losses), allow_pickle=True)
            #np.save('./valid_losses/test09_valid_losses.npy',
                    #np.array(valid_losses), allow_pickle=True)

            #read_count += 1

    #if read_lut_list == LUT_filelist:
    mean_train_losses.append(np.mean(train_losses))
    #mean_valid_losses.append(np.mean(valid_losses))
    #mean_test_losses.append(np.mean(test_losses))
    #print('len mean train losses ', len(mean_train_losses))

#test_loader
    #for batch_idx, (f_plot, r_plot) in enumerate(test_loader):
    #for batch_idx, (f_plot, r_plot) in enumerate(valid_loader):
        #outputs = model(f_plot)
        #loss = msre(outputs, radiances)
        ##if i % 10 == 0:
            ##print(i, outputs)
        #filename = ('./plot/09_rtm_nn_' + lat + '_alltoz_epoch_' + str(epoch).zfill(8) +
            #'_index_' + str(batch_idx).zfill(8))
        #if batch_idx == 0:
            #test_plot300(real_epoch, batch_idx, f_plot, wav300, r_plot,
                    #outputs, filename, lr)
        #if real_epoch % 5 == 0:
            #print(batch_idx)
            #test_plot300(real_epoch, batch_idx, f_plot, wav300, r_plot,
                    #outputs, filename, lr)

        #outputs = None
        #del outputs

    lossesfile = './result/09_rtm_nn_latL_tozall_mean_losses.txt' 
    print('lossesfile')
    if os.path.exists(lossesfile):
        with open(lossesfile, 'a') as f:
            f.write(str(real_epoch).zfill(8) + ',' + str(np.mean(train_losses))
                    + ',' + '\n')
    else:
        with open(lossesfile, 'w') as f:
            f.write('index,mean_train_losses,mean_valid_losses'+'\n')
            f.write(str(real_epoch).zfill(8) + ',' + str(np.mean(train_losses))
                    + ',' + '\n')


    #real_epoch = real_epoch + 1

    torchstatefile = './states_09/09_rtm_nn_latL_tozall_epoch_' + str(epoch).zfill(8) + '.pth'
    #if real_epoch% 100 == 0:
    torch.save({'epoch':real_epoch, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict':optimizer.state_dict(),
        'loss': np.mean(train_losses)}, torchstatefile)
    print('torch save, done')

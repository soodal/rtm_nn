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
#from ds_rtm_nn import MLPv02_6_800
class MLPv02_6_800(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(6, 200)
        linear2 = nn.Linear(200, 300)
        linear3 = nn.Linear(300, 400)
        linear4 = nn.Linear(400, 500)
        linear5 = nn.Linear(500, 600)
        linear6 = nn.Linear(600, 700)
        linear7 = nn.Linear(700, 800)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.0)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv02_6_800, self).__init__()
        self.layers = nn.Sequential(
            linear1,
            leakyrelu,
            dropout,
            linear2,
            leakyrelu,
            dropout,
            linear3,
            leakyrelu,
            dropout,
            linear4,
            leakyrelu,
            dropout,
            linear5,
            leakyrelu,
            dropout,
            linear6,
            leakyrelu,
            dropout,
            linear7
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
from ds_rtm_nn import msre
from ds_rtm_nn import test_plot300_logradtorad
from ds_rtm_nn import load_LUT
from ds_rtm_nn import features_maker_toz
from ds_rtm_nn import XY_data_loader_toz_800_v2_sza_train_radlog
from ds_rtm_nn import XY_data_loader_toz_800_v2_sza_test_radlog
from ds_rtm_nn import search_lat_LUT_files
#from ds_rtm_nn import rad_custom_normalize
from ds_rtm_nn import input_custom_normalize
from ds_rtm_nn import wav_custom_normalize


if __name__ == '__main__':
# set up the neural network

    projectname = '10'
    
    states_path = './states/project_' + projectname
    plot_path = './plot/project_' + projectname
    if not os.path.isdir(states_path):
        os.makedirs(states_path)
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    LUT_file = '/RSL2/soodal/1_DATA/GEMSTOOL/lutdata/LUTNC/LUT_sca02stL300_sza_test.nc'


    model = MLPv02_6_800()
    print(model)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()

    mean_train_losses = []
    mean_valid_losses = []
    mean_test_losses = []
    valid_acc_list = []

    epochs = 1000

    _epoch_list = []
    for (path, dir, files) in os.walk('./states/project_' + projectname + '/'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                _epoch = filename[-12:-4]
                _epoch_list.append(_epoch)


    if len(files) >= 1:
        torchstatefile = './states/project_' + projectname + '/' + sorted(files)[-1]
        real_epoch = int(sorted(_epoch_list)[-1]) + 1
        print(real_epoch)
        #print('real_epoch = ', real_epoch)
        checkpoint = torch.load(torchstatefile)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_local = checkpoint['epoch']
        loss = checkpoint['loss']
        if real_epoch -1 !=  epoch_local:
            quit()

    else:
        real_epoch = 0

    for epoch in range(epochs):
        epoch_local = real_epoch + epoch
        print('real_epoch is ', epoch_local)
        model.train()
        train_losses = np.array([])
        test_losses = np.array([])
        valid_losses = np.array([])
        timestamp = time.time()

        #with open('./LUT/read_lut_check.txt', 'r') as f:
            #read_lut_list = f.read().splitlines()

        #train_losses = np.load('./losses/test10_train_losses.npy')
        #valid_losses = np.load('./losses/test10_valid_losses.npy')

        #read_count = 0

        #for LUT_file in LUT_filelist:

        print('real_epoch check' , epoch_local)
        #print(LUT_file)
        lat = LUT_file[-16]
        #print(lat)
        toz = list(np.array([int(LUT_file[-15:-12])]))

        #print(toz)
        
    #if read_count == 4:
        #break

    #if not LUT_file in read_lut_list:
        (sza, vza, raa, alb, pre, wav, ps_in, zs_in, ts_in, o3_in, taudp_in, nl_in,
            rad, albwf, o3wf) = load_LUT(LUT_file)
        timestamp = time.time()

        wav_l = list(wav)
        wav300 = wav[660:]
        wav_num = len(wav300)
        wav_list = list(wav300)

        sza_train_idx = np.array([0, 16, 31, 44, 55, 64, 71, 77, 82, 86, 89,
            91], dtype=np.int)
        sza_test_idx = np.array([1,2,3,4,5,6,7,8,9,
                                10,11,12,13,14,15,17,18,19,
                                20,21,22,23,24,25,26,27,28,29,
                                30,32,33,34,35,36,37,38,39,
                                40,41,42,43,45,46,47,48,49,
                                50,51,52,53,54,56,57,58,59,
                                60,61,62,63,65,66,67,68,69,
                                70,72,73,74,75,76,78,79,
                                80,81,83,84,85,87,88,
                                90,92], dtype=np.int)
        
        (pre_list_train, alb_list_train, raa_list_train, vza_list_train,
                sza_list_train, toz_list_train) = features_maker_toz(
                pre, alb, raa, vza, sza[sza_train_idx], toz)    

        (pre_list_test, alb_list_test, raa_list_test, vza_list_test,
                sza_list_test, toz_list_test) = features_maker_toz(
                pre, alb, raa, vza, sza[sza_test_idx], toz)    
    
        train_loader = XY_data_loader_toz_800_v2_sza_train_radlog(pre_list_train,
                alb_list_train, raa_list_train, vza_list_train, sza_list_train,
                toz_list_train, 
                rad[0][0][0][0][sza_train_idx][:], 
                albwf[0][0][0][0][sza_train_idx][:], 
                o3wf[0][0][0][0][sza_train_idx][:])

        test_loader = XY_data_loader_toz_800_v2_sza_test_radlog(pre_list_test,
                alb_list_test, raa_list_test, vza_list_test, sza_list_test, 
                toz_list_test, 
                rad[0][0][0][0][sza_test_idx][:], 
                albwf[0][0][0][0][sza_test_idx][:], 
                o3wf[0][0][0][0][sza_test_idx][:])


        pre_list = None
        alb_list = None
        raa_list = None
        vza_list = None
        sza_list = None
        toz_list = None
        rad = None
        albwf = None
        o3wf = None

#train_loader
        for i, (features, radiances) in enumerate(train_loader):
            print(i, features.shape, radiances.shape)
            #if real_epoch == 0:
            optimizer.zero_grad()
            outputs = model(features)
            loss = msre(outputs, radiances)
            loss.backward()
            optimizer.step()
            #train_losses.append(loss.item())
            train_losses = np.append(train_losses, loss.item())
            #train_losses.append(loss.data)
            #if real_epoch == 0:

            # each batch is 128, print this for one of 10 batches
            if (i * 128) % (128 * 10) == 0: 
                print(f'{i * 128} / ', len(train_loader)*128, time.time() - timestamp,
                        datetime.datetime.now())
                print(loss.item())
                timestamp = time.time()

            if epoch_local % 100 == 0:
                for inbatch in range(12):
                    #print(i, outputs)
                    filename = ('./plot/project_' + projectname + 
                            '/' + projectname + 
                            '_rtm_nn_nl24_' + lat + 
                            '_toz300_epoch_' + str(epoch_local).zfill(8) + 
                            '_train_index_' + str(i).zfill(8) + 
                            '_inbatch_' + str(inbatch).zfill(8))
                    # every each epoch, plot for first
                    test_plot300_logradtorad(epoch_local, inbatch, features, wav300, 
                            radiances, outputs, filename, lr)
                    

            outputs = None
            #loss = None
            #del outputs, loss
                
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

#train_loader
        with torch.no_grad():
            for i, (features, radiances) in enumerate(test_loader):
                outputs = model(features)
                loss = msre(outputs, radiances)
                
                #valid_losses.append(loss.item())
                test_losses = np.append(test_losses, loss.item())
                #valid_losses.append(loss.data)
               
                #_, predicted = torch.max(outputs.data, 1)
                #correct += (predicted == radiances).sum().item()
                #total += radiances.size(0)
                if epoch_local % 500 == 0:
                    for inbatch in range(features.detach().numpy().shape[0]):
                        filename = ('./plot/project_' + projectname + 
                                '/' + projectname + 
                                'rtm_nn_nl24_' + lat + 
                                '_toz300_epoch_' + str(epoch_local).zfill(8) +
                                '_test_index_' + str(i).zfill(8) + 
                                '_inbatch_' + str(inbatch).zfill(8))
                        test_plot300_logradtorad(epoch_local, inbatch, features, wav300, radiances,
                                outputs, filename, lr)


        #with open('./LUT/read_lut_list', 'a') as f:
            #f.write(LUT_file + '\n')

        #np.save('./train_losses/test10_train_losses.npy',
                #np.array(train_losses), allow_pickle=True)
        #np.save('./valid_losses/test10_valid_losses.npy',
                #np.array(valid_losses), allow_pickle=True)

        #read_count += 1

        #if read_lut_list == LUT_filelist:
        mean_train_losses.append(np.mean(train_losses))
        #mean_valid_losses.append(np.mean(valid_losses))
        mean_test_losses.append(np.mean(test_losses))
        #print('len mean train losses ', len(mean_train_losses))

#valid_loader
        #for batch_idx, (f_plot, r_plot) in enumerate(test_loader):
        ##for batch_idx, (f_plot, r_plot) in enumerate(valid_loader):
            ##print(batch_idx)
            ##print(f_plot.detach().numpy().shape[0])
            #for inbatch in range(f_plot.detach().numpy().shape[0]):
                #idx = (batch_idx * 128) + inbatch
                #print(inbatch)
                #print(r_plot.shape,f_plot.shape)
                #outputs = model(f_plot)
                #loss = msre(outputs, radiances)
                ##print(f_plot.shape)
                ##if i % 10 == 0:
                    ##print(i, outputs)
                #filename = ('./plot/10_rtm_nn_nl24_' + lat + '_toz300_epoch_' + str(epoch_local).zfill(8) +
                    #'_test_index_' + str(idx).zfill(8))
                ## every each epoch, plot for first
                ##if inbatch == 0:
                    ##test_plot300_logradtorad(epoch_local, inbatch, f_plot, wav300, r_plot,
                            ##outputs, filename, lr)
                
                ##if epoch_local % 500 == 0:
                    ###print(idx)
                    ##test_plot300_logradtorad(epoch_local, inbatch, f_plot, wav300, r_plot,
                            ##outputs, filename, lr)
                #outputs = None
                #del outputs

        lossesfile = './result/10_rtm_nn_sza_test_mean_losses.txt' 
        #print('lossesfile')
        if os.path.exists(lossesfile):
            with open(lossesfile, 'a') as f:
                f.write(str(epoch_local).zfill(8) + ',' + str(np.mean(train_losses))
                        + ',' + str(np.mean(test_losses)) + '\n')
        else:
            with open(lossesfile, 'w') as f:
                f.write('index,mean_train_losses,mean_valid_losses'+'\n')
                f.write(str(epoch_local).zfill(8) + ',' + str(np.mean(train_losses))
                        + ',' + str(np.mean(test_losses))  + '\n')



        torchstatefile = ('./states/project_' + projectname + '/' + 
            '10_rtm_nn_sza_test_epoch_' + str(epoch_local).zfill(8) + '.pth')
        if epoch_local % 100 == 0:
            torch.save({'epoch':epoch_local, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': np.mean(train_losses)}, torchstatefile)
                
        print('done')

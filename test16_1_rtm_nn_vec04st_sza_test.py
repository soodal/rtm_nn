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

#from ds_rtm_nn import RTM
class MLPv03(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(6, 200)
        linear2 = nn.Linear(200, 400)
        linear3 = nn.Linear(400, 600)
        linear4 = nn.Linear(600, 800)
        linear5 = nn.Linear(800, 1000)
        linear6 = nn.Linear(1000, 1200)
        linear7 = nn.Linear(1200, 1460)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.0)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv03, self).__init__()
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
from ds_rtm_nn import radplot_logradtorad
from ds_rtm_nn import load_LUT
from ds_rtm_nn import features_maker_toz
from ds_rtm_nn import XY_data_loader_toz_v3_sza_radlog
from ds_rtm_nn import search_lat_LUT_files
#from ds_rtm_nn import rad_custom_normalize
from ds_rtm_nn import input_custom_normalize
from ds_rtm_nn import wav_custom_normalize


if __name__ == '__main__':
# set up the neural network
    projectname = '16_1_vec04st'
    
    states_path = './states/project_' + projectname + '/'
    plot_path = './plot/project_' + projectname + '/'
    loss_path = './result/' 
    if not os.path.isdir(states_path):
        os.makedirs(states_path)
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    if not os.path.isdir(loss_path):
        os.makedirs(loss_path)

    LUT_file = '/home/ubuntu/works/data/GEMSTOOL/lutdata/LUTNC/LUT_vec04stNL24L300_sza_test_0.5deg.nc'


    model = MLPv03()
    print(model)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()

    #mean_train_losses = []
    #mean_valid_losses = []
    #mean_test_losses = []
    valid_acc_list = []

    epochs = 1000
    #load_epoch = 12500

    _epoch_list = []
    for (path, dir, files) in os.walk('./states/project_' + projectname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                _epoch = filename[-12:-4]
                _epoch_list.append(_epoch)


    if len(files) >= 1:
        if 'load_epoch' not in globals().keys():
            torchstatefile = './states/project_' + projectname + sorted(files)[-1]
            load_epoch = int(sorted(_epoch_list)[-1])
        else:
            torchstatefile = ('./states/project_' + projectname +
                    projectname + '_rtm_nn_sza_test_epoch_' +
                str(load_epoch).zfill(8) + '.pth')
            print(load_epoch)
        epoch_total = int(sorted(_epoch_list)[-1]) + 1
        print(epoch_total)

        checkpoint = torch.load(torchstatefile)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_total = checkpoint['epoch']
        loss = checkpoint['loss']
        #if epoch_total -1 !=  epoch_total:
            ##quit()
            #pass
    else:
        epochs_done = 0
        epoch_total = 0
        if 'load_epoch' not in globals().keys():
            load_epoch = 0

    for epoch in range(epochs):
        _epoch = load_epoch + epoch + 1
        print('_epoch is ', _epoch)
        model.train()
        train_losses = np.array([])
        test_losses = np.array([])
        valid_losses = np.array([])
        timestamp = time.time()

        #with open('./LUT/read_lut_check.txt', 'r') as f:
            #read_lut_list = f.read().splitlines()


        #read_count = 0

        #for LUT_file in LUT_filelist:

        print('_epoch check' , _epoch)
        #print(LUT_file)
        lat = LUT_file[-23]
        #print(lat)
        toz = list(np.array([int(LUT_file[-22:-19])]))
        print(lat, toz)

        #print(toz)
        
    #if read_count == 4:
        #break

    #if not LUT_file in read_lut_list:
        (sza, vza, raa, alb, pre, wav, ps_in, zs_in, ts_in, o3_in, taudp_in, nl_in,
            rad, albwf, o3wf) = load_LUT(LUT_file)

        #timestamp = time.time()

        wav_l = list(wav)
        #wav300 = wav[660:]
        wav_num = len(wav)
        wav_list = list(wav)

        sza_train_idx = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
            22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 
            52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80,
            82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 
            102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
            122, 124, 126, 128, 130, 132, 134, 136, 138, 140,
            142, 144, 146, 148, 150, 152, 154, 156, 158, 160,
            162, 164, 166, 168, 170, 172, 174, 176, 178], dtype=np.int)

        sza_test_idx = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
            23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 
            53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81,
            83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 
            103, 105, 107, 109, 111, 113, 115, 117, 119, 121,
            123, 125, 127, 129, 131, 133, 135, 137, 139, 141,
            143, 145, 147, 149, 151, 153, 155, 157, 159, 161,
            163, 165, 167, 169, 171, 173, 175, 177], dtype=np.int)

        (pre_list_train, alb_list_train, raa_list_train, vza_list_train,
                sza_list_train, toz_list_train) = features_maker_toz(
                pre, alb, raa, vza, sza[sza_train_idx], toz)    

        (pre_list_test, alb_list_test, raa_list_test, vza_list_test,
                sza_list_test, toz_list_test) = features_maker_toz(
                pre, alb, raa, vza, sza[sza_test_idx], toz)
    
        train_loader = XY_data_loader_toz_v3_sza_radlog(pre_list_train,
                alb_list_train, raa_list_train, vza_list_train, sza_list_train,
                toz_list_train, 
                rad[0][0][0][0][sza_train_idx][:], 
                albwf[0][0][0][0][sza_train_idx][:], 
                o3wf[0][0][0][0][sza_train_idx][:])

        test_loader = XY_data_loader_toz_v3_sza_radlog(pre_list_test,
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
            #print(i, features.shape, radiances.shape)
            #if real_epoch == 0:
            optimizer.zero_grad()
            outputs = model(features)
            loss = msre(outputs, radiances)
            loss.backward()
            optimizer.step()
            train_losses = np.append(train_losses, float(loss))

            # each batch is 128, print this for one of 10 batches
            if (i * 128) % (128 * 10) == 0: 
                print(f'{i * 128} / ', len(train_loader)*128, time.time() - timestamp,
                        datetime.datetime.now())
                print(loss.item())
                timestamp = time.time()

            if _epoch % 100 == 0:
                for inbatch in range(features.detach().numpy().shape[0]):
                    #print(i, outputs)
                    plotfn = (plot_path +  
                            projectname + 
                            '_rtm_nn_nl24_' + lat + 
                            '_toz300_epoch_' + str(_epoch).zfill(8) + 
                            '_train_index_' + str(i).zfill(8) + 
                            '_inbatch_' + str(inbatch).zfill(8))
                    # every each epoch, plot for first
                    radplot_logradtorad(_epoch, inbatch, features, wav, 
                            radiances, outputs, plotfn, lr)
                    
        model.eval()
        correct = 0
        total = 0

#test_loader
        with torch.no_grad():
            for i, (features, radiances) in enumerate(test_loader):
                outputs = model(features)
                loss = msre(outputs, radiances)
                
                #test_losses = np.append(test_losses, loss.item())
                test_losses = np.append(test_losses, float(loss))
               
                if _epoch % 100 == 0:
                    for inbatch in range(features.detach().numpy().shape[0]):
                        plotfn = (plot_path + 
                                + projectname + 
                                '_rtm_nn_nl24_' + lat + 
                                '_toz300_epoch_' + str(_epoch).zfill(8) +
                                '_test_index_' + str(i).zfill(8) + 
                                '_inbatch_' + str(inbatch).zfill(8))
                        radplot_logradtorad(_epoch, inbatch, features, wav, radiances,
                                outputs, plotfn, lr)

        #with open('./LUT/read_lut_list', 'a') as f:
            #f.write(LUT_file + '\n')

        #read_count += 1

        #if read_lut_list == LUT_filelist:
        #mean_train_losses.append(np.mean(train_losses))
        #mean_valid_losses.append(np.mean(valid_losses))
        #mean_test_losses.append(np.mean(test_losses))
        #print('len mean train losses ', len(mean_train_losses))

        loss_path = './result/' 
        lossfn = loss_path + projectname + '_rtm_nn_sza_test_mean_losses.txt' 
        i
        if os.path.exists(lossfn):
            with open(lossfn, 'a') as f:
                f.write(str(_epoch).zfill(8) + ',' + str(np.mean(train_losses))
                        + ',' + str(np.mean(test_losses)) + '\n')
        else:
            with open(lossfn, 'w') as f:
                f.write('index,mean_train_losses,mean_valid_losses'+'\n')
                f.write(str(_epoch).zfill(8) + ',' + str(np.mean(train_losses))
                        + ',' + str(np.mean(test_losses))  + '\n')



        torchstatefile = ('./states/project_' + projectname + '/' + 
            projectname + '_rtm_nn_sza_test_epoch_' + str(_epoch).zfill(8) + '.pth')
        if _epoch % 100 == 0:
            torch.save({'epoch':_epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': np.mean(train_losses)}, torchstatefile)
                
        print('done')

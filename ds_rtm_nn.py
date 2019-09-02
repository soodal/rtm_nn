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
from sklearn.metrics import mean_squared_error as mse
import input_test01_rtm_nn as input_params



class RTM(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X.index)
    
    def __getitem__(self, index): # need for enumerate
        #image = self.X.iloc[index, ].values.astype(np.uint8).reshape((28, 28, 1))
        features = self.X.iloc[index, ].values.astype(np.float32)
        if self.transform is not None:
            #image = self.transform(image)
            features = self.transform(features)
        if self.y is not None:
            #return image, self.y.iloc[index]
            return torch.tensor(features), torch.tensor(self.y.iloc[index])
        else:
            #return image
            return features

class MLP(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(5, 2000)
        linear2 = nn.Linear(2000, 2000)
        linear3 = nn.Linear(2000, 2000)
        linear4 = nn.Linear(2000, 2000)
        linear5 = nn.Linear(2000, 2000)
        linear6 = nn.Linear(2000, 2000)
        linear7 = nn.Linear(2000, 800)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLP, self).__init__()
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
def msre(outputs, target):
    loss = torch.mean(((outputs - target)/target)**2)
    return loss

def test_plot300(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], r_plot[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 10))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'
    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances, diffcolor, 
            linestyle='--', label='Relative Differences')

# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')
    ax2.set_ylabel('Relative Differences Ratio')

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
            str(features[0]), fontsize=16) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
            str(features[1]), fontsize=16) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
            str(features[2]), fontsize=16) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
            str(features[3]), fontsize=16) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
            str(features[4]), fontsize=16) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
            str(batch_idx), fontsize=16) 

    plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
            str(loss_.item()), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    pngfile = filename + '.png'
    txtfile = filename + '.txt'

    fig.savefig(pngfile)
    plt.close()

    with open(txtfile, 'w') as f:
        f.write('learning_rate(lr),' + str(lr) + '\n')
        f.write('surface_pressure,' + str(features[0]) + '\n')
        f.write('surface_albedo,' + str(features[1]) + '\n')
        f.write('relative_azimuth_angle,' + str(features[2]) + '\n')
        f.write('viewing_zenith_angle,'+ str(features[3]) + '\n')
        f.write('solar_zenith_angle,'+ str(features[4]) + '\n')
        f.write('wavelength,radiances,nn_radiances\n')
        for (i, rad) in enumerate(radiances):
            f.write(str(wav300[i]) + ',' + str(rad) + ',' + str(nn_radiances[i]) + '\n')

def test_plot(epoch, batch_idx, f_plot, wav, r_plot, outputs, filename, lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], r_plot[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 10))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')
    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances, 'r', 
            linestyle='--', label='Relative Differences')

# labels, units 
    ax1.set_xlabels('Wavelength[nm]')
    ax1.set_ylabels('Normalized Radiance[1/sr]')
    ax1.plot(wav, radiances, 'k', label='LBL RTM (True)')
    ax1.plot(wav, nn_radiances, 'b', label='NN RTM results')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    plt.text(280, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
            str(features[0]), fontsize=16) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
            str(features[1]), fontsize=16) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
            str(features[2]), fontsize=16) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
            str(features[3]), fontsize=16) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
            str(features[4]), fontsize=16) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
            str(batch_idx), fontsize=16) 

    plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
            str(loss_.item()), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    pngfile = savefilename + '.png'
    txtfile = savefilename + '.txt'

    fig.savefig(pngfile)
    plt.close()

    with open(txtfile, 'w') as f:
        f.write('learning_rate(lr),' + str(lr) + '\n')
        f.write('surface_pressure,' + str(features[0]) + '\n')
        f.write('surface_albedo,' + str(features[1]) + '\n')
        f.write('relative_azimuth_angle,' + str(features[2]) + '\n')
        f.write('viewing_zenith_angle,'+ str(features[3]) + '\n')
        f.write('solar_zenith_angle,'+ str(features[4]) + '\n')
        f.write('wavelength,radiances,nn_radiances\n')
        for (i, rad) in enumerate(radiances):
            f.write(str(wav[i]) + ',' + str(rad) + ',' + str(nn_radiances[i]) + '\n')

def load_LUT(LUT_file):
    data = None
    data = netCDF4.Dataset(LUT_file, mode='r')
    sza = None
    vza = None
    raa = None
    alb = None
    pre = None
    wav = None
    ps_in = None
    zs_in = None
    ts_in = None
    o3_in = None
    taudp_in = None
    nl_in = None
    rad = None
    albwf = None
    o3wf = None
    sza = data.variables['sza'][:]
    vza = data.variables['vza'][:]
    raa = data.variables['raa'][:]
    alb = data.variables['alb'][:]
    pre = data.variables['pre'][:]
    wav = data.variables['wav'][:]
    ps_in = data.variables['ps_in'][:]
    zs_in = data.variables['zs_in'][:]
    ts_in = data.variables['ts_in'][:]
    o3_in = data.variables['o3_in'][:]
    taudp_in = data.variables['taudp_in'][:]
    nl_in = data.variables['nl_in'][:]
    rad = data.variables['Radiance'][:]
    albwf = data.variables['albwf'][:]
    o3wf = data.variables['o3wf'][:]
    data = None

    del data
    return (sza, vza, raa, alb, pre, wav, ps_in, zs_in, ts_in, o3_in, taudp_in, 
            nl_in, rad, albwf, o3wf)

def features_maker(pre, alb, raa, vza, sza):
    pre_list = []
    alb_list = []
    raa_list = []
    vza_list = []
    sza_list = []
    wav_list = []
    for ipre in pre:
        for ialb in alb:
            for iraa in raa:
                for ivza in vza:
                    for isza in sza:
                        pre_list.append(ipre)
                        alb_list.append(ialb)
                        raa_list.append(iraa)
                        vza_list.append(ivza)
                        sza_list.append(isza)
    return pre_list, alb_list, raa_list, vza_list, sza_list

def XY_data_loader(pre_list, alb_list, raa_list, vza_list, sza_list, rad,
        albwf, o3wf):
    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[]})
    X = DataFrame({
        'pre':np.array(pre_list)/1050, 
        'alb':alb_list, 
        'raa':np.array(raa_list)/180, 
        'vza':np.cos(np.array(vza_list)/180*np.pi), 
        'sza':np.cos(np.array(sza_list)/180*np.pi)})#, 'wav':wav_list}))

    rad_ = rad.reshape((12*3*8*8*12, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    X_, X_test, Y_, Y_test = train_test_split(X, Y, test_size=1/6, random_state=2160)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size = 1/6, random_state=2161)
    print('train X shape : ', X_train.shape)
    print('train Y shape : ', Y_train.shape)
    print('valid X shape : ', X_valid.shape)
    print('valid Y shape : ', Y_valid.shape)
    print('test X shape : ', X_test.shape)
    print('test Y shape : ', Y_test.shape)
    print(type(X_train))
    train_dataset = RTM(X=X_train, y=Y_train, transform=None)
    valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader, valid_loader, test_loader
        

def search_LUT_files():
    LUT_filelist = []
    for (path, dir, files) in os.walk('./LUT/'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.nc':
                LUT_filelist.append(path + filename)
    return LUT_filelist

def search_lat_LUT_files(lat):
    LUT_filelist = []
    for (path, dir, files) in os.walk('./LUT/'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            #if ext == '.nc':
            if filename[:12] == 'LUT_sca02st'+ lat:
                LUT_filelist.append(path + filename)
    return LUT_filelist

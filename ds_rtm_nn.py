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

class MLPv01_5_1460(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(5, 2000)
        linear2 = nn.Linear(2000, 2000)
        linear3 = nn.Linear(2000, 2000)
        linear4 = nn.Linear(2000, 2000)
        linear5 = nn.Linear(2000, 2000)
        linear6 = nn.Linear(2000, 2000)
        linear7 = nn.Linear(2000, 1460)
        leakyrelu = nn.LeakyReLU()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv01_5_1460, self).__init__()
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

class MLPv01_5_800(nn.Module):
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

        super(MLPv01_5_800, self).__init__()
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

class MLPv01_6_1460(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(6, 2000)
        linear2 = nn.Linear(2000, 2000)
        linear3 = nn.Linear(2000, 2000)
        linear4 = nn.Linear(2000, 2000)
        linear5 = nn.Linear(2000, 2000)
        linear6 = nn.Linear(2000, 2000)
        linear7 = nn.Linear(2000, 1460)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv01_6_1460, self).__init__()
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

class MLPv01_single_wav(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(7, 2000)
        linear2 = nn.Linear(2000, 2000)
        linear3 = nn.Linear(2000, 2000)
        linear4 = nn.Linear(2000, 2000)
        linear5 = nn.Linear(2000, 2000)
        linear6 = nn.Linear(2000, 2000)
        linear7 = nn.Linear(2000, 1)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv01_single_wav, self).__init__()
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

class MLPv01_6_800(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(6, 2000)
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

        super(MLPv01_6_800, self).__init__()
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
    print('test_plot300 def start')
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()
    print('test_plot300 def 01')

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


    print('test_plot300 rdef 01')
# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

    print('test_plot300 def 01')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
    plt.text(300, -0.03, 'Surface pressure = ' + 
            str(format(float(features[0]) * 1050, ".2f")), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
    plt.text(300, -0.02, 'Surface albedo = ' +
            str(features[1]), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
    plt.text(300, -0.01, 'Relative azimuth angle = ' +
            str(float(features[2]) * 180), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
    plt.text(300, 0, 'Viewing zenith angle = ' +
            str(float(features[3]) * 180 / np.pi), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
    plt.text(300, 0.01, 'Solar zenith angle =' +
            str(float(features[4]) * 180 / np.pi), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
    plt.text(300, 0.02, 'Batch_index =' +
            str(batch_idx), fontsize=16) 

    print('test_plot300 def 01')
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, 0.2, 'MSRE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-0.1, 0.1])
    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    print(pngfile)

    fig.savefig(pngfile)
    plt.close()

    print('test_plot300 def write')
    with open(txtfile, 'w') as f:
        f.write('learning_rate(lr),' + str(lr) + '\n')
        print('test_plot300 def write 01')
        f.write('surface_pressure,' + str(features[0]) + '\n')
        print('test_plot300 def write 02')
        f.write('surface_albedo,' + str(features[1]) + '\n')
        print('test_plot300 def write 03')
        f.write('relative_azimuth_angle,' + str(features[2]) + '\n')
        print('test_plot300 def write 04')
        f.write('viewing_zenith_angle,'+ str(features[3]) + '\n')
        print('test_plot300 def write 05')
        f.write('solar_zenith_angle,'+ str(features[4]) + '\n')
        print('test_plot300 def write 06')
        f.write('wavelength,radiances,nn_radiances\n')
        for (i, rad) in enumerate(radiances):
            f.write(str(wav300[i]) + ',' + str(rad) + ',' + str(nn_radiances[i]) + '\n')

def test_plot300(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    print('test_plot300 def start')
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()
    print('test_plot300 def 01')

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


    print('test_plot300 rdef 01')
# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

    print('test_plot300 def 01')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
    plt.text(300, -0.3, 'Surface pressure = ' + 
            str(format(float(features[0]) * 1050, ".2f")), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
    plt.text(300, -0.2, 'Surface albedo = ' +
            str(features[1]), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
    plt.text(300, -0.1, 'Relative azimuth angle = ' +
            str(float(features[2]) * 180), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
    plt.text(300, 0, 'Viewing zenith angle = ' +
            str(float(features[3]) * 180 / np.pi), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
    plt.text(300, 0.1, 'Solar zenith angle =' +
            str(float(features[4]) * 180 / np.pi), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
    plt.text(300, 0.2, 'Batch_index =' +
            str(batch_idx), fontsize=16) 

    print('test_plot300 def 01')
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, 0.2, 'MSRE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-0.1, 0.1])
    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    print(pngfile)

    fig.savefig(pngfile)
    plt.close()

    print('test_plot300 def write')
    with open(txtfile, 'w') as f:
        f.write('learning_rate(lr),' + str(lr) + '\n')
        print('test_plot300 def write 01')
        f.write('surface_pressure,' + str(features[0]) + '\n')
        print('test_plot300 def write 02')
        f.write('surface_albedo,' + str(features[1]) + '\n')
        print('test_plot300 def write 03')
        f.write('relative_azimuth_angle,' + str(features[2]) + '\n')
        print('test_plot300 def write 04')
        f.write('viewing_zenith_angle,'+ str(features[3]) + '\n')
        print('test_plot300 def write 05')
        f.write('solar_zenith_angle,'+ str(features[4]) + '\n')
        print('test_plot300 def write 06')
        f.write('wavelength,radiances,nn_radiances\n')
        for (i, rad) in enumerate(radiances):
            f.write(str(wav300[i]) + ',' + str(rad) + ',' + str(nn_radiances[i]) + '\n')

def test_plot(epoch, batch_idx, f_plot, wav, r_plot, outputs, filename, lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])
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

def features_maker_toz_single_wav(pre, alb, raa, vza, sza, toz, wav):
    pre_list = []
    alb_list = []
    raa_list = []
    vza_list = []
    sza_list = []
    wav_list = []
    toz_list = []
    for ipre in pre:
        for ialb in alb:
            for iraa in raa:
                for ivza in vza:
                    for isza in sza:
                        for itoz in toz:
                            for iwav in wav:
                                pre_list.append(ipre)
                                alb_list.append(ialb)
                                raa_list.append(iraa)
                                vza_list.append(ivza)
                                sza_list.append(isza)
                                toz_list.append(itoz)
                                wav_list.append(iwav)
    return pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, wav_list

def features_maker_toz(pre, alb, raa, vza, sza, toz):
    pre_list = []
    alb_list = []
    raa_list = []
    vza_list = []
    sza_list = []
    wav_list = []
    toz_list = []
    for ipre in pre:
        for ialb in alb:
            for iraa in raa:
                for ivza in vza:
                    for isza in sza:
                        for itoz in toz:
                            pre_list.append(ipre)
                            alb_list.append(ialb)
                            raa_list.append(iraa)
                            vza_list.append(ivza)
                            sza_list.append(isza)
                            toz_list.append(itoz)
    return pre_list, alb_list, raa_list, vza_list, sza_list, toz_list

def XY_data_loader(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):
    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[]})
    X = DataFrame({
        'pre':np.array(pre_list)/1050, 
        'alb':alb_list, 
        'raa':np.array(raa_list)/180, 
        'vza':np.cos(np.array(vza_list)/180*np.pi), 
        'sza':np.cos(np.array(sza_list)/180*np.pi),
        'toz':np.array(toz_list)/550})#, 'wav':wav_list}))

    rad_ = rad.reshape((12*3*8*8*12, 1460))
    #rad_ = rad_[:, 660:]
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

def XY_data_loader_toz(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):
    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':np.array(pre_list)/1050, 
        'alb':alb_list, 
        'raa':np.array(raa_list)/180, 
        'vza':np.cos(np.array(vza_list)/180*np.pi), 
        'sza':np.cos(np.array(sza_list)/180*np.pi),
        'toz':np.array(toz_list)/550})#, 'wav':wav_list}))

    rad_ = rad.reshape((12*3*8*8*12, 1460))
    #rad_ = rad_[:, 660:]
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

def XY_data_loader_800_train_valid(pre_list, alb_list, raa_list, vza_list, sza_list, 
        rad, albwf, o3wf):
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

    X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=1/6, random_state=2160)
    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size = 1/6, random_state=2161)
    print('train X shape : ', X_train.shape)
    print('train Y shape : ', Y_train.shape)
    print('valid X shape : ', X_valid.shape)
    print('valid Y shape : ', Y_valid.shape)
    #print('test X shape : ', X_test.shape)
    #print('test Y shape : ', Y_test.shape)
    #print(type(X_train))
    train_dataset = RTM(X=X_train, y=Y_train, transform=None)
    valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader, valid_loader#, test_loader

def XY_data_loader_800(pre_list, alb_list, raa_list, vza_list, sza_list, 
        rad, albwf, o3wf):
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

def XY_data_loader_toz_800(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):
    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    #print(toz_list)
    #print(np.array(toz_list))
    #print(np.array(toz_list)/550)
    X = DataFrame({
        'pre':np.array(pre_list)/1050, 
        'alb':alb_list, 
        'raa':np.array(raa_list)/180, 
        'vza':np.cos(np.array(vza_list)/180*np.pi), 
        'sza':np.cos(np.array(sza_list)/180*np.pi),
        'toz':np.array(toz_list)/550})#, 'wav':wav_list}))

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

def XY_data_loader_toz_800_train_valid(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):
    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    #print(toz_list)
    #print(np.array(toz_list))
    #print(np.array(toz_list)/550)
    X = DataFrame({
        'pre':np.array(pre_list)/1050, 
        'alb':alb_list, 
        'raa':np.array(raa_list)/180, 
        'vza':np.cos(np.array(vza_list)/180*np.pi), 
        'sza':np.cos(np.array(sza_list)/180*np.pi),
        'toz':np.array(toz_list)/550})#, 'wav':wav_list}))

    rad_ = rad.reshape((12*3*8*8*12, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=1/6, random_state=2160)
    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size = 1/6, random_state=2161)
    print('train X shape : ', X_train.shape)
    print('train Y shape : ', Y_train.shape)
    print('valid X shape : ', X_valid.shape)
    print('valid Y shape : ', Y_valid.shape)
    #print('test X shape : ', X_test.shape)
    #print('test Y shape : ', Y_test.shape)
    #print(type(X_train))
    train_dataset = RTM(X=X_train, y=Y_train, transform=None)
    valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader, valid_loader#, test_loader

def custom_normalize(pre_list, alb_list, raa_list, vza_list, sza_list,
        toz_list, wav_list, rad):
    pre_array = np.array(pre_list)/1050
    alb_array = alb_list
    raa_array = np.array(raa_list)/180
    vza_array = np.cos(np.array(vza_list)/180*np.pi) 
    sza_array = np.cos(np.array(sza_list)/180*np.pi)
    toz_array = np.array(toz_list)/550
    wav_array = np.array(wav_list)/340
    rad_array = np.log(rad)
    return (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array,
            wav_array, rad_array)


def XY_data_loader_toz_800_train_radlog(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array,
            wav_array, rad_array) = custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list, wav_list, rad)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    #print(toz_list)
    #print(np.array(toz_list))
    #print(np.array(toz_list)/550)
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array,
        'wav':wav_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((12*3*8*8*12, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)
    print(len(rad_))


    #X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0, random_state=2160)
    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size = 1/6, random_state=2161)
    train_dataset = RTM(X=X, y=Y, transform=None)
    #valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader#, test_loader

def XY_data_loader_single_wav_train(pre_list, alb_list, raa_list, vza_list, 
        sza_list, toz_list, wav_list, 
        rad, albwf, o3wf):
    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array,
            wav_array, rad_array) = custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list, wav_list, rad)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})

    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array,
        'wav':wav_array})#, 'wav':wav_list}))


    rad_array = rad_array.reshape((12*3*8*8*12*1460, 1))
    Y = DataFrame(rad_array)

    train_dataset = RTM(X=X, y=Y, transform=None)
    #valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader#, test_loader

def XY_data_loader_toz_800_train(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array,
            wav_array, rad_array) = custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list, wav_list, rad)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    #print(toz_list)
    #print(np.array(toz_list))
    #print(np.array(toz_list)/550)
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array,
        'wav':wav_array})#, 'wav':wav_list}))

    rad_ = rad.reshape((12*3*8*8*12, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)
    print(len(rad_))


    #X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0, random_state=2160)
    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size = 1/6, random_state=2161)
    train_dataset = RTM(X=X, y=Y, transform=None)
    #valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader#, test_loader
        

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
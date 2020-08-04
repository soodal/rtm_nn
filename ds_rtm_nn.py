#!/usr/bin/env python
# coding: utf-8

import os
import time
import itertools
import datetime
# from IPython.display import Image
# from IPython import display
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
# from torchvision import datasets
# from torchvision import transforms

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

class MLPv03_13_6_800(nn.Module):
    def __init__(self):

        linear01 = nn.Linear(6, 200)
        linear02 = nn.Linear(200, 300)
        linear03 = nn.Linear(300, 300)
        linear04 = nn.Linear(300, 400)
        linear05 = nn.Linear(400, 400)
        linear06 = nn.Linear(400, 500)
        linear07 = nn.Linear(500, 500)
        linear08 = nn.Linear(500, 600)
        linear09 = nn.Linear(600, 600)
        linear10 = nn.Linear(600, 700)
        linear11 = nn.Linear(700, 700)
        linear12 = nn.Linear(700, 800)
        linear13 = nn.Linear(800, 800)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear01.weight)
        nn.init.xavier_uniform_(linear02.weight)
        nn.init.xavier_uniform_(linear03.weight)
        nn.init.xavier_uniform_(linear04.weight)
        nn.init.xavier_uniform_(linear05.weight)
        nn.init.xavier_uniform_(linear06.weight)
        nn.init.xavier_uniform_(linear07.weight)
        nn.init.xavier_uniform_(linear08.weight)
        nn.init.xavier_uniform_(linear09.weight)
        nn.init.xavier_uniform_(linear10.weight)
        nn.init.xavier_uniform_(linear11.weight)
        nn.init.xavier_uniform_(linear12.weight)
        nn.init.xavier_uniform_(linear13.weight)

        super(MLPv03_13_6_800, self).__init__()
        self.layers = nn.Sequential(
            linear01,
            leakyrelu,
            dropout,
            linear02,
            leakyrelu,
            dropout,
            linear03,
            leakyrelu,
            dropout,
            linear04,
            leakyrelu,
            dropout,
            linear05,
            leakyrelu,
            dropout,
            linear06,
            leakyrelu,
            dropout,
            linear07,
            leakyrelu,
            dropout,
            linear08,
            leakyrelu,
            dropout,
            linear09,
            leakyrelu,
            dropout,
            linear10,
            leakyrelu,
            dropout,
            linear11,
            leakyrelu,
            dropout,
            linear12,
            leakyrelu,
            dropout,
            linear13
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
class MLPv03_7_6_800(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(6, 200)
        linear2 = nn.Linear(200, 300)
        linear3 = nn.Linear(300, 400)
        linear4 = nn.Linear(400, 500)
        linear5 = nn.Linear(500, 600)
        linear6 = nn.Linear(600, 700)
        linear7 = nn.Linear(700, 800)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv03_7_6_800, self).__init__()
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

class MLPv02_6_200(nn.Module):
    def __init__(self):

        linear1 = nn.Linear(6, 200)
        linear2 = nn.Linear(200, 200)
        linear3 = nn.Linear(200, 200)
        linear4 = nn.Linear(200, 200)
        linear5 = nn.Linear(200, 200)
        linear6 = nn.Linear(200, 200)
        linear7 = nn.Linear(200, 200)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        super(MLPv02_6_200, self).__init__()
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
            linear7,
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
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
        dropout = nn.Dropout(0.3)

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


def msre_max(outputs, target):
    loss = torch.max(((outputs - target)/target)**2)
    return loss

def msre_log(outputs, target):
    loss = torch.log(torch.mean(((outputs - target)/target)**2))
    return loss

def msre(outputs, target):
    loss = torch.mean(((outputs - target)/target)**2)
    return loss


def test_plot300_radlog(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = np.exp(r_plot.detach().numpy()[batch_idx])
    nn_radiances = np.exp(outputs.detach().numpy()[batch_idx])
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = mse(radiances, nn_radiances)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

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

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSE = ' +
    plt.text(320, 0.02, 'MSE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-0.1, 0.1])
    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def test_plot300_mse(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = mse(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

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

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSE = ' +
    plt.text(320, 0.02, 'MSE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-0.1, 0.1])
    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def test_plot300_01(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
    #plt.text(300, -0.03, 'Surface pressure = ' + 
            #str(format(float(features[0]) * 1050, ".2f")), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
    #plt.text(300, -0.02, 'Surface albedo = ' +
            #str(features[1]), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
    #plt.text(300, -0.01, 'Relative azimuth angle = ' +
            #str(float(features[2]) * 180), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
    #plt.text(300, 0, 'Viewing zenith angle = ' +
            #str(float(features[3]) * 180 / np.pi), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
    plt.text(300, 0.01, 'Solar zenith angle =' +
            str(float(features[0]) * 180 / np.pi), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
    plt.text(300, 0.02, 'Batch_index =' +
            str(batch_idx), fontsize=16) 

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, 0.02, 'MSE = ' +
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
    #print(pngfile)

    fig.savefig(pngfile)
    plt.close()

    with open(txtfile, 'w') as f:
        f.write('learning_rate(lr),' + str(lr) + '\n')
        #f.write('surface_pressure,' + str(features[0]) + '\n')
        #f.write('surface_albedo,' + str(features[1]) + '\n')
        #f.write('relative_azimuth_angle,' + str(features[2]) + '\n')
        #f.write('viewing_zenith_angle,'+ str(features[3]) + '\n')
        f.write('solar_zenith_angle,'+ str(features[0]) + '\n')
        f.write('wavelength,radiances,nn_radiances\n')
        for (i, rad) in enumerate(radiances):
            f.write(str(wav300[i]) + ',' + str(rad) + ',' + str(nn_radiances[i]) + '\n')

def test_plot300_mini(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])*100
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

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

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, 0.02, 'MSRE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-0.01, 0.01])
    line3 = ax2.plot(wav300, (np.exp(nn_radiances) - np.exp(radiances))/np.exp(radiances), diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def test_plot300_logradtorad(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = np.exp(r_plot.detach().numpy()[batch_idx])
    nn_radiances = np.exp(outputs.detach().numpy()[batch_idx])
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

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

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, 0.02, 'MSRE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-10, 10])

    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances*100, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def radplot_logradtorad_detached(epoch, batch_idx, f_plot, wav, r_plot, outputs, filename,
        lr):
    radiances = np.exp(r_plot[batch_idx,:])
    nn_radiances = np.exp(outputs[batch_idx,:])
    features = f_plot.detach().numpy()[batch_idx,:]
    #loss_fn = nn.L1Loss()
    print(r_plot[batch_idx,:].shape)
    print(r_plot[batch_idx, :].shape)
    print(outputs[batch_idx, :].shape)

    #loss_ = loss_fn(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

    x0, y0, x1, y1 = ax1.get_position().bounds
    #print(x0, y0, x1, y1)
    ax1.set_position([0.15, 0.12, 0.70, 0.76])

# line plot
    line1 = ax1.plot(wav, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav, nn_radiances, 'b', label='NN RTM results')

# line plot for 0.05%
    #y05p = np.zeros(1460) + 0.05
    #y05m = -1 * y05p
    #line4 = ax2.plot(wav, y05p, 'grey', linestyle = ':')
    #line4 = ax2.plot(wav, y05m, 'grey', linestyle = ':')

    diffcolor = 'r'

# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

    sza_str = str(np.round(np.arccos(float(features[4])) * 180 / np.pi, 1))
    #'V015R090A10.0%B1050hPaL300'
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    #plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    plt.title('S' + sza_str + 'V015R090A10.0%B1050hPaL300')
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
    #plt.text(300, -6, 'Surface pressure = ' + 
            #str(format(float(features[0]) * 1050, ".2f")), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
    #plt.text(300, -4, 'Surface albedo = ' +
            #str(features[1]), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
    #plt.text(300, -2, 'Relative azimuth angle = ' +
            #str(float(features[2]) * 180), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
    #plt.text(300, 0, 'Viewing zenith angle = ' +
            #str(float(features[3]) * 180 / np.pi), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +

    #plt.text(300, 2, 'Solar zenith angle =' +
            #str(np.round(np.arccos(float(features[4])) * 180 / np.pi)), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
    #plt.text(300, 4, 'Batch_index =' +
            #str(batch_idx), fontsize=16) 

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    #plt.text(270, -2.5, 'MAE = ' +
            #str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Difference[%]')
    ax2.set_ylim([-1, 1])

    line3 = ax2.plot(wav, (nn_radiances - radiances)/radiances*100, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav)))
    
    line4 = ax2.plot(wav, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def radplot_logradtorad(epoch, batch_idx, f_plot, wav, r_plot, outputs, filename,
        lr):
    radiances = np.exp(r_plot.detach().numpy()[batch_idx])
    nn_radiances = np.exp(outputs.detach().numpy()[batch_idx])
    features = f_plot.detach().numpy()[batch_idx]
    loss_fn = nn.L1Loss()
    loss_ = loss_fn(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7*5, 5*5))
    ax2 = ax1.twinx()

    x0, y0, x1, y1 = ax1.get_position().bounds
    #print(x0, y0, x1, y1)
    ax1.set_position([0.15, 0.12, 0.70, 0.76])

# line plot
    line1 = ax1.plot(wav, radiances, 'k', linestyle='solid', 
                    label='LBL RTM (True)', linewidth=1*2)
    line2 = ax1.plot(wav, nn_radiances, 'b', linestyle='dashed', 
                    label='NN RTM results', linewidth=1*5)

# line plot for 0.05%
    #y05p = np.zeros(1460) + 0.05
    #y05m = -1 * y05p
    #line4 = ax2.plot(wav, y05p, 'grey', linestyle = ':')
    #line4 = ax2.plot(wav, y05m, 'grey', linestyle = ':')

    diffcolor = 'r'

# labels, units 
    ax1.set_xlabel('Wavelength[nm]', fontsize=14*5)
    ax1.set_ylabel('Normalized Radiance[1/sr]', fontsize=14*5)

    sza_str = str(np.round(np.arccos(float(features[4])) * 180 / np.pi, 1))
    #'V015R090A10.0%B1050hPaL300'
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best', fontsize=12*5)
    ax1.tick_params(which='major', labelsize=10*5)
    #plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    plt.title('S' + sza_str + 'V015R090A10.0%B1050hPaL300', fontsize=16*5)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
    #plt.text(300, -6, 'Surface pressure = ' + 
            #str(format(float(features[0]) * 1050, ".2f")), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
    #plt.text(300, -4, 'Surface albedo = ' +
            #str(features[1]), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
    #plt.text(300, -2, 'Relative azimuth angle = ' +
            #str(float(features[2]) * 180), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
    #plt.text(300, 0, 'Viewing zenith angle = ' +
            #str(float(features[3]) * 180 / np.pi), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +

    #plt.text(300, 2, 'Solar zenith angle =' +
            #str(np.round(np.arccos(float(features[4])) * 180 / np.pi)), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
    #plt.text(300, 4, 'Batch_index =' +
            #str(batch_idx), fontsize=16) 

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(270, -2.5, 'MAE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Difference[%]', color='red', fontsize=14*5)
    ax2.set_ylim([-1, 1])
    ax2.set_yticklabels([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], fontsize=14*5)
    ax2.tick_params(axis='y', labelcolor='red')

    line3 = ax2.plot(wav, (nn_radiances - radiances)/radiances*100, diffcolor, 
            linestyle='--', linewidth=1*5, label='Relative Differences')

    zeros = np.zeros((len(wav)))
    
    line4 = ax2.plot(wav, zeros, 'grey', linestyle=':', linewidth=2)

    #ax2.spines["right"].set_edgecolor(line3.get_color())

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def radplot300_logradtorad(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = np.exp(r_plot.detach().numpy()[batch_idx])
    nn_radiances = np.exp(outputs.detach().numpy()[batch_idx])
    features = f_plot.detach().numpy()[batch_idx]
    loss_fn = nn.L1Loss()
    loss_ = loss_fn(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

    x0, y0, x1, y1 = ax1.get_position().bounds
    #print(x0, y0, x1, y1)
    ax1.set_position([0.15, 0.12, 0.70, 0.76])

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

    sza_str = str(np.round(np.arccos(float(features[4])) * 180 / np.pi))
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='best')
    #plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    plt.title('S' + sza_str + 'V015R090A10.0%B1050hPaL300')
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
    #plt.text(300, -6, 'Surface pressure = ' + 
            #str(format(float(features[0]) * 1050, ".2f")), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
    #plt.text(300, -4, 'Surface albedo = ' +
            #str(features[1]), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
    #plt.text(300, -2, 'Relative azimuth angle = ' +
            #str(float(features[2]) * 180), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
    #plt.text(300, 0, 'Viewing zenith angle = ' +
            #str(float(features[3]) * 180 / np.pi), fontsize=16) 
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
    #plt.text(300, 2, 'Solar zenith angle =' +
            #str(float(features[4]) * 180 / np.pi), fontsize=16)
    #plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
    #plt.text(300, 4, 'Batch_index =' +
            #str(batch_idx), fontsize=16) 

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, -4, 'MAE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences [%]')
    ax2.set_ylim([-10, 10])

    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances*100, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

    fig.savefig(pngfile)
    #plt.show()
    #quit()
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

def test_plot300(epoch, batch_idx, f_plot, wav300, r_plot, outputs, filename,
        lr):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], outputs[batch_idx])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax2 = ax1.twinx()

# line plot
    line1 = ax1.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    line2 = ax1.plot(wav300, nn_radiances, 'b', label='NN RTM results')

    diffcolor = 'r'


# labels, units 
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Normalized Radiance[1/sr]')

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

    #plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' +
    plt.text(320, 0.02, 'MSRE = ' +
            str(format(loss_.item(), ".10f")), fontsize=16)
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSRE total = ' + str(loss))

    ax2.set_ylabel('Relative Differences Ratio')
    ax2.set_ylim([-10, 10])

    line3 = ax2.plot(wav300, (nn_radiances - radiances)/radiances*100, diffcolor, 
            linestyle='--', label='Relative Differences')
    zeros = np.zeros((len(wav300)))
    
    line4 = ax2.plot(wav300, zeros, 'grey', linestyle=':')

    pngfile = filename + '.png'
    txtfile = filename + '.txt'
    #print(pngfile)

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

def input_custom_normalize(pre_list, alb_list, raa_list, vza_list, sza_list,
        toz_list):
    pre_array = np.array(pre_list)/1050
    alb_array = alb_list
    raa_array = np.array(raa_list)/180
    vza_array = np.cos(np.array(vza_list)/180*np.pi) 
    sza_array = np.cos(np.array(sza_list)/180*np.pi)
    toz_array = np.array(toz_list)/550
    return (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array)



def wav_custom_normalize(wav_list):
    wav_array = np.array(wav_list)/340
    return wav_array

def wav_custom_normalize_v2(wav_list):
    wav_array = (np.array(wav_list)-270)/340
    return wav_array

def XY_data_loader_toz_800_test(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)
    npre = len(pre_array)
    nalb = len(alb_array)
    nraa = len(raa_array)
    nvza = len(vza_array)
    nsza = len(sza_array)
    ntoz = len(toz_array)



    #rad_array = np.log(rad)
    rad_array = np.array(rad)

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
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((npre*nalb*nraa*nvza*nsza, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)
    print(len(rad_))


    #X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0, random_state=2160)
    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size = 1/6, random_state=2161)
    dataset = RTM(X=X, y=Y, transform=None)
    #valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return loader#, test_loader

def XY_data_loader_toz_800_test_radlog(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)
    npre = len(pre_array)
    nalb = len(alb_array)
    nraa = len(raa_array)
    nvza = len(vza_array)
    nsza = len(sza_array)
    ntoz = len(toz_array)



    rad_array = np.log(rad)

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
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((npre*nalb*nraa*nvza*nsza, 1460))
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
def XY_data_loader_toz_800_train_radlog(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)

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
        'toz':toz_array})#, 'wav':wav_list}))

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

def XY_data_loader_toz_800_v2_sza_all(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((93*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_800_v3_sza_train(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    #rad_array = np.log(rad)
    rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((12*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_800_v2_sza_train_01(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    #rad_array = np.log(rad)
    rad_array = rad

    #X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        #'pre':pre_array,
        #'alb':alb_array,
        #'raa':raa_array,
        #'vza':vza_array,
        'sza':sza_array})#,
        #'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((12*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_800_v2_sza_train_radlog_mini(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    prelen = len(pre_list)

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)
    #rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((prelen, 1460))
    rad_ = rad_[:, 1260:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_v3_sza_rad_standard(pre_list, alb_list, raa_list, 
        vza_list, sza_list, toz_list, rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    prelen = len(pre_list)

    #rad_array = np.log(rad)
    rad_array = np.array(rad)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))
    #print(rad.shape)
    #print(rad_array.shape)
    #print(rad[1, 1], rad[1, 1])
    #print(rad[0, 0], rad[0, 1])
    
    #print(rad_array.shape)
    rad_ = rad_array.reshape((prelen, 1460))

    #print(rad_[1, 1], rad_[1, 1])
    #print(rad_[0, 0], rad_[0, 1])
    rad_mean = np.mean(rad_array, axis=0)
    rad_std = np.std(rad_array, axis=0)
    #print(rad_mean.shape)
    rad_szamean = np.ndarray([prelen, 1460])
    rad_szastd = np.ndarray([prelen, 1460])

    #print(rad_szamean.shape)
    for i in range(prelen):
        rad_szamean[i, :] = rad_mean
        rad_szastd[i, :] = rad_std


    #print(rad_.shape)
    rad_ = (rad_ - rad_szamean) / rad_szastd

    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

# flag
def XY_data_loader_toz_v3_sza_radlog(pre_list, alb_list, raa_list, vza_list, 
        sza_list, toz_list, 
        rad, albwf, o3wf):
    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    prelen = len(pre_list)

    rad_array = np.log(rad)
    #rad_array = rad

    #X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((prelen, 1460))
    #rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_800_v2_sza_train_radlog(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)
    #rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((12*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_800_v2_sza_train(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    #rad_array = np.log(rad)
    rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((12*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    train_dataset = RTM(X=X, y=Y, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader

def XY_data_loader_toz_800_v3_sza_test(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    #rad_array = np.log(rad)
    rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((81*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    test_dataset = RTM(X=X, y=Y, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return test_loader

def XY_data_loader_toz_800_v2_sza_test_01(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    #rad_array = np.log(rad)
    rad_array = rad

    #X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        #'pre':pre_array,
        #'alb':alb_array,
        #'raa':raa_array,
        #'vza':vza_array,
        'sza':sza_array})#,
        #'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((81*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    test_dataset = RTM(X=X, y=Y, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return test_loader

def XY_data_loader_toz_800_v2_sza_test_radlog_mini(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    prelen = len(pre_list)

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)
    #rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((prelen, 1460))
    rad_ = rad_[:, 1260:]
    Y = DataFrame(rad_)

    test_dataset = RTM(X=X, y=Y, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return test_loader
def XY_data_loader_toz_v3_sza_test_radlog(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)
    #rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((81*1*1*1*1, 1460))
    #rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    test_dataset = RTM(X=X, y=Y, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return test_loader

def XY_data_loader_toz_800_v2_sza_test_radlog(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    rad_array = np.log(rad)
    #rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((81*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    test_dataset = RTM(X=X, y=Y, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return test_loader
def XY_data_loader_toz_800_v2_sza_test(pre_list, alb_list, raa_list, vza_list, sza_list, toz_list, 
        rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    #rad_array = np.log(rad)
    rad_array = rad

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

    rad_ = rad_array.reshape((81*1*1*1*1, 1460))
    rad_ = rad_[:, 660:]
    Y = DataFrame(rad_)

    test_dataset = RTM(X=X, y=Y, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return test_loader

# flag
def XY_data_loader_single_wav_train(pre_list, alb_list, raa_list, vza_list, 
        sza_list, toz_list, wav_list, 
        rad, albwf, o3wf):
    prelen = len(pre_list)
    (pre_array, alb_array, raa_array, vza_array, sza_array, toz_array
            ) = input_custom_normalize(pre_list, alb_list, 
                    raa_list, vza_list, sza_list, toz_list)

    prelen = len(pre_list)

    wav_array = wav_custom_normalize(wav_list)
    rad_array = np.log(rad) 


    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[],
            'wav':[]})

    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array,
        'wav':wav_array})#, 'wav':wav_list}))

    print(rad_array.shape)
    print(pre_array.shape)
    print(prelen)

    rad_array = rad_array.reshape((prelen, 1))
    Y = DataFrame(rad_array)

    train_dataset = RTM(X=X, y=Y, transform=None)
    #valid_dataset = RTM(X=X_valid, y=Y_valid, transform=None)
    #test_dataset = RTM(X=X_test, y=Y_test, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader#, test_loader

def XY_data_loader_toz_800_train(pre_list, alb_list, raa_list, vza_list,
        sza_list, toz_list, rad, albwf, o3wf):

    (pre_array, alb_array, raa_array, vza_array, sza_array, 
            toz_array) = input_custom_normalize(pre_list, alb_list, raa_list, vza_list, 
                    sza_list, toz_list)

    X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[], 'toz':[]})
    X = DataFrame({
        'pre':pre_array,
        'alb':alb_array,
        'raa':raa_array,
        'vza':vza_array,
        'sza':sza_array,
        'toz':toz_array})#, 'wav':wav_list}))

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

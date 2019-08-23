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
        super(MLP, self).__init__()

        linear1 = nn.Linear(5, 2000)
        linear2 = nn.Linear(2000, 2000)
        linear3 = nn.Linear(2000, 2000)
        linear4 = nn.Linear(2000, 2000)
        linear5 = nn.Linear(2000, 2000)
        linear6 = nn.Linear(2000, 2000)
        linear7 = nn.Linear(2000, 1460)
        leakyrelu = nn.LeakyReLU()
        dropout = nn.Dropout(0.5)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)

        self.layers = nn.Sequential(
            linear1,
            dropout,
            leakyrelu,
            linear2,
            dropout,
            leakyrelu,
            linear3,
            dropout,
            leakyrelu,
            linear4,
            dropout,
            leakyrelu,
            linear5,
            dropout,
            leakyrelu,
            linear6,
            dropout,
            leakyrelu,
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

def test_plot(epoch, batch_idx, f_plot, wav300, r_plot, outputs):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    loss_ = msre(r_plot[batch_idx], r_plot[batch_idx])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 10))
    ax.plot(wav300, radiances, 'k', label='LBL RTM (True)')
    ax.plot(wav300, nn_radiances, 'b', label='NN RTM results')
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    plt.text(300, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
            str(features[0])) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
            str(features[1])) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
            str(features[2])) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
            str(features[3])) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
            str(features[4])) 
    plt.text(300, max(max(radiances), max(nn_radiances))*0.8, 'Batch_index =' +
            str(batch_idx)) 
    
    plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSRE = ' + str(loss_.item()))


    pngfile = ('./plot/02_rtm_nn_' + lat + toz + '_epoch_' + str(epoch).zfill(5) +
        '_index_' + str(batch_idx).zfill(5) + '.png')
    txtfile = ('./plot/02_rtm_nn_' + lat + toz + '_epoch_' + str(epoch).zfill(5) +
        '_index_' + str(batch_idx).zfill(5) + '.txt')

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
    zs_in = Nona
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

class RTM_Dataset(Dataset):
    """custom dataset"""

# Initialize my data, download, etc
    def __init__(self, x, transforms=None):
        data = netCDF4.Dataset(LUT_file, mode='r')
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

        
        self.x_data = DataFrame({
            'pre':np.array(pre_list)/1050, 
            'alb':alb_list, 
            'raa':np.array(raa_list)/180, 
            'vza':np.cos(np.array(vza_list)/180*np.pi), 
            'sza':np.cos(np.array(sza_list)/180*np.pi)})#, 'wav':wav_list}))
        self.y_data = DataFrame(rad.reshape(12*3*8*8*12, 1460))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return len(self.x_data)

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
    #Y = DataFrame(rad.reshape(12*3*8*8*12, 1460))
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
            #if ext == '.nc':
            if filename[:13] == 'LUT_sca02stL':
                LUT_filelist.append(path + filename)
    return LUT_filelist


if __name__ == '__main__':
# set up the neural network

    LUT_filelist = search_LUT_files()

    model = MLP()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()


    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []

    epochs = 1000

    _epoch_list = []
    for (path, dir, files) in os.walk('./states_02/'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                _epoch = filename[35:40]
                _epoch_list.append(_epoch)

    if len(files) >= 1:
        torchstatefile = './states_02/' + sorted(files)[-1]
        model.load_state_dict(torch.load(torchstatefile))
        real_epoch = int(sorted(_epoch_list)[-1]) + 1
        print('real_epoch = ', real_epoch)
    else:
        real_epoch = 0

#for epoch in range(epochs):
    print('real_epoch is ', real_epoch)
    model.train()
    train_losses = np.array([])
    valid_losses = np.array([])
    timestamp = time.time()
    LUT_filelist = search_LUT_files()

    #with open('./LUT/read_lut_check.txt', 'r') as f:
        #read_lut_list = f.read().splitlines()

    train_losses = np.load('./losses/test02_train_losses.npy')
    valid_losses = np.load('./losses/test02_valid_losses.npy')

    #read_count = 0

    for LUT_file in LUT_filelist:
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


            #dataiter = iter(train_loader)
            #features, radiances = dataiter.next()
            #print(type(features), type(radiances))

            #print('features shape on PyTroch : ', features.size())
            #print('radiances shape on PyTroch : ', radiances.size())


            for i, (features, radiances) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, radiances)
                loss.backward()
                optimizer.step()
                #train_losses.append(loss.item())
                train_losses = np.append(train_losses, loss.item())
                #train_losses.append(loss.data)
                if (i * 128) % (128 * 10) == 0:
                    print(f'{i * 128} / ', len(train_loader)*128, time.time() - timestamp)
                    print(type(loss), float(loss), loss.item(), loss.data, len(train_loader),
                            len(train_losses))
                    print(type(loss.data), type(loss.item()))

                    timestamp = time.time()
                outputs = None
                loss = None
                del outputs, loss
                    
            print('model.eval()')
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (features, radiances) in enumerate(valid_loader):
                    outputs = model(features)
                    loss = loss_fn(outputs, radiances)
                    
                    #valid_losses.append(loss.item())
                    valid_losses = np.append(valid_losses, loss.item())
                    #valid_losses.append(loss.data)
                   
                    #_, predicted = torch.max(outputs.data, 1)
                    #correct += (predicted == radiances).sum().item()
                    #total += radiances.size(0)
                    outputs = None
                    loss = None
                    del outputs, loss

            # garbage collector


            with open('./LUT/read_lut_list', 'a') as f:
                f.write(LUT_file + '\n')

            np.save('./train_losses/test02_train_losses.npy',
                    np.array(train_losses), allow_pickle=True)
            np.save('./valid_losses/test02_valid_losses.npy',
                    np.array(valid_losses), allow_pickle=True)

            read_count += 1

    if read_lut_list == LUT_filelist:
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
        print('len mean train losses ', len(mean_train_losses))

        for i, (f_plot, r_plot) in enumerate(test_loader):
            outputs = model(f_plot)
            if i % 10 == 0:
                print(i, outputs)
#test_loader
        batch_idx = 0
        test_plot(real_epoch, batch_idx, f_plot, wav300, r_plot, outputs)
        outputs = None
        del outputs
        #if real_epoch % 5 == 0:
        torchstatefile = './states_02/02_rtm_nn_lat_toz_epoch_' + str(epoch).zfill(5) + '.pth'
        torch.save(model.state_dict(), torchstatefile)

        lossesfile = './result/02_rtm_nn_lat_toz_mean_losses.txt' 
        if os.path.exists(lossesfile):
            with open(lossesfile, 'a') as f:
                f.write(str(real_epoch).zfill(5) + ',' + str(np.mean(train_losses))
                        + ',' + str(np.mean(valid_losses))+'\n')
        else:
            with open(lossesfile, 'w') as f:
                f.write('index,mean_train_losses,mean_valid_losses'+'\n')
                f.write(str(real_epoch).zfill(5) + ',' + str(np.mean(train_losses))
                        + ',' + str(np.mean(valid_losses))+'\n')

#with open('./result/02_rtm_nn_normalized_lat_toz_epoch_' + str(epoch).zfill(5) + '')

#accuracy = 100*correct/total
#valid_acc_list.append(accuracy)
#print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%'\
#     .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))
        real_epoch = real_epoch + 1
        train_losses = None
        valid_losses = None
        del train_losses, valid_losses

        with open('./LUT/read_LUT_check.txt', 'w') as f:
            pass


# In[ ]:


#print(outputs.data)
#print(len(_))
#print(loss)
#print(valid_losses)
#print(predicted)
#print(len(predicted))
#print(radiances)
#print(len(radiances))
#print(outputs)
#print(len(outputs))

#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        #ax.plot(mean_train_losses, label='train loss')
        #ax.plot(mean_valid_losses, label='valid loss')
        #lines, labels = ax1.get_legend_handles_labels()
        #ax.legend(lines, labels, loc='best')
        #ax.title('Losses')

#ax2.plot(valid_acc_list, label='valid acc')
#ax2.legend()
        #fig.savefig('./plot/02_rtm_nn_normalized_lat_toz_losses.png')
        #fig.cla()

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=15, 10))
#ax.plot(radiances, label='Radiances(True)', 'k')
#ax.plot(outputs, label='NN RTM results', 'b')
#lines, labels = ax.get_legend_handles_labels()
#ax.legend(lines, labels, loc='best')
#fig.savefig('./plot/')
#fig.cla()

        model.eval()
        test_preds = torch.LongTensor()

        for batch_idx, (features, radiances) in enumerate(test_loader):
            outputs = model(features)
            loss = loss_fn(outputs, radiances)
            valid_losses.append(loss.item())
            #valid_losses.append(loss.data)

            test_plot(real_epoch -1, batch_idx, f_plot, wav300, r_plot, outputs)

        


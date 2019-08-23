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


def test_plot(epoch, batch_idx, f_plot, wav, r_plot, outputs):
    radiances = r_plot.detach().numpy()[batch_idx]
    nn_radiances = outputs.detach().numpy()[batch_idx]
    features = f_plot.detach().numpy()[batch_idx]
    

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 10))

    ax.plot(wav, radiances, 'k', label='LBL RTM (True)')
    ax.plot(wav, nn_radiances, 'b', label='NN RTM results')

    lines, labels = ax.get_legend_handles_labels()

    ax.legend(lines, labels, loc='best')

    plt.title('Neural Network Radiance Simulator Epoch:' + str(epoch).zfill(5))
    
    plt.text(280, max(max(radiances), max(nn_radiances))*0.3, 'Surface pressure = ' + 
            str(features[0])) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.4, 'Surface albedo = ' +
            str(features[1])) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.5, 'Relative azimuth angle = ' +
            str(features[2])) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.6, 'Viewing zenith angle = ' +
            str(features[3])) 
    plt.text(280, max(max(radiances), max(nn_radiances))*0.7, 'Solar zenith angle =' +
            str(features[4])) 

    plt.text(320, max(max(radiances), max(nn_radiances))*0.2, 'MSE = ' + str(loss_.item()))
    #plt.text(320, max(max(radiances), max(nn_radiances))*0.1, 'MSE total = ' + str(loss))

    pngfile = ('./plot/01_rtm_nn_' + lat + toz + '_normalize_epoch_' + str(epoch).zfill(5) +
        '_index_' + str(batch_idx).zfill(5) + '.png')
    txtfile = ('./plot/01_rtm_nn_' + lat + toz + '_normalize_epoch_' + str(epoch).zfill(5) +
        '_index_' + str(batch_idx).zfill(5) + '.txt')
    with open(txtfile, 'w') as f:
        f.write('surface_pressure,' + str(features[0]) + '\n')
        f.write('surface_albedo,' + str(features[1]) + '\n')
        f.write('relative_azimuth_angle,' + str(features[2]) + '\n')
        f.write('viewing_zenith_angle,'+ str(features[3]) + '\n')
        f.write('solar_zenith_angle,'+ str(features[4]) + '\n')
        f.write('wavelength,radiances,nn_radiances\n')
        for (i, rad) in enumerate(radiances):
            f.write(str(wav[i]) + ',' + str(radiances[i]) + ',' +
                    str(nn_radiances[i]) + '\n')
    fig.savefig(pngfile)
    plt.close()


LUT_file = input_params.LUT_file
lat = LUT_file[17]
toz = LUT_file[18:21]
lr = input_params.lr

data = netCDF4.Dataset(LUT_file, mode='r')
print('nc read done')

sza = data.variables['sza'][:]
print('sza allocated')
vza = data.variables['vza'][:]
print('vza allocated')
raa = data.variables['raa'][:]
print('raa allocated')
alb = data.variables['alb'][:]
print('alb allocated')
pre = data.variables['pre'][:]
print('pre allocated')
wav = data.variables['wav'][:]
print('wav allocated')
ps_in = data.variables['ps_in'][:]
print('ps_in allocated')
zs_in = data.variables['zs_in'][:]
print('zs_in allocated')
ts_in = data.variables['ts_in'][:]
print('ts_in allocated')
o3_in = data.variables['o3_in'][:]
print('os_in allocated')
taudp_in = data.variables['taudp_in'][:]
print('taudp_in allocated')
nl_in = data.variables['nl_in'][:]
print('nl_in allocated')
rad = data.variables['Radiance'][:]
print('rad allocated')
albwf = data.variables['albwf'][:]
print('albwf allocated')
o3wf = data.variables['o3wf'][:]
print('o3wf allocated')

print('cast read variables')
# In[4]:

X = DataFrame({'pre':[], 'alb':[], 'raa':[], 'vza':[], 'sza':[]})

timestamp = time.time()

pre_list = []
alb_list = []
raa_list = []
vza_list = []
sza_list = []
wav_list = []
wav_l = list(wav)
wav_num = len(wav)
for ipre in pre:
    #print(ipre)
    for ialb in alb:
        #print(ialb)
        for iraa in raa:
            #print(iraa)
            for ivza in vza:
                #print(ivza)
                for isza in sza:
                    #print(isza)
                    #for iwav in wav:
                        #print(iwav)
                    #pre_list = pre_list + [ipre for i in range(wav_num)]
                    #alb_list = alb_list + [ialb for i in range(wav_num)]
                    #raa_list = raa_list + [iraa for i in range(wav_num)]
                    #vza_list = vza_list + [ivza for i in range(wav_num)]
                    #sza_list = sza_list + [isza for i in range(wav_num)]
                    #wav_list = wav_list + wav_l
                    pre_list.append(ipre)
                    alb_list.append(ialb)
                    raa_list.append(iraa)
                    vza_list.append(ivza)
                    sza_list.append(isza)
                    #wav_list.append(iwav)

                    
                    #X_dict = {'pre':[ipre for i in range(wav_num)], 
                    #          'alb':[ialb for i in range(wav_num)], 
                    #          'raa':[iraa for i in range(wav_num)], 
                    #          'vza':[ivza for i in range(wav_num)], 
                    #          'sza':[isza for i in range(wav_num)], 
                    #          'wav':[iwav for i in range(wav_num)]})])
        #print('--- %s seconds ---' % (time.time() - timestamp))
        #timestamp = time.time()
        
print(len(pre_list))
#print(len(wav_list))


#print(data.variables['Radiance'])
print(np.cos(np.array(vza)/180*np.pi))
print(np.log(pre))
print(alb)
print(sza)
print(vza)
print(raa/180)
X = DataFrame(
    {'pre':np.array(pre_list)/1050, 
     'alb':alb_list, 
     'raa':np.array(raa_list)/180,
     'vza':np.cos(np.array(vza_list)/180*np.pi), 
     'sza':np.cos(np.array(sza_list)/180*np.pi)})#, 'wav':wav_list}))
print(X)


# In[6]:



# In[7]:



#X = np.array(X)
#print(X)
#print(X.shape)
#X = X.reshape(12*3*8*8*12, 5)
#print(type(rad))
rad_ = rad[:, :, :, :, :, 660:]
Y = DataFrame(rad_.reshape(12*3*8*8*12, 800))
print(Y)

print(len(X), Y.shape)
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


# In[9]:


dataiter = iter(train_loader)
features, radiances = dataiter.next()

print('features shape on PyTroch : ', features.size())
print('radiances shape on PyTroch : ', radiances.size())


# In[10]:


# neural network model

# In[11]:


model = MLP()
print(model)

#import weight_init
#weight_init.weight_init(model)


# In[12]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
#loss_fn = msre()


# In[13]:


mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []

epochs = 5

_epoch_list = []

statefiles = []
for (path, dir, files) in os.walk('./states_02/'):
    
    for filename in files:
        basename = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[-1]
        if basename[0:19] == '02_rtm_nn_normalize' and ext == '.pth':
            _epoch = filename[26:31]
            print(_epoch)
            _epoch_list.append(_epoch)
            statefiles.append(filename)

if len(statefiles) >= 1:
    torchstatefile = './states_01_' + lat + toz + '/'  + sorted(statefiles)[-1]
    print('model before load')
    print(model)
    print(torchstatefile)
    model.load_state_dict(torch.load(torchstatefile))
    print('model after load')
    print(model)
    #quit()
    
    real_epoch = int(sorted(_epoch_list)[-1]) + 1
    print('real_epoch = ', real_epoch)
else:
    real_epoch = 0

model.eval()

with torch.no_grad():
    for i, (f_plot, r_plot) in enumerate(test_loader):
        timestamp = time.time()
        outputs = model(f_plot)
        #print(outputs, len(outputs))
        #print(r_plot, len(r_plot))
        loss = loss_fn(outputs, r_plot)
        print(time.time() - timestamp, loss)
        test_plot(real_epoch, i, f_plot, wav, r_plot, outputs, loss)


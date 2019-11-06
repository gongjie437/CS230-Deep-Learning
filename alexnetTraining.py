# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:05:19 2019
some of the codes are from open source cifar 10 model 
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
@author: jgong
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
print(os.listdir("./processedData"))
import warnings
warnings.filterwarnings("ignore")

#set to enable GPU device 0 on the desktop
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

#%% read the data and split it into training and testing
df=pd.read_pickle("./processedData/processedPatternData.pkl")
df.info()

numClass = 8 #from 0 to 7

df.head()
df.tail()

mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7}

mapping_type_inv={0:'Center',1:'Donut',2:'Edge-Loc',3:'Edge-Ring', 4:'Loc',5:'Random',6:'Scratch',7:'Near-full'}

mapping_traintest={'Training':0,'Test':1}
#df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

tol_wafers = df.shape[0]  #there are 811457 wafer points
imgW = 42
imgH = 42
nChannel = 1 #gray scale image
allImg = np.zeros([tol_wafers, imgW, imgW, nChannel])
allLabel = np.zeros([tol_wafers, 1])

for i in range(tol_wafers):
    allImg[i,:,:,0] = df.waferMap[i]
    allLabel[i] = df.failureNum[i]

#data will be shuffled during this spliting     
X_train, X_test, Y_train, Y_test = train_test_split(allImg, allLabel, test_size=0.3, random_state=42)

numTrain = X_train.shape[0]
numTest = X_test.shape[0]
onehotencoder = OneHotEncoder(categorical_features = [0]) 
Y_train_hot = onehotencoder.fit_transform(Y_train).toarray() 
Y_test_hot = onehotencoder.fit_transform(Y_test).toarray() 

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_train OneHotEncoder shape: " + str(Y_train_hot.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
print ("Y_test OneHotEncoder shape: " + str(Y_test_hot.shape))


#%% now let check whether it is correct by plotting
#fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(10, 10))
#ax = ax.ravel(order='C')
#
#imgCheck =  np.zeros([numClass, imgW, imgW])
#imgIdx = np.zeros([numClass,1])
#for i in range(numClass):
#    itemindex = np.where(Y_train==i)    
#    index = itemindex[0][0]
#    imgIdx[i] = index
#    img = X_train[index,:,:]
#    imgCheck[i,:,:] = img
#    ax[i].imshow(img)
#    ax[i].set_title(mapping_type_inv[i])
#plt.tight_layout()
#plt.show() 

#%% start to training
from CNN_utility import model
_, _, parameters = model(X_train, Y_train_hot, X_test, Y_test_hot)
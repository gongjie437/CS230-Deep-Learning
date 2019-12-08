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
#import ConfusionMatrix
#from ConfusionMatrix import *
from ConfusionMatrix import ConfusionMatrix
from CNN_utility import plot_confusion_matrix
np.random.seed(1)

#%% read the training and testing data, 
X_train_3D = np.load('X_train_after_aug.npy')
Y_train_3D = np.load('Y_train_after_aug.npy')

X_test_3D = np.load('X_test_before_aug.npy')
Y_test_3D = np.load('Y_test_before_aug.npy')


numClass = 8 #from 0 to 7

mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7}

mapping_type_inv={0:'Center',1:'Donut',2:'Edge-Loc',3:'Edge-Ring', 4:'Loc',5:'Random',6:'Scratch',7:'Near-full'}

mapping_traintest={'Training':0,'Test':1}
#df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

numTrain = X_train_3D.shape[0]
numTest = X_test_3D.shape[0]


tol_wafers = numTrain + numTest  #there are 811457 wafer points
imgW = 42
imgH = 42
nChannel = 1 #gray scale image
X_train = np.zeros([numTrain, imgH, imgW, nChannel])
Y_train = np.zeros([numTrain, 1])
for i in range(numTrain):
    X_train[i,:,:,0] = X_train_3D[i,:,:]
    Y_train[i] = Y_train_3D[i]

X_test = np.zeros([numTest, imgH, imgW, nChannel])
Y_test = np.zeros([numTest, 1])
for i in range(numTest):
    X_test[i,:,:,0] = X_test_3D[i,:,:]
    Y_test[i] = Y_test_3D[i]

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
testSplitFlag = 0
if testSplitFlag:
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(10, 10))
    ax = ax.ravel(order='C')
    
    imgCheck =  np.zeros([numClass, imgW, imgW])
    imgIdx = np.zeros([numClass,1])
    for i in range(numClass):
        itemindex = np.where(Y_train==i)    
        index = itemindex[0][0]
        imgIdx[i] = index
        img = X_train[index,:,:]
        imgCheck[i,:,:] = img.reshape(imgW, imgW)
        ax[i].imshow(imgCheck[i,:,:])
        ax[i].set_title(mapping_type_inv[i])
    plt.tight_layout()
    plt.show() 
    fig.savefig('random_traninging2.png')
    
    
    #do the same thing for testing to verify that train_test_split() is reproducible
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(10, 10))
    ax = ax.ravel(order='C')
    
    imgCheck =  np.zeros([numClass, imgW, imgW])
    imgIdx = np.zeros([numClass,1])
    for i in range(numClass):
        itemindex = np.where(Y_test==i)    
        index = itemindex[0][0]
        imgIdx[i] = index
        img = X_test[index,:,:]
        imgCheck[i,:,:] = img.reshape(imgW, imgW)
        ax[i].imshow(imgCheck[i,:,:])
        ax[i].set_title(mapping_type_inv[i])
    plt.tight_layout()
    plt.show() 
    fig.savefig('random_test2.png')

#%% start to do hyperatmater tunning 
from CNN_utility import model, onehotConvertNP, initialize_parameters
#_, _, train_predict_class, train_correct_prediction, parameters = model(X_train, Y_train_hot, X_test, Y_test_hot)

# when lambd = 0, there is no L2 regularzation 
learning_rate = [0.01,0.005,0.001]
num_epochs = 200
minibatch_size = 128
lambd=[0.1,0.01,0.001]   #lambd is very sensitive to the final accuracy, 
dropout = [1.0, 0.75, 0.5] #keep rate

hyperTuning = np.zeros((len(learning_rate)*len(lambd)*len(dropout),5))   
count = 0       
for lr in learning_rate:
    for ld in lambd:
        for do in dropout:
            hyperTuning[count, 0] = lr
            hyperTuning[count, 1] = ld
            hyperTuning[count, 2] = do
            train_accuracy, test_accuracy, train_predict_class, train_actual_class, train_correct_prediction, test_correct_prediction, \
    test_predict_class, test_actual_class, parameters = model(X_train, Y_train_hot, X_test, Y_test_hot, learning_rate = lr,
          num_epochs = num_epochs, minibatch_size = minibatch_size, lambd=ld, keepRate=do)
            hyperTuning[count, 3] = train_accuracy
            hyperTuning[count, 4] = test_accuracy
            count = count + 1
    
np.save('hyperTuningResult', hyperTuning)
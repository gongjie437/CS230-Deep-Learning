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

#%% start to training
from CNN_utility import model, onehotConvertNP, initialize_parameters
#_, _, train_predict_class, train_correct_prediction, parameters = model(X_train, Y_train_hot, X_test, Y_test_hot)

# when lambd = 0, there is no L2 regularzation 
learning_rate = 0.008
num_epochs = 200
minibatch_size = 128
lambd=0.001 #lambd is very sensitive to the final accuracy, 
dropout = 0.5 #keep rate
          
train_accuracy, test_accuracy, train_predict_class, train_actual_class, train_correct_prediction, test_correct_prediction, \
    test_predict_class, test_actual_class, parameters = model(X_train, Y_train_hot, X_test, Y_test_hot, learning_rate = learning_rate,
          num_epochs = num_epochs, minibatch_size = minibatch_size, lambd=lambd, keepRate=dropout)

#%% ConfusionMatrix for Train
train_actual_class_vec = onehotConvertNP(train_actual_class, train_predict_class, train_correct_prediction, train_accuracy)
labels_output = train_actual_class_vec
preditiction_output = train_predict_class

filePostFix = "learning_rate_%.4f_num_epochs_%d_lambd=%.4f_dropout%.2f" %(learning_rate, num_epochs, lambd, dropout)
trainConfusionMaxtrixFile = 'trainConfusionMaxtrix_%s.csv' %(filePostFix)


#print Confusion Matrix
CMObject = ConfusionMatrix(labels_output, preditiction_output)
CMObject.generate_defect_result_list(labels_output, preditiction_output)
CMObject.generate_confusion_matrix()
CMObject.print_confusion_matrix()
CMObject.write_confusion_matrix_to_file(trainConfusionMaxtrixFile)

np.set_printoptions(precision=2)
class_names = ['Center','Donut', 'Edge-Loc', 'Edge-Ring','Loc','Random', 'Scratch', 'Near-full'] #it is a list

class_names = np.asarray(class_names)

# Plot non-normalized confusion matrix
plot_confusion_matrix(labels_output, preditiction_output, classes=class_names,
                      title='Training Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(labels_output, preditiction_output, classes=class_names, normalize=True,
                      title='Training Normalized confusion matrix')

plt.show()


#ConfusionMatrix for Test
test_actual_class_vec = onehotConvertNP(test_actual_class, test_predict_class, test_correct_prediction, test_accuracy)
labels_output = test_actual_class_vec
preditiction_output = test_predict_class

testConfusionMaxtrixFile = 'testConfusionMaxtrix_%s.csv' %(filePostFix)
#print Confusion Matrix
CMObjectTest = ConfusionMatrix(labels_output, preditiction_output)
#if input_manualcode_list != []:
#    CMObject.label_list = input_manualcode_list
CMObjectTest.generate_defect_result_list(labels_output, preditiction_output)
CMObjectTest.generate_confusion_matrix()
CMObjectTest.print_confusion_matrix()
CMObjectTest.write_confusion_matrix_to_file(testConfusionMaxtrixFile)


#% try the new confusion matrix plot 

np.set_printoptions(precision=2)
class_names = ['Center','Donut', 'Edge-Loc', 'Edge-Ring','Loc','Random', 'Scratch', 'Near-full'] #it is a list

class_names = np.asarray(class_names)

# Plot non-normalized confusion matrix
plot_confusion_matrix(labels_output, preditiction_output, classes=class_names,
                      title='Testing Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(labels_output, preditiction_output, classes=class_names, normalize=True,
                      title='Testing Normalized confusion matrix')

plt.show()

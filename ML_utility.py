# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:51:57 2019

@author: jgong
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os

import skimage
from skimage.transform import rescale, resize, rotate
from skimage.util import img_as_float

#defect density is calucated by counting how many pixel == 2 and divide by the size, this indicates that pixel = 2 is defect, 
#input argument x is the wafer map array 
def cal_den(x):
    return 100*(np.sum(x==2)/np.size(x))  


def imgNormResize(img):
    img_norm = img/2 #now it is float64 now
    image_resized = resize(img_norm, (42,42))
    return image_resized

#Flip array in the left/right direction.
def imgFliplr(img):
    return np.fliplr(img)    

#Flip array in the up/down direction.
def imgFlipud(img):
    return np.flipud(img)    

#image rotation

def imgRotate45(img):
    return rotate(img, 45)

def imgRotate90(img):
    return rotate(img, 90)

def imgRotate135(img):
    return rotate(img, 135)

def imgRotate180(img):
    return rotate(img, 180)

def imgRotate225(img):
    return rotate(img, 225)
    
def imgRotate270(img):
    return rotate(img, 270)
    
def imgRotate315(img):
    return rotate(img, 315)
    

#return the regions intensity, 
def find_regions(x):
    rows=np.size(x,axis=0)
    cols=np.size(x,axis=1)
    ind1=np.arange(0,rows,rows//5)
    ind2=np.arange(0,cols,cols//5)
    
    reg1=x[ind1[0]:ind1[1],:]
    reg3=x[ind1[4]:,:]
    reg4=x[:,ind2[0]:ind2[1]]
    reg2=x[:,ind2[4]:]

    reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
    reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
    reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
    reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
    reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
    reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
    reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
    reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
    reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
    fea_reg_den = []
    fea_reg_den = [cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)]
    return fea_reg_den

def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1


def change_val(img):
    img[img==1] =0  
    return img

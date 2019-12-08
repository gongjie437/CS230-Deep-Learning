# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:30:49 2019
reference: https://www.kaggle.com/ashishpatel26/wm-811k-wafermap
@author: jgong
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
# print(os.listdir("./input2"))
import warnings
warnings.filterwarnings("ignore")


#import helper function 
from ML_utility import cal_den
from ML_utility import find_regions
from ML_utility import find_dim
from ML_utility import change_val, imgNormResize, imgRotate45, imgRotate90, imgRotate135, imgRotate180, imgRotate225, imgRotate270, imgRotate315, imgFliplr, imgFlipud
import skimage
from skimage.transform import rescale, resize, rotate
from skimage.util import img_as_float

#%% Data Selection & Preprocessing
df=pd.read_pickle("./input2/LSWMD.pkl")
df.info()

df.head()
df.tail()


#waferIndex from 1 to 25, the distribution shows that they are not equal, which indicates 
#that some lots are not full, The figure shows that not all lots have perfect 25 wafer maps and it may caused by sensor failure or other unknown problems.

uni_Index=np.unique(df.waferIndex, return_counts=True)
plt.bar(uni_Index[0],uni_Index[1], color='gold', align='center', alpha=0.5)
plt.title(" wafer Index distribution")
plt.xlabel("index #")
plt.ylabel("frequency")
plt.xlim(0,26)
plt.ylim(30000,34000)
plt.show()

#Fortunately, we do not need wafer index feature in our classification so we can just drop the variable.
df = df.drop(['waferIndex'], axis = 1)
#after this, waferIndex column is gone


#We can not get much information from the wafer map column but we can see the die size for each instance is different.
#We create a new variable 'waferMapDim' for wafer map dim checking.

df['waferMapDim']=df.waferMap.apply(find_dim)
#after this, a new column called waferMapDim is created, it covers the wafer map diemnsion, 
# 0: no die
# 1 no defects
# 2 defects

#display five samples
df.sample(5)

max(df.waferMapDim), min(df.waferMapDim)
uni_waferDim=np.unique(df.waferMapDim, return_counts=True)
uni_waferDim[0].shape[0]

##The dimension of wafer map, or saying the image size are not always the same. 
#We noticed that there are 632 different size for our wafer map.

#For this reason, we must do data tranformation (feature extraction) to make input
# the same dim and the method will be introducedin the following section.

#
#Missing value check
#
#Do not be afraid to handle so large dataset. When you look into the dataset you may notice quite number of data are useless due to missing values.
#
#Do missing value check is an important part during data preparing process. Since we only interested in wafer with patterns, we may remove those data without failure type labels.

df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
#after this, two more columns, failureNum and trainTestNum are added, copied from failureType and trianTestLabel column

#create 2 dictionary, and replace failureType and trianTestLabel column with the new 2 columns using number instead of text, 
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

tol_wafers = df.shape[0]  #there are 811457 wafer points


#now only 

df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel.info()
#after this, df_withlabel is a new data frame, which only contains these ones with a valid failureNum, 172950 entries
df_withlabel =df_withlabel.reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern.info()
# only 25519 entries has failureType is not none

df_withpattern = df_withpattern.reset_index()
df_nonpattern = df[(df['failureNum']==8)] # 147431 entries has failureType is none
df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]

#should no pattern (failureType = none) be included in training as well?

from matplotlib import gridspec
fig = plt.figure(figsize=(20, 4.5)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5]) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
no_wafers=[tol_wafers-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

colors = ['silver', 'red', 'blue']
explode = (0.1, 0, 0)  # explode 1st slice
labels = ['no-label','label&pattern','label&non-pattern']
# generate the pie plot to display the distribution of the wafer status, 
ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=100)

uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='gold', align='center', alpha=0.9)

assert (np.sum(uni_pattern[1]) == df_withpattern.shape[0])
ax2.set_title("failure type frequency")
ax2.set_ylabel("% of pattern wafers")
ax2.set_xticklabels(labels2)

plt.show()

#%% now take look some pictures, show the first 100 samples with pattern labeled from our datasets.
fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20))
ax = ax.ravel(order='C')
for i in range(100):
    img = df_withpattern.waferMap[i]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
    ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 

#showing the wafer map by failure patterns
x = [0,1,2,3,4,5,6,7]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

for k in x:
    fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(18, 12))
    ax = ax.ravel(order='C')
    for j in [k]:
        img = df_withpattern.waferMap[df_withpattern.failureType==labels2[j]]
        for i in range(10):
            ax[i].imshow(img[img.index[i]])
            ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
            ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 

#%% try to include none
x = [0,1,2,3,4,5,6,7, 8]
labels3 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full', 'none']

for k in x:
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(18, 12))
    ax = ax.ravel(order='C')
    for j in [k]:
        img = df_withlabel.waferMap[df_withlabel.failureType==labels3[j]]
        for i in range(3):
            ax[i].imshow(img[img.index[i]])
            ax[i].set_title(df_withlabel.failureType[img.index[i]][0][0], fontsize=10)
            ax[i].set_xlabel(df_withlabel.index[img.index[i]], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 

#fig.savefig('plot.png')
 
#check the image size dimentions for 
max(df_withpattern.waferMapDim), min(df_withpattern.waferMapDim)
uni_waferDim_df_withpattern=np.unique(df_withpattern.waferMapDim, return_counts=True)
numWaferDim = uni_waferDim_df_withpattern[0].shape[0]

# there are 329 different size for our wafer map. how to get the distribution, 
# size (25, 27) has the max. number of wafer, 2900
#index 3, 68, 77, 161 and 215 
( (38, 36), (39, 37), (53, 52), (71, 70) )

#plot 3D distribution, 
xy_axis = uni_waferDim_df_withpattern[0]
z_count = uni_waferDim_df_withpattern[1]   

x_axis = np.zeros((numWaferDim, 1))
y_axis = np.zeros((numWaferDim, 1))

for i in range(numWaferDim):
    x_axis[i,0] = xy_axis[i][0]
    y_axis[i,0] = xy_axis[i][1]
    
fig = plt.figure() #create a figure object
#add_subplot is the method of fig, Add an Axes to the figure as part of a subplot arrangement.
# 111 nrows, ncols, index
#pos is a three digit integer, where the first digit is the number of rows, the second the number of columns, and the third the index of the subplot. i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5). Note that all integers must be less than 10 for this form to work

ax = fig.add_subplot(111, projection='3d')
#x =[1,2,3,4,5,6,7,8,9,10]
#y =[5,6,2,3,13,4,1,2,4,8]
#z =[2,3,3,3,5,7,9,11,9,10]

ax.scatter(x_axis, y_axis, z_count, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
#fig.savefig('waferDimDis.png')
#%%From each failure type, we selected the most typical failure pattern for visualization.
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

#ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern.waferMap[x[i]]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=24)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 


#%% decide to sample to [42,42] , 42 is weighted dimension of x and y to minize the upsampling and downsampling error, 


#dimWeight = z./sum(z);
#x_norm = sum(x.*w);
#y_norm = sum(y.*w)

tgtXDim = 42
tgtYDim = 42


idx = 340
img = df_withpattern.waferMap[idx]
#image is unsigned 8 bits, how to do img resize, divided by 2 first? 
img_norm = img/2 #now it is float64 now
#resizing 
image_resized = resize(img_norm, (42,42),)
#                       anti_aliasing=True), this version of skiimage (0.13.1) doesn't support anti_aliasing, 
#but need to know downsampling may have anti_aliasing issue, check whehter resizing change the classification or not

#plot to check
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
ax = axes.ravel(order='C')

#ax = axes.ravel()


ax[0].imshow(img_norm)
#ax[0].imshow(img_norm, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(image_resized)
ax[1].set_title("Resized (downsampling) image")
plt.tight_layout()
plt.show()
fig.savefig('downsampling_plot.png')

#%% image resizing is done now data augmentation, 

#image flip, wafer rotation, 

idx = 340
img = df_withpattern.waferMap[idx]
#image is unsigned 8 bits, how to do img resize, divided by 2 first? 
img_norm = img/2 #now it is float64 now
#resizing 
image_resized = resize(img_norm, (42,42))

image_resized_flip = np.flip(image_resized, 0)
image_resized_flip = np.flip(image_resized, 1) #Reverse the order of elements in an array along the given axis
image_resized_flip = np.fliplr(image_resized) #Flip array in the left/right direction.
image_resized_flip = np.flipud(image_resized) #Flip array in the up/down direction.

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
ax = axes.ravel(order='C')

#ax = axes.ravel()

ax[0].imshow(image_resized, cmap='gray')
ax[0].set_title("Original resized image")

ax[1].imshow(image_resized_flip, cmap='gray')
ax[1].set_title("flipped image with flip with  up/down direction")
plt.tight_layout()
plt.show()

#%% try image rotation, if rotation is performed on original image, 
idx = 37
img = df_withpattern.waferMap[idx]
img_norm = img/2 #now it is float64 now
image_resized = resize(img_norm, (42,42))

angle = [45,90,135,180,270]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
ax = axes.ravel(order='C')

#ax = axes.ravel()

ax[0].imshow(image_resized, cmap='gray')
ax[0].set_title("resized image for Donut")

for i in range(len(angle)):
    rotated = rotate(image_resized, angle[i], center=None)
    ax[i+1].imshow(rotated, cmap='gray')
    ax[i+1].set_title("rotated image with theta = %d" % (angle[i]))
plt.tight_layout()
plt.show()

#%% now we should merge all the data, and save it to a new pickle file so that we can use anytime, 

#1. normalize and also transfer it to float and then resize
df_withpattern['waferMapResize'] = df_withpattern.waferMap.apply(imgNormResize)

# this is to display everything
#df_withpattern.info

# display the heading
df_withpattern.info()
img = df_withpattern.waferMapResize[0]
assert(img.shape == (42,42))


#%% 2. augumentation for several classes, and add the new row for the data frame, 
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
#based on histogram, different failure type needs different augmentaion, 

#first let's try nearfull first, but eventually, we should include every type that needed this augumentation and perform the calculation once
df_nearfull = df_withpattern[(df_withpattern['failureNum']==7)]
df_nearfull = df_nearfull.reset_index()
df_nearfull.info()
df_nearfull.head()


df_nearfull_fliplr = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_fliplr['waferMap'] = df_nearfull.waferMapResize.apply(imgFliplr)
# df_nearfull_fliplr['failureNum'] = 7

df_nearfull_fliplr.info()
df_nearfull_fliplr.head()

df_nearfull_flipud = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_flipud['waferMap'] = df_nearfull.waferMapResize.apply(imgFlipud)

#rotation, 
theta = np.arange(45,360,45)
#df_nearfull_rotationAll = pd.DataFrame(None, columns = ['waferMap','failureNum'])

df_nearfull_rotation_45 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_rotation_90 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_rotation_135 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_rotation_180 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_rotation_225 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_rotation_270 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_nearfull_rotation_315 = pd.DataFrame(None, columns = ['waferMap','failureNum'])


df_nearfull_rotation_45['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate45)
df_nearfull_rotation_90['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate90)
df_nearfull_rotation_135['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate135)
df_nearfull_rotation_180['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate180)
df_nearfull_rotation_225['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate225)
df_nearfull_rotation_270['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate270)
df_nearfull_rotation_315['waferMap'] = df_nearfull.waferMapResize.apply(imgRotate315)

df_nearfull_augmentation = pd.concat([df_nearfull_fliplr, df_nearfull_flipud, df_nearfull_rotation_45, \
                            df_nearfull_rotation_90, df_nearfull_rotation_135, df_nearfull_rotation_180, \
                            df_nearfull_rotation_225, df_nearfull_rotation_270, df_nearfull_rotation_315],axis=0,ignore_index=True)


df_nearfull_augmentation['failureNum'] = 7

#nearfull samples from 149 to 149*9

df_nearfull_augmentation.info()
df_nearfull_augmentation.head()
df_nearfull_augmentation.tail()
num_nearfull_augmentation = df_nearfull_augmentation.shape[0]


#%% working on Donut in the same way, 
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
#based on histogram, different failure type needs different augmentaion, 

#first let's try nearfull first, but eventually, we should include every type that needed this augumentation and perform the calculation once
df_donut = df_withpattern[(df_withpattern['failureNum']==1)]
df_donut = df_donut.reset_index()
num_donut = df_donut.shape[0]
#there are 555 samples, applied two flips and 90, 180 and 270 rotation, 555 * 5 

df_donut_fliplr = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_donut_fliplr['waferMap'] = df_donut.waferMapResize.apply(imgFliplr)
# df_nearfull_fliplr['failureNum'] = 7

#df_nearfull_fliplr.info()
#df_nearfull_fliplr.head()

df_donut_flipud = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_donut_flipud['waferMap'] = df_donut.waferMapResize.apply(imgFlipud)


df_donut_rotation_90 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_donut_rotation_180 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_donut_rotation_270 = pd.DataFrame(None, columns = ['waferMap','failureNum'])

df_donut_rotation_90['waferMap'] = df_donut.waferMapResize.apply(imgRotate90)
df_donut_rotation_180['waferMap'] = df_donut.waferMapResize.apply(imgRotate180)
df_donut_rotation_270['waferMap'] = df_donut.waferMapResize.apply(imgRotate270)

df_donut_augmentation = pd.concat([df_donut_fliplr, df_donut_flipud, df_donut_rotation_90, \
                            df_donut_rotation_180, df_donut_rotation_270],axis=0,ignore_index=True)

    
df_donut_augmentation['failureNum'] = 1
df_donut_augmentation.info()
df_donut_augmentation.head()
df_donut_augmentation.tail()
num_donut_augmentation = df_donut_augmentation.shape[0]


#%% try to plot the images to see whether augmentation is done correctly
labels_donut = ['original image','left/right flip','up/down flip','theta = 90','theta = 180','theta = 270']
numLabels_donut = len(labels_donut)
#ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
nrows = 2
ncols = 3
fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10, 10))
ax = ax.ravel(order='C')

startIdx = 0;
img = df_donut.waferMapResize[startIdx]
#ax[0].imshow(img, cmap='gray')
ax[0].imshow(img)
ax[0].set_title(labels_donut[0])

img = df_donut_fliplr.waferMap[startIdx]
ax[1].imshow(img)
ax[1].set_title(labels_donut[1])

img = df_donut_flipud.waferMap[startIdx]
ax[2].imshow(img)
ax[2].set_title(labels_donut[2])

img = df_donut_rotation_90.waferMap[startIdx]
ax[3].imshow(img)
ax[3].set_title(labels_donut[3])

img = df_donut_rotation_180.waferMap[startIdx]
ax[4].imshow(img)
ax[4].set_title(labels_donut[4])

img = df_donut_rotation_270.waferMap[startIdx]
ax[5].imshow(img)
ax[5].set_title(labels_donut[5])

#for i in range(1, nrows*ncols):
#    img = df_donut_augmentation.waferMap[startIdx + (i-1)*numLabels_donut]
#    ax[i].imshow(img)
#    ax[i].set_title(labels_donut[i])
#    ax[i].set_xticks([])
#    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 

#%% working on random in the same way, 
df_random = df_withpattern[(df_withpattern['failureNum']==5)]
df_random = df_random.reset_index()
num_random = df_random.shape[0]
#there are 555 samples, applied two flips and 90, 180 and 270 rotation, 555 * 5 

df_random_fliplr = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_random_fliplr['waferMap'] = df_random.waferMapResize.apply(imgFliplr)
# df_nearfull_fliplr['failureNum'] = 7

#df_nearfull_fliplr.info()
#df_nearfull_fliplr.head()

df_random_flipud = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_random_flipud['waferMap'] = df_random.waferMapResize.apply(imgFlipud)


df_random_rotation_90 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_random_rotation_180 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_random_rotation_270 = pd.DataFrame(None, columns = ['waferMap','failureNum'])

df_random_rotation_90['waferMap'] = df_random.waferMapResize.apply(imgRotate90)
df_random_rotation_180['waferMap'] = df_random.waferMapResize.apply(imgRotate180)
df_random_rotation_270['waferMap'] = df_random.waferMapResize.apply(imgRotate270)

df_random_augmentation = pd.concat([df_random_fliplr, df_random_flipud, df_random_rotation_90, \
                            df_random_rotation_180, df_random_rotation_270],axis=0,ignore_index=True)


df_random_augmentation['failureNum'] = 5
df_random_augmentation.info()
df_random_augmentation.head()
df_random_augmentation.tail()
num_random_augmentation = df_random_augmentation.shape[0]

#%% working on Scratch in the same way, 
df_scratch = df_withpattern[(df_withpattern['failureNum']==6)]
df_scratch = df_scratch.reset_index()
num_scratch = df_scratch.shape[0]
#there are 555 samples, applied two flips and 90, 180 and 270 rotation, 555 * 5 

df_scratch_fliplr = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_scratch_fliplr['waferMap'] = df_scratch.waferMapResize.apply(imgFliplr)
# df_nearfull_fliplr['failureNum'] = 7

#df_nearfull_fliplr.info()
#df_nearfull_fliplr.head()

df_scratch_flipud = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_scratch_flipud['waferMap'] = df_scratch.waferMapResize.apply(imgFlipud)


df_scratch_rotation_90 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_scratch_rotation_180 = pd.DataFrame(None, columns = ['waferMap','failureNum'])
df_scratch_rotation_270 = pd.DataFrame(None, columns = ['waferMap','failureNum'])

df_scratch_rotation_90['waferMap'] = df_scratch.waferMapResize.apply(imgRotate90)
df_scratch_rotation_180['waferMap'] = df_scratch.waferMapResize.apply(imgRotate180)
df_scratch_rotation_270['waferMap'] = df_scratch.waferMapResize.apply(imgRotate270)

df_scratch_augmentation = pd.concat([df_scratch_fliplr, df_scratch_flipud, df_scratch_rotation_90, \
                            df_scratch_rotation_180, df_scratch_rotation_270],axis=0,ignore_index=True)

df_scratch_augmentation['failureNum'] = 6
df_scratch_augmentation.info()
df_scratch_augmentation.head()
df_scratch_augmentation.tail()
num_scratch_augmentation = df_scratch_augmentation.shape[0]

#%% after data augmentation, need to replot the histogram, 
uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
uni_pattern_augmentation = list(uni_pattern[1])
uni_pattern_augmentation[1] = uni_pattern_augmentation[1] + num_donut_augmentation
uni_pattern_augmentation[5] = uni_pattern_augmentation[5] + num_random_augmentation
uni_pattern_augmentation[6] = uni_pattern_augmentation[6] + num_scratch_augmentation
uni_pattern_augmentation[7] = uni_pattern_augmentation[7] + num_nearfull_augmentation

nrows = 2
ncols = 1
fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10, 10))
ax = ax.ravel(order='C')


ax[0].bar(uni_pattern[0],uni_pattern[1], color='green', align='center', alpha=0.9)
assert (np.sum(uni_pattern[1]) == df_withpattern.shape[0])
ax[0].set_title("failure type frequency before data augmentation")
ax[0].set_ylabel("# of pattern wafers")
ax[0].set_xticklabels(labels2)


ax[1].bar(uni_pattern[0],uni_pattern_augmentation, color='green', align='center', alpha=0.9)
ax[1].set_title("failure type frequency after data augmentation")
ax[1].set_ylabel("# of pattern wafers")
ax[1].set_xticklabels(labels2)

plt.show()

#%% adding some none to the sample?

#%% 3. save it, to reduce data size, only need to save new image and data number

df_augmentation = pd.concat([df_scratch_augmentation, df_random_augmentation, df_nearfull_augmentation, \
                            df_donut_augmentation],axis=0,ignore_index=True)
    
df_augmentation.info()
df_augmentation.head()
df_augmentation.tail()
num_augmentation = df_augmentation.shape[0]

#add a column to indicate it is from augmentation
df_augmentation['augmentationLabel'] = 1
df_augmentation.info()
df_augmentation.head()
df_augmentation.tail()

#extract non-augmentation data, 
df_raw = pd.DataFrame(None, columns = ['waferMap','failureNum','augmentationLabel'])

df_raw['waferMap'] = df_withpattern['waferMapResize']
df_raw['failureNum'] = df_withpattern['failureNum']
df_raw['augmentationLabel'] = 0
df_raw.info()
df_raw.head()
df_raw.tail()
#%% need to randomly plot some to take a look whether it is correct for df_raw


#df_output = pd.DataFrame(None, columns = ['waferMap','failureNum','augmentationLabel'])
df_output = pd.concat([df_raw, df_augmentation],axis=0,ignore_index=True)

df_output.info()
df_output.head()
df_output.tail()
num_df_output = df_output.shape[0]
assert(num_df_output == sum(uni_pattern_augmentation))
#connect all the dots together, and do we need to shuffle the rows

df_output.to_pickle("./processedPatternData.pkl")
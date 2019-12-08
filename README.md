# CS230-Deep-Learning
course project for Stanford CS230 Deep Learning

original data and problem statement can be downloaded from https://www.kaggle.com/ashishpatel26/wm-811k-wafermap

In this work, we implemented simplified AlexNet and simplified VGG16 CNNs for automating wafer map defect pattern classification with high training and testing performance. This dataset is its highly imbalanced data distribution among the eight failure pattern types. We applied two data augmentation strategies to solve this data imbalance problem, namely flipping and rotating. 

As for the data split, two different approaches were used. Approach 1 is first performing data augmentation on the normalized wafer maps and then dividing the resulting data into the training and test datasets based on a 7:3 ratio. Approach 2 is first splitting the normalized wafer maps into the training and test datasets according to a 7:3 ratio and then applying data augmentation only on the training dataset. Compared with Approach 1, Approach 2 has the advantage of excluding the data augmentation effect from the test performance results, resulting in potentially better evaluation of the model performance on the test dataset.

After downloading the data, you can run pre_processing_Approach1.py and pre_processing_Approach2.py seperately to generate two different data set. The data set is generated either as a pickle file or npy files. Therefore, you don't need to run pre-processing again and again. 

then run alexnetTraining_Approach1.py and alexnetTraining_Approach2.py to perform a simplified AlexNet training and testing.  
then run VGGTraining_Approach1.py and VGGTraining_Approach2.py to perform a VGG16 training and testing.  
The results will be displayed as the form of confusion matrix. It will be printed out on the console and dumped into CSV files. 

CNN_utility.py, ConfusionMatrix.py, ML_utility.py and vgg2_reduced include helper funtions for both processing and deep learning. 

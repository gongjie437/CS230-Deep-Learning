# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:22:32 2019

@author: jgong
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
import csv
import os
import re
import glob
import matplotlib.pyplot as plt


#%% second implmentation 
class ConfusionMatrix:
    #default constructor of the ConfusionMatrix class
    def __init__(self, labels, predictions):
        #defect_dict_list, labels and predictions: private member of the ConfusionMatrix class
        self.defect_dict_list = []
        self.labels = labels
        self.predictions = predictions
        #form a set to get the different types of the labels
        label_set = set()
        for label in labels:
            label_set.add(label)
        #label_list, another member of class
        self.label_list = []
        for label in label_set:
            self.label_list.append(label)
        #sort the label_list
        self.label_list.sort()

    def get_label_list(self):
        return self.label_list

    #generate a defect_dict_list containing a dict. with original_label and predicted_label
    def generate_defect_result_list(self, labels, prediction):
        num_defects = len(labels)
        for n in range(num_defects):
            defect_dict = {'original_label': labels[n],
                           'predicted_label': prediction[n]
                          }
            self.defect_dict_list.append(defect_dict)

    def generate_confusion_matrix(self):
        self.confusion_matrix = []
        #confusion_matrix size should be num_label*num_label, initilize it
        for n in self.label_list: 
            for m in self.label_list:
                dict = {'original_label': n,
                        'predict_label': m,
                        'manual_code':0,
                        'predict_manual_code': 0,
                        'group_code': 0,
                        'predict_group_code': 0,
                        'num_def': 0}
                self.confusion_matrix.append(dict)

        num_defects = len(self.labels)
        num_label = len(self.label_list)
        for n in range(num_defects):
            original_label = self.labels[n]
            predict_label = self.predictions[n]
            #increase the counter num_def for the element in the confusion_matrix, index = predict_label*num_label+original_label
            self.confusion_matrix[predict_label*num_label+original_label]['num_def'] = self.confusion_matrix[predict_label*num_label+original_label]['num_def']+1

        #calculate accuracy
        self.accuracy = []
        self.defect_count_label_dict = Counter(label for label in self.labels)
        overall_correct_count = 0;
        for label in self.label_list:
            correct_count = self.confusion_matrix[label*num_label+label]['num_def']
            total_count = self.defect_count_label_dict[label]
            overall_correct_count = overall_correct_count + correct_count
            if total_count == 0:
                accuracy = 0.0
            else:
                accuracy = 100.0*float(correct_count)/float(total_count)
            dict = {'label': label,
                    'accuracy': accuracy}
            self.accuracy.append(dict)
        self.overall_accuracy = 100*float(overall_correct_count)/float(num_defects)

        #calculate purity
        self.purity = []
        for predict_label in self.label_list:
            correct_count = self.confusion_matrix[predict_label*num_label+predict_label]['num_def']
            total_count = 0
            for label in range(num_label):
                total_count = total_count + self.confusion_matrix[predict_label*num_label+label]['num_def']
            if total_count > 0:
                purity = 100.0*float(correct_count)/float(total_count)
            else:
                purity = 0
            dict = {'label': predict_label,
                    'purity': purity}
            self.purity.append(dict)


    def print_confusion_matrix(self):
        num_label = len(self.label_list)
        print("              ", end=' ')
        for n in range(num_label):
            print("  Class",n, end=' ')
        print("     Purity")
        for m in range(num_label):
            print("PredictedClass",m, end = '    ')
            for n in range(num_label):
                print(self.confusion_matrix[m*num_label+n]['num_def'], end='        ')
            print("%.2f"%self.purity[m]['purity'])
        print("Accuracy           ", end=' ')
        for m in range(num_label):
            print("%.2f"%self.accuracy[m]['accuracy'], end='      ')
        print()
        print('Overall accruacy:', "%.2f"%self.overall_accuracy)


    def write_confusion_matrix_to_file(self, filename):
        with open(filename, 'w') as cm_file:
            num_label = len(self.label_list)
            cm_file.write("               ,")
            for n in range(num_label):
                cm_file.write("  Class %d," % n)
            cm_file.write("     Purity" + '\n')
            for m in range(num_label):
                cm_file.write("PredictedClass%d,     " % m)
                for n in range(num_label):
                    cm_file.write("%d        ,"%self.confusion_matrix[m*num_label+n]['num_def'])
                cm_file.write("%.2f"%self.purity[m]['purity']+'\n')
            cm_file.write("Accuracy            ,")
            for m in range(num_label):
                cm_file.write("%.2f      ,"%self.accuracy[m]['accuracy'])
            cm_file.write('\n')
            cm_file.write("Overall accruacy, %.2f" % self.overall_accuracy)

    def write_confusion_matrix_to_file_V2(self, filename):
        with open(filename, 'w') as cm_file:
            num_label = len(self.label_list)
            cm_file.write("               ,")
            for n in range(num_label):
                cm_file.write("  Class %d," % n)
            cm_file.write("     Purity" + '\n')
            for m in range(num_label):
                cm_file.write("PredictedClass%d,     " % m)
                for n in range(num_label):
                    cm_file.write("%d        ,"%self.confusion_matrix[m*num_label+n]['num_def'])
                cm_file.write("%.2f"%self.purity[m]['purity']+'\n')
            cm_file.write("Accuracy            ,")
            for m in range(num_label):
                cm_file.write("%.2f      ,"%self.accuracy[m]['accuracy'])
            cm_file.write('\n')
            cm_file.write("Overall accruacy, %.2f" % self.overall_accuracy)

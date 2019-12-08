# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:33:55 2019

@author: jgong
https://blog.csdn.net/zhangwei15hh/article/details/78417789

"""


import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from CNN_utility import create_placeholders, compute_cost,random_mini_batches
from tensorflow.python.framework import ops


use_fp16 = False
numClass = 8
imgW = 42
imgH = 42
mapping_type_inv={0:'Center',1:'Donut',2:'Edge-Loc',3:'Edge-Ring', 4:'Loc',5:'Random',6:'Scratch',7:'Near-full'}
#directory for tensorboard
LOGDIR = 'E:/personal/cs230/project/log/vgg16Wafermap/'

def conv_op(input_op,name,kernelh,kernelw,n_out,stridesHeight,stridesWidth,p):
#def conv_op(input_op,name,kernelh,kernelw,n_out,stridesHeight,stridesWidth,p, seed=1):
    input_op = tf.convert_to_tensor(input_op)
#    input_op = tf.dtypes.cast(input_op, tf.float32)
    # , the last dimension of the tensor is the number of filter input
    n_in = input_op.get_shape()[-1].value 
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",shape=[kernelh,kernelw,n_in,n_out],\
                                 dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_op, kernel, (1,stridesHeight,stridesWidth,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val , trainable=True , name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases] #update parameter list
        return activation

#def fc_op(input_op, name, n_out, seed=1):
def fc_op(input_op, name, n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                shape = [n_in, n_out],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.0, shape = [n_out], dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope) 
        p += [kernel,biases]
        return activation

def mpool_op(input_op, name, kernelh, kernelw, stridesHeight, stridesWidth):
    return  tf.nn.max_pool(input_op,
                           ksize = [1, kernelh, kernelw, 1],
                           strides = [1, stridesHeight, stridesWidth, 1],
                           padding = 'SAME',
                           name = name)
    
#assume input is    42*42*1 
def inference_op(input_op, keep_prob):
    p = []
    # block 1 -- outputs 21x21x64
    conv1_1 = conv_op(input_op, name="conv1_1", kernelh=3, kernelw=3, n_out=8, stridesHeight=1, stridesWidth=1, p=p)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kernelh=3, kernelw=3, n_out=8, stridesHeight=1, stridesWidth=1, p=p)
    pool1 = mpool_op(conv1_2,   name="pool1",   kernelh=2, kernelw=2, stridesHeight=2, stridesWidth=2)
 
    # block 2 -- outputs 11x11x128, since SAME padding in the max pooling 
    conv2_1 = conv_op(pool1,    name="conv2_1", kernelh=3, kernelw=3, n_out=16, stridesHeight=1, stridesWidth=1, p=p)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kernelh=3, kernelw=3, n_out=16, stridesHeight=1, stridesWidth=1, p=p)
    pool2 = mpool_op(conv2_2,   name="pool2",   kernelh=2, kernelw=2, stridesHeight=2, stridesWidth=2)
 
    # # block 3 -- outputs 6x6x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kernelh=3, kernelw=3, n_out=32, stridesHeight=1, stridesWidth=1, p=p)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kernelh=3, kernelw=3, n_out=32, stridesHeight=1, stridesWidth=1, p=p)
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kernelh=3, kernelw=3, n_out=32, stridesHeight=1, stridesWidth=1, p=p)    
    pool3 = mpool_op(conv3_3,   name="pool3",   kernelh=2, kernelw=2, stridesHeight=2, stridesWidth=2)
 
    # block 4 -- outputs 3x3x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kernelh=3, kernelw=3, n_out=64, stridesHeight=1, stridesWidth=1, p=p)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kernelh=3, kernelw=3, n_out=64, stridesHeight=1, stridesWidth=1, p=p)
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kernelh=3, kernelw=3, n_out=64, stridesHeight=1, stridesWidth=1, p=p)
    pool4 = mpool_op(conv4_3,   name="pool4",   kernelh=2, kernelw=2, stridesHeight=2, stridesWidth=2)
 
    # block 5 -- outputs 2x2x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kernelh=3, kernelw=3, n_out=128, stridesHeight=1, stridesWidth=1, p=p)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kernelh=3, kernelw=3, n_out=128, stridesHeight=1, stridesWidth=1, p=p)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kernelh=3, kernelw=3, n_out=128, stridesHeight=1, stridesWidth=1, p=p)
    pool5 = mpool_op(conv5_3,   name="pool5",   kernelh=2, kernelw=2, stridesHeight=2, stridesWidth=2)
 
    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value #first dimension is number of samples
    print("flattened_shape is" + str(flattened_shape))
    assert(flattened_shape == 128*2*2)
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="reshape1")
 
    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=256, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_dropout")
 
    fc7 = fc_op(fc6_drop, name="fc7", n_out=256, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_dropout")
 
    logits = fc_op(fc7_drop, name="fc8", n_out=numClass, p=p)
    return logits

#to avoid OOM on GPU running evluation of accuray, no shuffel needed
def miniBatchForEval(imgData,labelY, miniBatchSize = 128):
    m = imgData.shape[0]                  # number of training examples
    print ("number of examples: " + str(m))
    mini_batches = []
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/miniBatchSize) # number of mini batches of size mini_batch_size in your partitionning
    num_minibatches = num_complete_minibatches
    for k in range(0, num_complete_minibatches):
        #mini_batch_X = shuffled_X[:, (k) * mini_batch_size : (k+1) * mini_batch_size]
        #mini_batch_Y = shuffled_Y[:, (k) * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X = imgData[(k) * miniBatchSize : (k+1) * miniBatchSize, :,:,:]
        mini_batch_Y = labelY[(k) * miniBatchSize : (k+1) * miniBatchSize, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
#        print ("mini_batch_X shape, mini_batch: " + str(mini_batch_X.shape))
#        print ("mini_batch_Y shape, mini_batch: " + str(mini_batch_Y.shape))
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % miniBatchSize != 0:
#        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
#        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        num_minibatches += 1 #increment the counter
        mini_batch_X = imgData[num_complete_minibatches * miniBatchSize : m, :,:,:]
        mini_batch_Y = labelY[num_complete_minibatches * miniBatchSize : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return num_minibatches, mini_batches


def convertListTo2DNp(train_actual_class_list):
    train_actual_class = []
    chkFlag = True
    for item in train_actual_class_list:
        if chkFlag:
            train_actual_class = item
            chkFlag = False
        else:
            train_actual_class = np.vstack((train_actual_class, item))
    return train_actual_class
        
def convertListTo2DNp_V2(train_actual_class_list):
    train_actual_class = []
    chkFlag = True
    for item in train_actual_class_list:
        item = item.reshape(item.size, 1)
        if chkFlag:
            train_actual_class = item
            chkFlag = False
        else:
            train_actual_class = np.vstack((train_actual_class, item))
    return train_actual_class

        
def VGG16(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 128, lambd=0.1, keepRate=1.0, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]  
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("n_y, number of class: " + str(n_y))                          
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    logits = inference_op(X,keep_prob)
    
    cost = compute_cost(logits, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss=cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
        # Do the training loop
    with tf.Session() as sess:
    
    # Run the initialization
        sess.run(init)
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        for epoch in range(num_epochs):
    
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, epoch, minibatch_size, seed)
            #print("# of mini batches: " +str(num_minibatches))
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
    
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run(
                                                fetches=[optimizer, cost],
                                                feed_dict={X: minibatch_X,
                                                           Y: minibatch_Y,
                                                           keep_prob: keepRate}
                                                )                
                minibatch_cost += temp_cost / num_minibatches
                
    
            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        writer.close()
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        #fig.savefig('training_cost.png')


        # Calculate the correct predictions
        predict_op = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        print(accuracy)
        #tensor.eval() returns a Numpy array
        train_predict_class = []
        train_actual_class = []
        num_minibatches_train, trainMiniBatchForEval = miniBatchForEval(X_train,Y_train, miniBatchSize = 256)
        for minibatch_train in trainMiniBatchForEval:
            (minibatch_X_train, minibatch_Y_train) = minibatch_train            
#            test_correct_prediction += correct_prediction.eval({X: minibatch_X_test, Y: minibatch_Y_test, keep_prob: 1.0})
            train_predict_class_cur = predict_op.eval({X: minibatch_X_train, Y: minibatch_Y_train, keep_prob: 1.0})
            train_predict_class.append(train_predict_class_cur)
            train_actual_class_cur = Y.eval({X: minibatch_X_train, Y: minibatch_Y_train, keep_prob: 1.0})
            train_actual_class.append(train_actual_class_cur)
            #test_accuracy = accuracy.eval({X: minibatch_X_test, Y: minibatch_Y_test, keep_prob: 1.0})
            

        train_accuracy = 0.5
        train_correct_prediction = 0
        test_predict_class = []
        test_actual_class = []
        test_accuracy = 0.5
        test_correct_prediction = 1
        num_minibatches_test, testMiniBatchForEval = miniBatchForEval(X_test,Y_test, miniBatchSize = 256)
        for minibatch_test in testMiniBatchForEval:
            (minibatch_X_test, minibatch_Y_test) = minibatch_test            
#            test_correct_prediction += correct_prediction.eval({X: minibatch_X_test, Y: minibatch_Y_test, keep_prob: 1.0})
            test_predict_class_cur = predict_op.eval({X: minibatch_X_test, Y: minibatch_Y_test, keep_prob: 1.0})
            test_predict_class.append(test_predict_class_cur)
            test_actual_class_cur = Y.eval({X: minibatch_X_test, Y: minibatch_Y_test, keep_prob: 1.0})
            test_actual_class.append(test_actual_class_cur)
            #test_accuracy = accuracy.eval({X: minibatch_X_test, Y: minibatch_Y_test, keep_prob: 1.0})
            
#        print("Train Accuracy:", train_accuracy)
#        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, train_predict_class, train_actual_class, train_correct_prediction, test_correct_prediction, \
    test_predict_class, test_actual_class
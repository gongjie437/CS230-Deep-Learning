# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:25:35 2019

@author: jgong
"""

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


use_fp16 = False
numClass = 8
imgW = 42
imgH = 42
mapping_type_inv={0:'Center',1:'Donut',2:'Edge-Loc',3:'Edge-Ring', 4:'Loc',5:'Random',6:'Scratch',7:'Near-full'}
#directory for tensorboard
LOGDIR = 'E:/personal/cs230/project/log/alexWafermap/'


#%% first implmentation 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(title + '.png')
    return ax

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    """

    X = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    keep_prob = tf.placeholder(tf.float32)
    return X, Y, keep_prob


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:

    Normally, functions should take values as inputs rather than hard coding.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1) #fixed randomation
    W1 = tf.get_variable("W1", [3, 3, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    
    b1 = tf.get_variable("b1", [8], initializer = tf.constant_initializer(0.0))
    b2 = tf.get_variable("b2", [16], initializer = tf.constant_initializer(0.0))
    b3 = tf.get_variable("b3", [32], initializer = tf.constant_initializer(0.0))
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "b1": b1,
                  "b2": b2,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters, keep_prob = 1.0):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']    
    print(str(W1))
    # conv 1, CONV2D: stride of 1, padding 'valid'
    conv1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'VALID')
    Z1 = tf.nn.bias_add(conv1, b1)
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 2x2, stride 2, padding 'VALID'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 

    
    #conv 2, CONV2D: stride of 1, padding 'valid'
    conv2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'VALID')
    Z2 = tf.nn.bias_add(conv2, b2)
    # RELU
    A2 = tf.nn.relu(Z2)  
    
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    
    #conv 3, CONV2D: filters W3, stride 1, padding 'VALID'
    conv3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'VALID')
    Z3 = tf.nn.bias_add(conv3, b3)
    # RELU
    A3 = tf.nn.relu(Z3)
    
    A3_dropout = tf.nn.dropout(A3, keep_prob)
    
    # Flattens the input while maintaining the batch_size, Assumes that the first dimension represents the batch.
    F = tf.contrib.layers.flatten(A3_dropout)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(F, numClass, activation_fn=None)
    ### END CODE HERE ###

    return Z3

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  
  #tf.device(): A context manager that specifies the default device to use for newly created ops.
  #All operations constructed in this context will be placed on CPU 0.
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:  #when weight decay is zero, no regularzation, only for weights, bias, no weight decay is used. 
      #Returns x * y element-wise.
      # tf.nn.l2_loss(var), l2 loss, Computes half the L2 norm of a tensor without the sqrt:
      # is this regularlization with l2 loss☻? 
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    #add weight_decay to the losses collection
    tf.add_to_collection('losses', weight_decay)
  return var


#add L2 regularization
#def compute_cost_with_L2_regularization(Z3, Y, parameters, lambd=0.1):
def compute_cost(Z3, Y):
  
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
   """
    
    # Cost function: Add cost function to tensorflow graph
#    W1 = parameters['W1']
#    W2 = parameters['W2']
#    W3 = parameters['W3']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))        
#    L2_regularization_cost = lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*m) 
#    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def random_mini_batches(X, Y, epoch = 0, mini_batch_size = 64, seed = 0):
#def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    #print ("number of training examples: " + str(m))
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
#    Randomly permute a sequence, or return a permuted range.
    permutation = list(np.random.permutation(m))
    
    shuffled_X = X[permutation,:,:,:]
    
    shuffled_Y = Y[permutation,:]
    #print ("shuffled_X shape, mini_batch: " + str(shuffled_X.shape))
    #print ("shuffled_Y shape, mini_batch: " + str(shuffled_Y.shape))

    #plot the first 8 images from permutation
    testRandomFlag = False    
    if testRandomFlag:
        fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(10, 10))
        ax = ax.ravel(order='C')
        for i in range(8):
            img = shuffled_X[i,:,:,:]
            labelY = np.argmax(shuffled_Y[i,:])
            imgCheck = img.reshape(imgW, imgW)
            ax[i].imshow(imgCheck)
            ax[i].set_title(mapping_type_inv[labelY])
        plt.tight_layout()
        plt.show() 
        figFilename = "./Imgfolder/first8Imgs_mini_batches%depoch%d.png" %(seed,epoch)
        fig.savefig(figFilename)
        
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        #mini_batch_X = shuffled_X[:, (k) * mini_batch_size : (k+1) * mini_batch_size]
        #mini_batch_Y = shuffled_Y[:, (k) * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X = shuffled_X[(k) * mini_batch_size : (k+1) * mini_batch_size, :,:,:]
        mini_batch_Y = shuffled_Y[(k) * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        #print ("mini_batch_X shape, mini_batch: " + str(mini_batch_X.shape))
        #print ("mini_batch_Y shape, mini_batch: " + str(mini_batch_Y.shape))
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
#        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
#        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def onehotConvertNP(train_actual_class, train_predict_class, train_correct_prediction, train_accuracy):
    #need to convert the X_test and Y_test_hot to standard numpy array for confusion matrix printting
    train_actual_class_vec = np.argmax(train_actual_class,axis=1)
    train_correct_prediction_vec = np.equal(train_actual_class_vec, train_predict_class)
    #check whether they are correct or not
    f = (train_correct_prediction_vec == train_correct_prediction)
#    assert(f.all())
    train_accuracy1 = np.sum(train_correct_prediction_vec)/train_correct_prediction.size
    print(str(train_accuracy1))
#    assert(abs(train_accuracy1-train_accuracy) < 0.01)
    #print(train_accuracy)
    #print(train_accuracy1)
    return train_actual_class_vec

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 128, lambd=0.1, keepRate=1.0, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
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

    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, keep_prob)
    
    # Cost function: Add cost function to tensorflow graph
    #cross_entropy_cost = compute_cost(Z3, Y)
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']    
    #output = sum(t ** 2) / 2, but this misses the fully connection layer weight
    w1Loss = tf.nn.l2_loss(W1)
    w2Loss = tf.nn.l2_loss(W2)
    w3Loss = tf.nn.l2_loss(W3)
#    weightLoss = lambd*(w1Loss + w2Loss + w3Loss)/minibatch_size
    weightLoss = lambd*(w1Loss + w2Loss + w3Loss)
    #vars   = tf.trainable_variables() 
    #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) 
    cost = compute_cost(Z3, Y) + weightLoss
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss=cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    printWeightIni = False #this is to control whether you want to print out the initilize number or not
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
#        writer = tf.summary.FileWriter('/path/to/logs', tf.get_default_graph())
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)
#        writer.add_graph(sess.graph)
        if printWeightIni:
            print ("W1 is: ")
            print(sess.run(W1))
            print ("W2 is: ")
            print(sess.run(W2))
            print ("W3 is: ")
            print(sess.run(W3))
            print ("b1 is: ")
            print(sess.run(b1))
            print ("b2 is: ")
            print(sess.run(b2))
            print ("b3 is: ")
            print(sess.run(b3))
        # Do the training loop
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
                """
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost.
                # The feedict should contain a minibatch for (X,Y).
                """
                _ , temp_cost = sess.run(
                                                fetches=[optimizer, cost],
                                                feed_dict={X: minibatch_X,
                                                           Y: minibatch_Y,
                                                           keep_prob: keepRate}
                                                )                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        writer.close()
        # plot the cost
        fig = plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        #fig.savefig('training_cost.png')


        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        #tensor.eval() returns a Numpy array
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1.0})
        train_correct_prediction = correct_prediction.eval({X: X_train, Y: Y_train, keep_prob: 1.0})
        train_predict_class = predict_op.eval({X: X_train, Y: Y_train, keep_prob: 1.0})
        train_actual_class = Y.eval({X: X_train, Y: Y_train, keep_prob: 1.0})
        
        test_correct_prediction = correct_prediction.eval({X: X_test, Y: Y_test, keep_prob: 1.0})
        test_predict_class = predict_op.eval({X: X_test, Y: Y_test, keep_prob: 1.0})
        test_actual_class = Y.eval({X: X_test, Y: Y_test, keep_prob: 1.0})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1.0})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, train_predict_class, train_actual_class, train_correct_prediction, test_correct_prediction, \
    test_predict_class, test_actual_class, parameters
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


use_fp16 = False
numClass = 8

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    """

    X = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    
    return X, Y


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:

    Normally, functions should take values as inputs rather than hard coding.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    W1 = tf.get_variable("W1", [3, 3, 1, 8], initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer())
    
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


def forward_propagation(X, parameters):
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
    
    # Flattens the input while maintaining the batch_size, Assumes that the first dimension represents the batch.
    F = tf.contrib.layers.flatten(A3)
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



def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
   """
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))    
    return cost


#def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    #np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    print ("number of training examples: " + str(m))
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
#    Randomly permute a sequence, or return a permuted range.
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    
    shuffled_Y = Y[permutation,:]
    print ("shuffled_X shape, mini_batch: " + str(shuffled_X.shape))
    print ("shuffled_Y shape, mini_batch: " + str(shuffled_Y.shape))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        #mini_batch_X = shuffled_X[:, (k) * mini_batch_size : (k+1) * mini_batch_size]
        #mini_batch_Y = shuffled_Y[:, (k) * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X = shuffled_X[(k) * mini_batch_size : (k+1) * mini_batch_size, :,:,:]
        mini_batch_Y = shuffled_Y[(k) * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        print ("mini_batch_X shape, mini_batch: " + str(mini_batch_X.shape))
        print ("mini_batch_Y shape, mini_batch: " + str(mini_batch_Y.shape))
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

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 128, print_cost = True):
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
    #tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    #seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]  
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("n_y, number of class: " + str(n_y))                          
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss=cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            #seed = seed + 1
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            print("# of mini batches: " +str(num_minibatches))
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
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
                                                           Y: minibatch_Y}
                                                )                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters
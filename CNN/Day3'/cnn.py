# Basic Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Loading the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)

# Convolution Function

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


# Max Pooling Function
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


# Initializing the weights and the bias

def init_weights(shape):
    init_random_wt = tf.truncated_normal(shape, stddev=0.1)
    return init_random_wt

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# Creating Helper functions for convolutional Layer

def convolutional_layer(input_x, size):
    W = init_weights(size)
    b = init_bias(size)
    return tf.nn.relu(conv2d(input_x, W, b, strides=1))

# Helper Function for a normal fully connected Layer

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W)+b

# Creating the Placeholders

x = tf.placeholder(tf.float32, shape=[None, 784])

y_true = tf.placeholder(tf.float32, shape=[None, 10])
# Converting into rank 4 vector
x_image = tf.reshape(x, [-1, 28 , 28, 1])

# Creating a Model

## shape - > [kernel_size, kernel_size, input_channels, output_channels]
conv_1 = convolutional_layer(x_image, size=[5,5,1,32])
conv_1_pool = maxpool2d(conv_1)

conv_2 = convolutional_layer(conv_1, size=[5,5,32,64])
conv_2_pool = maxpool2d(conv_2)

conv_2_flat = tf.reshape(conv_2_pool, [-1, 7*7*64])

full_layer = tf.nn.relu(normal_full_layer(conv_2_flat, 1024))

# Creating dropout

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer, keep_prob=hold_prob)

# Output 

y_pred = normal_full_layer(full_one_dropout, 10)

# Loss Function 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

# Run the session 
steps = 1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})
        print("Accuracy :: ")
        matches = tf.equal(tf.argmax(y_true, 1), tf.softmax(y_pred, 1))
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))
        print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}))




### Keras implementation of Resnet ###

from __future__ import print_function
from keras.layers import Activation, Input, Flatten, BatchNormalization, Conv2D, AveragePooling2D, Dense
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.datasets import cifar10
import numpy as np
import os





# Loading the dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Extrating input image shape 
input_shape = x_train.shape[1:]

# Normalizing the data 
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255



# Learning rate scheduler

def lr_schedule(epoch):

    # Define an initial lr
    lr = 0.001
    if(epoch>200):
        lr *=0.5e-3
    elif(epoch>160):
        lr*=1e-3
    elif(epoch>120):
        lr*=1e-2
    elif(epoch>80):
        lr*=1e-1
    print("Learning Rate changed to : ", lr)
    return lr

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    # Create a dummy convolution instance
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer='he_normal', kernel_regularizer=12(1e-4))
    x = inputs
    # Check if convolution has to be applied before batch normalization
    if(conv_first):
        # Apply convolution first and then Batch Normalize
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        # Batch Normalize first and then apply convolution
        if batch_normalization:
            x = BatchNormalization(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet(input_shape, depth, num_classes=10):

    ## The Depth must be of the order 6n+2 ####
    if(depth-2)%6!=0:
        raise ValueError('Depth should be of the form 6n+2\n')
    
    # Setting the number of filters in the first stack
    num_filters=16
    # Calculating number of residual blocks
    num_res_block = int((depth-2)/6)
    inputs = Input(shape=input_shape)
    # Stacking up residual blocks in a layer
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # First layer but not the first stack
            if(stack>0) and res_block==0:
                # Down Sample
                strides=2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if(stack>0) and res_block==0:
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)

        num_filters *=2
    # Average Pooling the stacked layers output
    x = AveragePooling2D(pool_size=8)(x)
    # Flattening the output to apply softmax
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernal_initializer='he_normal')
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = resnet(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy', optimizer='', metrics=['accuracy'])
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience-=5, min_lr=0.5e-6)
callbacks = [lr_reducer, lr_scheduler]

if not data_augmentation:
    print("Not using data augmentation \n")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)
else:
    print('Using Data Augmentation')
    datagen = ImageDataGenerator()


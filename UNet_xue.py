#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:23:17 2017

@author: zhenggangxue
"""

# In[1]:


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, UpSampling2D, Cropping2D, ZeroPadding2D 
from keras.layers import Conv2DTranspose
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras.optimizers import SGD, Adam, Adagrad
from keras.layers.merge import concatenate
from keras import backend as K
import tensorflow as tf
import cv2

from sklearn.metrics import precision_score, recall_score

import data_preproc

K.image_data_format() == 'channels_last'

# In[ ]:
# # 2 Architecture of U-net

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#def precision(y_true, y_pred):
##    Only computes a batch-wise average of precision.
#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#    precision = true_positives / (predicted_positives + K.epsilon())
#    return precision
#
# 
#def recall(y_true, y_pred):
#    """Recall metric.
#
#    Only computes a batch-wise average of recall.
#
#    Computes the recall, a metric for multi-label classification of
#    how many relevant items are selected.
#    """
#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#    recall = true_positives / (possible_positives + K.epsilon())
#    return recall

# In[]

def get_crop_shape(target, refer):
    # height, the 1 dimension
    print(K.get_variable_shape(target))
    print(K.get_variable_shape(refer))

    ch = (K.get_variable_shape(target)[1] - K.get_variable_shape(refer)[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)
        
    cw = (K.get_variable_shape(target)[2] - K.get_variable_shape(refer)[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    
    return (ch1, ch2), (cw1, cw2)
    
def get_pad_shape(target, refer):
    ch = (K.get_variable_shape(refer)[1] - K.get_variable_shape(target)[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)
        
    cw = (K.get_variable_shape(refer)[2] - K.get_variable_shape(target)[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    
    return (ch1, ch2), (cw1, cw2)
    
# In[]
        
#    unet with flexible input shape. e.g. (None, 136, 256 <, 2>), (None, 240, 240 <, 2>), (.. 256, 256 ..)
def unet(input_shape=(240,240,2),bn=True, do=0, ki="he_normal",lr=0.001):
    '''
    bn: if use batchnorm layer
    do: dropout prob
    ki: kernel initializer (glorot_uniform, he_normal, ...)
    lr: learning rate of Adam
    '''
    concat_axis = -1 #the last axis (channel axis)
    
    inputs = Input(input_shape) # channels is 2: <t1, flair>
    
    conv1 = Conv2D(64, (5,5), padding="same", activation="relu", kernel_initializer=ki)(inputs)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    conv1 = Dropout(do)(conv1) if do else conv1
    conv1 = Conv2D(64, (5,5), padding="same", activation="relu", kernel_initializer=ki)(conv1)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(96, (3,3), padding="same", activation="relu", kernel_initializer=ki)(pool1)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    conv2 = Dropout(do)(conv2) if do else conv2
    conv2 = Conv2D(96, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv2)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer=ki)(pool2)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    conv3 = Dropout(do)(conv3) if do else conv3
    conv3 = Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv3)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = Conv2D(256, (3,3), padding="same", activation="relu", kernel_initializer=ki)(pool3)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    conv4 = Dropout(do)(conv4) if do else conv4
    conv4 = Conv2D(256, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv4)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = Conv2D(512, (3,3), padding="same", activation="relu", kernel_initializer=ki)(pool4)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    conv5 = Dropout(do)(conv5) if do else conv5
    conv5 = Conv2D(512, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv5)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    upconv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv5)    

    ch, cw = get_crop_shape(conv4, upconv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    cat6 = concatenate([upconv5, crop_conv4], axis=concat_axis)
    
    conv6 = Conv2D(256, (3,3), padding="same", activation="relu", kernel_initializer=ki)(cat6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    conv6 = Dropout(do)(conv6) if do else conv6
    conv6 = Conv2D(256, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    upconv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv6)
    
    ch, cw = get_crop_shape(conv3, upconv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([upconv6, crop_conv3], axis=concat_axis)
    
    conv7 = Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer=ki)(up7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    conv7 = Dropout(do)(conv7) if do else conv7
    conv7 = Conv2D(128, (3,3), padding="same", activation="relu")(conv7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    upconv7 = Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv7)
    
    ch, cw = get_crop_shape(conv2, upconv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([upconv7, crop_conv2], axis=concat_axis)
    
    conv8 = Conv2D(96, (3,3), padding="same", activation="relu", kernel_initializer=ki)(up8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    conv8 = Dropout(do)(conv8) if do else conv8
    conv8 = Conv2D(96, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    upconv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv8)
    
    ch, cw = get_crop_shape(conv1, upconv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([upconv8, crop_conv1], axis=concat_axis)
    
    conv9 = Conv2D(64, (3,3), padding="same", activation="relu", kernel_initializer=ki)(up9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(64, (3,3), padding="same", activation="relu", kernel_initializer=ki)(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    ch,cw = get_pad_shape(conv9, conv1)
    pad_conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv9 = Conv2D(1, (1,1), padding="same", activation="sigmoid", kernel_initializer=ki)(pad_conv9) #change to sigmoid
    
    model = Model(inputs=inputs, outputs=conv9)
    

#    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)   #default
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    optimizer = Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
#    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    
    optimizer = SGD(lr=0.1, momentum=0.8, decay=lr/30, nesterov=False)
    
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef, 'accuracy'])

    return model
    
# In[]:
def train():
    model = unet(input_shape=(240,240,2),bn=True, do=0, ki="he_normal", lr=0.001)    
    
    model_checkpoint = ModelCheckpoint('weights_240.h5', monitor='val_loss', save_best_only=True)
    
    X_train, Y_train = data_preproc.load_train_data(size=240)
    
    print("Fitting model...")
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", Y_train.shape)

    model.fit(x=X_train, y=Y_train.reshape(Y_train.shape[0],
                                       Y_train.shape[1],
                                       Y_train.shape[2],1),
          batch_size=8, #32 resource exhausted error
          epochs=30,
          verbose=1,
          shuffle=True,
          validation_split=0.1,
          callbacks=[model_checkpoint])

    print("Finish Fitting!")

#    model.load_weights('weights.h5')
#    p_test = model.predict(x_test, batch_size=8,verbose=1)
#
#    print(type(p_test))

def predict_testVersion(input_shape = (240,240,2), weights_path = "weights.h5"):
    model = unet(input_shape=input_shape, bn=True, do=0, ki="he_normal", lr=0.001)
    model.load_weights(weights_path)
    
    X_test, Y_test = data_preproc.load_test_data(size=240)
    print(X_test.shape)
    print(Y_test.shape)
    
    
    # since slices are randomly pick, try to find a slice with significant labels
    slice_index = 0
    for i in range(Y_test.shape[0]):
        if np.sum(Y_test[i, :,:]) > 1800:
            slice_index = i
            break
    
#    result = model.evaluate(X_test, np.expand_dims(Y_test,axis=-1), batch_size=8)
#    print(result)
    re = model.predict(np.expand_dims(X_test[slice_index,:,:,:], axis=0))
    re = np.where(re < 0.5, 0, 1)
    
    y_true = Y_test[slice_index,:,:]
    y_pred = re.reshape( (re.shape[1], re.shape[2]) )
    
    plt.imshow(y_pred)
    plt.show()
    
    plt.imshow(y_true)
    plt.show()

    print("precision:", precision_score(y_true, y_pred, average="micro"))
    print("recall:", recall_score(y_true, y_pred, average="micro"))
    
    
#    plt.imshow(y_pred)

# In[ ]:
if __name__ == "__main__":
    train()
#    predict_testVersion(weights_path = "weights_240_sgd_0.75.h5")




# ### Thoughts
# * Selecting slices for training. The first m slices and last n ones of each patient were removed for a good training set. (the value is empirically 1/8)
# * seperate all sets into train/val/test sets
# * Masking the brain. (How?)
# * Each slice was cropped or padded to 200×200.
# * ? Voxel intensity normalization per patient was performed using Gaussian Normalization during the training and testing stage. 
# * Data augmentation by flipping, rotation, shearing and scaling.
# * 80% patients of each training dataset were combined as training set and the rest 20% patients were used as test set.
# 

# ### Test Part

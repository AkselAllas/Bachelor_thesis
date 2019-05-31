#!/usr/bin/env python
# coding: utf-8

#Import all the dependencies

#This disables python on GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import ndarray
import time
import sys
import matplotlib
import matplotlib.pyplot as plt

from s2_preprocessor import *
from s2_model import *
from plotter import *

version_start = str(sys.argv[1])

#Because fit_generator needs different data preprocessing functions, then we define functions for windowing in this script
def input_windows_preprocessing(preprocessor_X_output, preprocessor_Y_output, s2_preprocessor):
    nb_tile_pixels = s2_preprocessor.tile_dimension*s2_preprocessor.tile_dimension
    dim = (s2_preprocessor.window_dimension,s2_preprocessor.window_dimension,s2_preprocessor.nb_images)
    input_data = preprocessor_X_output.astype('float32')

    input_labels = np.reshape(preprocessor_Y_output,(nb_tile_pixels,s2_preprocessor.nb_classes))

    #Get Region of Interest mask from loaded array
    ROI_mask = input_data[:,:,0,5]
    X_2D_nowindows = input_data[:,:,:,0:5]
    reshaped_ROI_mask = np.reshape(ROI_mask,(nb_tile_pixels))
    valid_pixels_count = np.count_nonzero(reshaped_ROI_mask)
    X = np.zeros((0,s2_preprocessor.nb_bands,*dim))
    Y = np.zeros((0,s2_preprocessor.nb_classes))


    X = np.concatenate((X,np.zeros((valid_pixels_count, s2_preprocessor.nb_bands, *dim))),axis=0)
    Y = np.concatenate((Y,np.zeros((valid_pixels_count, s2_preprocessor.nb_classes))))

    for j in range(s2_preprocessor.nb_images):
        for i in range(s2_preprocessor.nb_bands):

            padded_overpad = skimage.util.pad(X_2D_nowindows[:s2_preprocessor.tile_dimension,:,i,j],4,'reflect')
            padded = padded_overpad[:-1,:-1].copy() #Copy is made so that next view_as_windows wouldn't throw warning about being unable to provide views. Without copy() interestingly enough, it doesn't take extra RAM, just throws warnings.
            windows = skimage.util.view_as_windows(padded,(s2_preprocessor.window_dimension,s2_preprocessor.window_dimension))
            reshaped_windows = np.reshape(windows,(nb_tile_pixels,s2_preprocessor.window_dimension,s2_preprocessor.window_dimension))
            k=0
            l=0
            for mask_element in reshaped_ROI_mask:
                if(mask_element==True):
                    X[k,i,:,:,j] = reshaped_windows[l]
                    Y[k] = input_labels[l]
                    k+=1
                l+=1
    return X,Y


s2_preprocessor_params = {'input_dimension':5120, #5120
    'label_dir':'./Label_tifs/',
    'data_dir':'./Data/',
    'input_data_dir':'./Big_tile_data/',
    'region_of_interest_shapefile':'./ROI/ROI.shp',
    'window_dimension':8,
    'tile_dimension':512,
    'nb_images':5,
    'nb_bands':22,
    'nb_steps':8, #This is unused!! #nb_steps defines how many parts the tile will be split into for training
    'rotation_augmentation':0,
    'flipping_augmentation':0
}

s2_preprocessor = s2_preprocessor(**s2_preprocessor_params)

class_weights = np.load("class_weights.npy")

optimizer_params = {
    'lr':0.001,
    }#'clipvalue':0.5,


#Callback for CTRL+Z to stop training
stop_cb = SignalStopping()

filepath="best_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stopping_params = {
    'monitor':'val_loss',
    'min_delta':0.,
    'patience':5,
    'verbose':1,
    #'mode':'auto'
}

s2_model_params = {
    's2_preprocessor' : s2_preprocessor,
    'batch_size' : 512,
    'nb_epochs' : 1000,
    'nb_filters' : [32, 32, 64],
    'max_pool_size' : [2,2,1],
    'conv_kernel_size' : [3,3,3],
    'optimizer' : SGD(**optimizer_params),
    'loss_function' : 'categorical_crossentropy',
    'metrics' : ['mse', 'accuracy'],
    'version' : '0',
    'cb_list' : [EarlyStopping(**early_stopping_params),stop_cb,checkpoint]
}

s2_model = s2_model(**s2_model_params)

#for layer in s2_model.model.layers:
#    print(layer.get_config())
#    print(np.array(layer.get_weights()).shape)

conv_layer_one_weights = np.array(s2_model.model.layers[0].get_weights()[0])
print(np.array(conv_layer_one_weights[:,:,0,13,0]).shape)
#conv_layer_two_weights = np.array(s2_model.model.layers[4].get_weights()[0])
#conv_layer_three_weights = np.array(s2_model.model.layers[8].get_weights()[0])
#print(s2_model.model.layers[0].get_config())
for i in range(22):
    print("Indeksi "+i+" filtri kaalude abosluutv√√rtuse summa on: "+str(np.sum(np.absolute(conv_layer_one_weights[:,:,:,i,:]))))

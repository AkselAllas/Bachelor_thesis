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
    'version' : version_start,
    'cb_list' : [EarlyStopping(**early_stopping_params),stop_cb,checkpoint]
}

s2_model = s2_model(**s2_model_params)

selected_tile = "1-1" #Put as "xx" or "0-0"
list_of_zero_tiles = []
select_mode = 1
predict_mode = 0
current_tile = [9,3] #NB!! 1st element is tile in y-dimension 2nd element is tile in x-dimension

plotter = plotter(s2_preprocessor, cmap='tab10')

label_map = s2_preprocessor.construct_label_map(selected_tile)
#plotter.plot_labels(labels)
#plotter.plot_tile(label_map_tiles, tile_location)
label_map_tiles = s2_preprocessor.tile_label_map(label_map)

if (os.path.exists('current.h5')):
    s2_model.load("current.h5")

##Setting validation set. Label location can be different. Doesn't work with data augmentation!!! 
#labels_location=[1,3]
#val_data = s2_preprocessor.construct_input_data(labels_location, 0)
#val_labels = s2_preprocessor.construct_labels(label_map_tiles, labels_location, 0)
#val_input_data = val_data.astype('float32')
#del val_data

for a in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):
    for b in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):
        for augmentation_nr in range(s2_preprocessor.nb_augmentations):

            tile_location=[a,b]

            if(tile_location in list_of_zero_tiles):
                continue

            if(select_mode==1):
                if(a!=current_tile[0]):
                    continue
                if(b!=current_tile[1]):
                    continue

            data = s2_preprocessor.construct_input_data(tile_location, selected_tile)

            labels = s2_preprocessor.construct_labels(label_map_tiles, tile_location)
            print('Labels size: '+str(sys.getsizeof(labels)))
            labels_unique = np.unique(labels)
            labels_size = labels.size
            zero_percentage = (labels_size - np.count_nonzero(labels)) / labels_size
            print("Zero percentage is:"+str(zero_percentage))

            #if(zero_percentage>0.9):
            #    list_of_zero_tiles.append(tile_location)
            #    print(list_of_zero_tiles)
            #    continue

            #plotter.plot_tile(label_map_tiles,tile_location)
            #plotter.plot_labels(labels)
            #plotter.plot_labels(val_labels)

            #Use lower accuracy, to use 2x less RAM
            input_data = data.astype('float32')
            del data

            #Convert to one-hot notation matrices
            one_hot_labels = np_utils.to_categorical(labels, num_classes=s2_preprocessor.nb_classes)
            del labels

            X, Y = input_windows_preprocessing(input_data, one_hot_labels, s2_preprocessor)
            #plotter.plot_input_vs_labels_v2(Y,X)

            if(predict_mode==1):
                y_predictions = s2_model.predict(input_data)
                plotter.plot_model_prediction(y_predictions, tile_location, label_map_tiles) 
                input("Press Enter to continue...")

            #Splitting data to train, val sets:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, random_state=4)

            get_activations_1 = K.function([s2_model.model.layers[0].input, K.learning_phase()], [s2_model.model.layers[0].output,])
            activations_1 = get_activations_1([X_train[:3],0])
            ax_1=np.array(activations_1)

            #get_activations_2 = K.function([s2_model.model.layers[4].input, K.learning_phase()], [s2_model.model.layers[4].output,])
            #activations_2 = get_activations_2([X_train[:3],0])
            #ax_2=np.array(activations_2)

            #get_activations_3 = K.function([s2_model.model.layers[8].input, K.learning_phase()], [s2_model.model.layers[8].output,])
            #activations_3 = get_activations_3([X_train[:3],0])
            #ax_3=np.array(activations_3)

            print(ax_1.shape)

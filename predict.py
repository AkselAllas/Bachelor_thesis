#!/usr/bin/env python
# coding: utf-8
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(sess)  # set this TensorFlow session as the default 
from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import SGD, RMSprop, adam
import sys

from plotter import *
from data_generator import *
from s2_preprocessor import * 
from s2_model import * 
from read_processed_tiles import * 

#This disables python on GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  


#V2ga ohtlik, kui panna =3:
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tile_arg = str(sys.argv[2])
name_arg = "_"+str(sys.argv[1])+"_"+tile_arg
version = str(sys.argv[1])

s2_preprocessor = s2_preprocessor(
    input_dimension=5120, #5120
    label_dir='./Label_tifs/',
    data_dir='./Data/',
    input_data_dir='./Input_data/',
    region_of_interest_shapefile='./ROI/ROI.shp',
    window_dimension=8,
    tile_dimension=512,
    nb_images=5,
    nb_bands=22,
    nb_steps=8, #This is unused!! #nb_steps defines how many parts the tile will be split into for training
    rotation_augmentation=0,
    flipping_augmentation=0
)
predictions_path="./Predictions/"

plotter = plotter(s2_preprocessor, cmap='tab10')

params = {'dim': (8,8,5),
          'batch_size': 1, #See tuleb 3*512*512 ??
          'nb_classes': 28,
          'nb_channels': 22,
          'dir_path':'Big_tile_data/',
          'nb_tile_pixels':512*512,
          'tile_dimension':512,
          'shuffle': True}

list_IDs = ['1-1_(9,3)','1-1_(9,4)']
if(tile_arg=="0"):
    list_IDs= ['1-1_(9,3)']
elif(tile_arg=="1"):
    list_IDs=['1-1_(9,4)']
else:
    tile_arg = int(tile_arg)
    list_IDs = read_processed_tiles_512x512()
    list_IDs = [str(list_IDs[tile_arg])]

#print(read_processed_tiles_512x512()[20:])
#print(len(read_processed_tiles_512x512()))
#input("Press Enter to continue...")

#print("Prediction for:")
#print(list_IDs)
#input("Press Enter to continue...")

training_generator = data_generator(list_IDs, **params)
#X_val, Y_val = validation_generator.gene(list_validation[:1])

s2_model = s2_model(
    s2_preprocessor = s2_preprocessor,
    batch_size = 1,
    nb_epochs = 32,
    nb_filters = [32, 32, 64],
    max_pool_size = [2,2,1],
    conv_kernel_size = [3,3,3],
    optimizer = SGD(lr=0.1), #, clipvalue=0.5
    loss_function = 'categorical_crossentropy',
    metrics = ['mse', 'accuracy'],
    version = version,
    cb_list = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto'),
               CSVLogger('training.log') #After running it for the first time, I will add ,append=True to the parameters
              ]
)

X_train, Y_train = training_generator._data_generator__data_generation(list_IDs)
print("Starting_predictions")
y_predictions, unique_classes = s2_model.predict(X_train)
print("Ending_predictions")
#plotter.plot_input_vs_labels_v2(Y_train,X_train)
accuracy, empty_percent = s2_model.get_accuracy_and_empty_percent(y_predictions, Y_train)
print("accuracy: "+str(accuracy))
print("empty_percent: "+str(empty_percent))
np.save(predictions_path+"prediction_stats"+name_arg+".npy",[accuracy, empty_percent, unique_classes])
plotter.save_plot_model_prediction_v2(y_predictions, Y_train, name_arg, predictions_path)

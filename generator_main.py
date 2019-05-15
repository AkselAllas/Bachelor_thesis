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
import time
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

version = str(sys.argv[1])

s2_preprocessor_params = {'input_dimension':5120, #5120
    'label_dir':'./Label_tifs/',
    'data_dir':'./Data/',
    'input_data_dir':'./Input_data/',
    'region_of_interest_shapefile':'./ROI/ROI.shp',
    'window_dimension':8,
    'tile_dimension':128,
    'nb_images':5,
    'nb_bands':22,
    'nb_steps':8, #This is unused!! #nb_steps defines how many parts the tile will be split into for training
    'rotation_augmentation':0,
    'flipping_augmentation':0
}

s2_preprocessor = s2_preprocessor(**s2_preprocessor_params)

plotter = plotter(s2_preprocessor, cmap='tab10')

generator_params = {
    'dim': (8,8,5),
    'batch_size': 3, #See tuleb 3*512*512 ??
    'nb_classes': 28,
    'nb_channels': 22,
    'dir_path':'./Input_data/',
    'nb_tile_pixels':128*128,
    'tile_dimension':128,
    'shuffle': True
}

list_IDs = read_processed_tiles()
list_IDs = list_IDs[4285:]
list_IDs_len = len(list_IDs)
print(list_IDs_len)
input("Press Enter to continue...")
train_val_split_index = int(list_IDs_len*3/4)
print(train_val_split_index)
list_train = list_IDs[:train_val_split_index]
list_validation = list_IDs[train_val_split_index:]

training_generator = data_generator(list_train, **generator_params)
validation_generator = data_generator(list_validation, **generator_params)
#X_val, Y_val = validation_generator.gene(list_validation[:1])

#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


optimizer_params = { 
    'lr':0.001, 
    'clipvalue':0.5
}

early_stopping_params = {
    'monitor':'val_loss',
    'min_delta':0, 
    'patience':2,
    'verbose':0, 
    'mode':'auto'
}

s2_model_params = {
    's2_preprocessor' : s2_preprocessor,
    'batch_size' : 1,
    'nb_epochs' : 32,
    'nb_filters' : [32, 32, 64],
    'max_pool_size' : [2,2,1],
    'conv_kernel_size' : [3,3,3],
    'optimizer' : SGD(**optimizer_params),
    'loss_function' : 'categorical_crossentropy',
    'metrics' : ['mse', 'accuracy'],
    'cb_list' : [EarlyStopping(**early_stopping_params)]
}

s2_model = s2_model(**s2_model_params)

#X_train, Y_train = training_generator.gene(list_train[:2])
#y_predictions = s2_model.predict(X_train)
##plotter.plot_input_vs_labels_v2(Y_train,X_train)
#accuracy, empty_percent = s2_model.get_accuracy_and_empty_percent(y_predictions, Y_train)
#print("accuracy: "+str(accuracy))
#print("empty_percent: "+str(empty_percent))
#plotter.plot_model_prediction_v2(y_predictions, Y_train)

class_weights = np.load("class_weights.npy")
print(class_weights)

fit_params = {
    'workers':4,
    'class_weight':class_weights,
    'max_queue_size':6,
    'epochs':5,
    'steps_per_epoch':100,
    'use_multiprocessing':True,
}

start_time = time.time()
hist = s2_model.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    **fit_params,
)
time_elapsed = time.time() - start_time

s2_model.save("current.h5")

train_loss=hist.history['loss']
epochs_done=len(train_loss)
del s2_model_params['s2_preprocessor']
del s2_model_params['optimizer']
del s2_model_params['cb_list']
metadata_dict = {
    'Epochs_done' : epochs_done,
    'Version': version,
    'Time_elapsed': time_elapsed,
    'Input_data_nb': list_IDs_len,
    'fit_params': fit_params,
    's2_model_params': s2_model_params,
    'optimizer': optimizer_params,
    'early_stopping': early_stopping_params,
    's2_preprocessor_params': s2_preprocessor_params,
    'data_generator_params': generator_params,
}

np.save('Models/hist'+version+'.npy', hist)
np.save('Models/metadata'+version+'.npy', metadata_dict)


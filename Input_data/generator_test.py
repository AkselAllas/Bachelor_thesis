#!/usr/bin/env python
# coding: utf-8
import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import SGD, RMSprop, adam
import sys
sys.path.append("..")

from plotter import *
from data_generator import *
from s2_preprocessor import * 
from s2_model import * 
from read_processed_tiles import * 

s2_preprocessor = s2_preprocessor(
    input_dimension=5120, #5120
    label_dir='../Label_tifs',
    data_dir='../Data',
    input_data_dir='../Input_data',
    region_of_interest_shapefile='../ROI/ROI.shp',
    window_dimension=8,
    tile_dimension=512,
    nb_images=5,
    nb_bands=22,
    nb_steps=8, #This is unused!! #nb_steps defines how many parts the tile will be split into for training
    rotation_augmentation=0,
    flipping_augmentation=0
)

plotter = plotter(s2_preprocessor, cmap='tab10')

params = {'dim': (8,8,5),
          'batch_size': 1, #See tuleb 3*512*512 ??
          'nb_classes': 28,
          'nb_channels': 22,
          'dir_path':'../',
          'nb_tile_pixels':512*512,
          'tile_dimension':512,
          'shuffle': True}

list_IDs = read_processed_tiles()
list_train = list_IDs[0:270]
list_validation = list_IDs[270:]

training_generator = data_generator(list_train, **params)
print(list_train[:2])
training_generator.gene(list_train[:2])
input("Press Enter to continue...")
validation_generator = data_generator(list_validation, **params)

s2_model = s2_model(
    s2_preprocessor = s2_preprocessor,
    batch_size = 512,
    nb_epochs = 512,
    nb_filters = [32, 32, 64],
    max_pool_size = [2,2,1],
    conv_kernel_size = [3,3,3],
    optimizer = SGD(lr=0.001, clipvalue=0.5),
    loss_function = 'categorical_crossentropy',
    metrics = ['mse', 'accuracy'],
    cb_list = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'),
               CSVLogger('training.log') #After running it for the first time, I will add ,append=True to the parameters
              ]
)

s2_model.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1)

s2_model.save("current.h5")

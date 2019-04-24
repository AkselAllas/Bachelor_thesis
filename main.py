#!/usr/bin/env python
# coding: utf-8

#Import all the dependencies
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import ndarray

from s2_preprocessor import *
from s2_model import *
from plotter import *

s2_preprocessor = s2_preprocessor(
    input_dimension=2048,
    label_dir='./Label_tifs',
    data_dir='./Data',
    window_dimension=8,
    tile_dimension=256, 
    nb_images=5, 
    nb_bands=22,
    rotation_augmentation=1, 
    flipping_augmentation=1
) 

class_weights = [0.31779653, 5.42426751, 4.09190809, 2.71449281, 0.65598975, 1.77081251, 1.03404967]

#Create CNN model
s2_model = s2_model(
    s2_preprocessor = s2_preprocessor,
    batch_size = 32,
    nb_epochs = 2,
    nb_filters = [32, 32, 64],
    max_pool_size = [2,2,1],
    conv_kernel_size = [3,3,3],
    class_weights = class_weights,
    optimizer = RMSprop(lr=0.0005),
    loss_function = 'categorical_crossentropy',
    metrics = ['mse', 'accuracy'],
    cb_list = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
)
plotter = plotter(s2_preprocessor, s2_model)

label_map = s2_preprocessor.construct_label_map()
#plotter.plot_labels(labels)
#plotter.plot_tile(label_map_tiles, tile_location)
label_map_tiles = s2_preprocessor.tile_label_map(label_map)

#Siit vaja teha for loop. Koodi muuta, et saaks mudeli laadida.
#saab vast teha S2_model objekti loomise siin juba?
#s2_model.load("current.h5")
for a in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):
    for b in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):
        tile_location=[a,b]
        s2_model.load("current.h5")
        if(a==7):
            continue
        
        data = s2_preprocessor.construct_input_data(tile_location)
        labels = s2_preprocessor.construct_labels(label_map_tiles, tile_location)
        #plotter.plot_tile(label_map_tiles,tile_location)
        labels_unique = np.unique(labels)

        #Use lower accuracy, to use 2x less RAM
        input_data = data.astype('float32')
        del data

        #Set weights for unbalanced classes
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

        #Convert to one-hot notation matrices
        one_hot_labels = np_utils.to_categorical(labels, num_classes=s2_preprocessor.nb_classes)
        del labels

        #Splitting data to train, val sets:
        X_train, X_val, Y_train, Y_val = train_test_split(input_data, one_hot_labels, test_size=0.2, random_state=4)
        del input_data
        del one_hot_labels
        ##X_train, X_val = X_tr[:13107,:,:,:,:], X_tr[13107:,:,:,:,:]
        ##Y_train, Y_val = Y_train[:13107,:], Y_train[13107:,:]

        print(tile_location)

        hist = s2_model.fit(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)

        s2_model.save("current.h5")

#print(hist.history)

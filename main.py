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
    tile_dimension=512, 
    nb_images=5,
    nb_bands=22,
    rotation_augmentation=1, 
    flipping_augmentation=1
) 

class_weights = [0.1, 5.42426751, 4.09190809, 2.71449281, 0.65598975, 1.77081251, 1.03404967]

#Create CNN model
s2_model = s2_model(
    s2_preprocessor = s2_preprocessor,
    batch_size = 512,
    nb_epochs = 512,
    nb_filters = [32, 32, 64],
    max_pool_size = [2,2,1],
    conv_kernel_size = [3,3,3],
    class_weights = class_weights,
    optimizer = SGD(lr=0.00001, clipvalue=0.5),
    loss_function = 'categorical_crossentropy',
    metrics = ['mse', 'accuracy'],
    cb_list = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
)
plotter = plotter(s2_preprocessor, s2_model, cmap='tab10')

label_map = s2_preprocessor.construct_label_map()
#plotter.plot_labels(labels)
#plotter.plot_tile(label_map_tiles, tile_location)
label_map_tiles = s2_preprocessor.tile_label_map(label_map)

if (os.path.exists('current.h5')):
    s2_model.load("current.h5")

##Setting validation set. Label location can be different. Doesn't work with data augmentation!!! 
labels_location=[1,3]
#val_data = s2_preprocessor.construct_input_data(labels_location, 0)
#val_labels = s2_preprocessor.construct_labels(label_map_tiles, labels_location, 0)
#val_input_data = val_data.astype('float32')
#del val_data

#Siit vaja teha for loop. Koodi muuta, et saaks mudeli laadida.
#saab vast teha S2_model objekti loomise siin juba?
#s2_model.load("current.h5")
prev_val_loss = 99999999
for a in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):
    for b in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):
        for augmentation_nr in range(s2_preprocessor.nb_augmentations):

            #if (os.path.exists('current.h5')):
            #    s2_model.load("current.h5")

            tile_location=[a,b]

            #if(b==((int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension)))-1)):
            #    labels_location=[a+1,b]
            #else:
            #    labels_location=[a,b+1]


            #if(not(a==0 or a==5) or (b!=0 and a!=5)):
            #    continue
            #if(not(b==0 or b==6) or (b!=6 and a!=0)):
            #    continue

            #if(a==1):
            #    continue
            #if(b==3):
            #    continue

            if(a!=3):
                continue
            if(b!=0):
                continue

            #if(a==1 and b==3):
            #    continue
            #if(b==3):
            #    continue
            
            data = s2_preprocessor.construct_input_data(tile_location, augmentation_nr)
            labels = s2_preprocessor.construct_labels(label_map_tiles, tile_location, augmentation_nr)
            labels_unique = np.unique(labels)
            labels_size = labels.size
            zero_percentage = (labels_size - np.count_nonzero(labels)) / labels_size
            print("Zero percentage is:"+str(zero_percentage))

            if(zero_percentage>0.9):
                continue

            val_data = s2_preprocessor.construct_input_data(labels_location, augmentation_nr)
            val_labels = s2_preprocessor.construct_labels(label_map_tiles, labels_location, augmentation_nr)

            #plotter.plot_tile(label_map_tiles,tile_location)
            #plotter.plot_labels(labels)

            #Use lower accuracy, to use 2x less RAM
            input_data = data.astype('float32')
            del data
            val_input_data = val_data.astype('float32')
            del val_data

            #Set weights for unbalanced classes
            #class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

            #Convert to one-hot notation matrices
            one_hot_labels = np_utils.to_categorical(labels, num_classes=s2_preprocessor.nb_classes)
            del labels
            val_one_hot_labels = np_utils.to_categorical(val_labels, num_classes=s2_preprocessor.nb_classes)
            del val_labels

            #plotter.plot_input_vs_label(label_map_tiles, tile_location, input_data)

            y_predictions = s2_model.predict(input_data)
            plotter.plot_model_prediction(y_predictions, tile_location, label_map_tiles) 
            input("Press Enter to continue...")

            #Splitting data to train, val sets:
            #X_train, X_val, Y_train, Y_val = train_test_split(input_data, one_hot_labels, test_size=0.5, random_state=4)
            unit = int(s2_preprocessor.nb_tile_pixels/16)
            for step in range(16):
                X_train, X_val, Y_train, Y_val = input_data[step*unit:(step+1)*unit], val_input_data, one_hot_labels[step*unit:(step+1)*unit], val_one_hot_labels 
                
                #del input_data
                #del one_hot_labels
                #del val_input_data
                #del val_one_hot_labels

                ##X_train, X_val = X_tr[:13107,:,:,:,:], X_tr[13107:,:,:,:,:]
                ##Y_train, Y_val = Y_train[:13107,:], Y_train[13107:,:]

                hist = s2_model.fit(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, tile_location=tile_location, augmentation_nr=augmentation_nr, step=step)
                val_loss = hist.history['val_loss'][-1]
                
                if(val_loss < prev_val_loss):
                    s2_model.save("current.h5")
                    prev_val_loss=val_loss
                else:
                    s2_model.load("current.h5")

            #plotter.plot_model_result(hist)
            #print(hist.history)

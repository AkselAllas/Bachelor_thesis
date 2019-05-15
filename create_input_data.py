#!/usr/bin/env python
# coding: utf-8

#Import all the dependencies
import numpy as np
from keras.utils import np_utils

from s2_preprocessor import *
from plotter import *

#Create labels
s2_preprocessor = s2_preprocessor(
    input_dimension=5120, #5120
    label_dir='./Label_tifs',
    data_dir='./Data',
    input_data_dir='./Input_data',
    region_of_interest_shapefile='./ROI/ROI.shp',
    window_dimension=8,
    tile_dimension=128,
    nb_images=5,
    nb_bands=22,
    nb_steps=8, #This is unused!! #nb_steps defines how many parts the tile will be split into for training
    rotation_augmentation=0,
    flipping_augmentation=0
)

plotter = plotter(s2_preprocessor, cmap='tab10')

selected_big_tile_array = ["0-0","0-1","1-0","1-1"] #Notation: 0-1 means 0-th tile in y-axis and 1-st tile in x-axis. Same applies to tile_location. This is how 2D arrays are indexed in python.
if (os.path.exists('zero_tile_dict.npy')):
    np_dict_loaded = np.load('zero_tile_dict.npy')
    zero_tile_dict = np_dict_loaded.item()
    print("Loaded Zero Tile Dictionary:")
    print(zero_tile_dict)
else:
    zero_tile_dict =  {
      "0-0": [], 
      "0-1": [], 
      "1-0": [], 
      "1-1": []
    }


if (os.path.exists('processed_tile_dict.npy')):
    np_dict_loaded_2 = np.load('processed_tile_dict.npy')
    processed_tile_dict = np_dict_loaded_2.item()
    print("Loaded Processed Tile Dictionary:")
    print(processed_tile_dict)
else:
    processed_tile_dict =  {
      "0-0": [], 
      "0-1": [], 
      "1-0": [], 
      "1-1": []
    }

for selected_big_tile in selected_big_tile_array:

    label_map = s2_preprocessor.construct_label_map(selected_big_tile)
    label_map_tiles = s2_preprocessor.tile_label_map(label_map)

    for a in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):   
        for b in range(int(round(s2_preprocessor.input_dimension/s2_preprocessor.tile_dimension))):   

            tile_location=[a,b]
            print(tile_location)
            if(tile_location in zero_tile_dict[selected_big_tile]):
                print(str(tile_location)+' '+selected_big_tile+" skipped - ZERO")
                continue

            if(tile_location in processed_tile_dict[selected_big_tile]):
                print(str(tile_location)+' '+selected_big_tile+" skipped - ALREADY_DONE")
                continue
            
            data = s2_preprocessor.construct_input_data(tile_location, selected_big_tile)
            labels = s2_preprocessor.construct_labels(label_map_tiles, tile_location)

            labels_size = np.count_nonzero(data[:,:,0,s2_preprocessor.nb_images])
            if(labels_size == 0):
                zero_percentage=1
            else:
                zero_percentage = (labels_size - np.count_nonzero(labels)) / labels_size


            if(zero_percentage>0.95):
                zero_tile_dict[selected_big_tile].append(tile_location)
                print(str(tile_location)+' '+selected_big_tile+" added as zero tile")
                np.save('zero_tile_dict.npy', zero_tile_dict)
                continue

            input_data = data.astype('float32')
            del data

            one_hot_labels = np_utils.to_categorical(labels, num_classes=s2_preprocessor.nb_classes)
            del labels

            X_filename = s2_preprocessor.input_data_dir+'/X_'+selected_big_tile+'_'+'('+str(tile_location[0])+','+str(tile_location[1])+')'+'.npy'
            Y_filename = s2_preprocessor.input_data_dir+'/Y_'+selected_big_tile+'_'+'('+str(tile_location[0])+','+str(tile_location[1])+')'+'.npy'
            np.save(X_filename, input_data)
            np.save(Y_filename, one_hot_labels)

            processed_tile_dict[selected_big_tile].append(tile_location)
            print(str(tile_location)+' '+selected_big_tile+" added as processed tile")
            np.save('processed_tile_dict.npy', processed_tile_dict)
            del input_data
            del one_hot_labels
    del label_map
    del label_map_tiles

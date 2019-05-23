#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys

from read_processed_tiles import *

list_IDs = read_processed_tiles()

nb_class_pixels=np.zeros((28)).astype(dtype=np.int64)

for tile_nr, loaded_str in enumerate(list_IDs):
            ID = loaded_str[:-1]

            Y_np_loaded = np.load('Input_data/Y_' + ID + '.npy')
            Y_np_loaded = np.reshape(Y_np_loaded,(128*128,28))
            Y = np.array([np.where(r==1)[0][0] for r in Y_np_loaded])
            unique, counts = np.unique(Y, return_counts=True)
            count_dict = dict(zip(unique, counts))
            for i, count in count_dict.items():
                nb_class_pixels[i]+=count
np.save("class_counts.npy",nb_class_pixels)

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
sys.path.append("..")

from s2_preprocessor import *
from plotter import *

version = str(sys.argv[1])

s2_preprocessor = s2_preprocessor(
    input_dimension=5120, #5120
    label_dir='../Label_tifs',
    data_dir='../Data',
    input_data_dir='../Input_data',
    region_of_interest_shapefile='../ROI/ROI.shp',
    window_dimension=8,
    tile_dimension=128,
    nb_images=5,
    nb_bands=22,
    nb_steps=8, #This is unused!! #nb_steps defines how many parts the tile will be split into for training
    rotation_augmentation=0,
    flipping_augmentation=0
)
plotter = plotter(s2_preprocessor, cmap='tab10')


hist = np.load("hist"+version+".npy").item()

plotter.plot_model_result(hist)

#!/usr/bin/env python
# coding: utf-8

#Import all the dependencies
import georaster
import skimage
from skimage import io
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils, generic_utils
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import ndarray
import skimage as sk
from skimage import transform

from s2_preprocessor import *
from cnn_plotter import *
import gc
import sys

preprocessor = s2_preprocessor(
    input_dimension=2048,
    label_dir='./Label_tifs',
    data_dir='./Data',
    window_dimension=8,
    tile_dimension=256, 
    nb_images=5, 
    nb_bands=22,
    rotation_augmentation=1, 
    flipping_augmentation=0
) 

label_map = preprocessor.construct_label_map()
plotter = cnn_plotter(preprocessor.nb_classes)
#plotter.plot_labels(labels)
tile_location = [5,6]
label_map_tiles = preprocessor.tile_label_map(label_map)
X_tr = preprocessor.construct_training_set(tile_location)
labels = preprocessor.construct_labels(label_map_tiles, tile_location)
plotter.plot_tile(label_map_tiles, tile_location)


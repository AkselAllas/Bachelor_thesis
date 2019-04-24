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

class cnn_plotter:

    def __init__(self, nb_classes):
        self.nb_classes  = nb_classes    

    def plot_labels(self, labels):
        f, arr = plt.subplots()
        img = arr.imshow(labels,vmin=0,vmax=self.nb_classes)
        cig = f.colorbar(img)
        plt.show()
   
    def plot_tile(self, labels_tiles, tile_location): 
        plt.imshow(labels_tiles[tile_location[0]][tile_location[1]])
        plt.show()


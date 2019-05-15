from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras import regularizers
import tensorflow as tf
import os
import numpy as np

class s2_model:

    def __init__(self, s2_preprocessor, batch_size, nb_epochs, nb_filters, max_pool_size, conv_kernel_size, loss_function, optimizer, metrics, cb_list):
        self.s2_preprocessor = s2_preprocessor
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.nb_filters = nb_filters
        self.max_pool_size = max_pool_size
        self.conv_kernel_size = conv_kernel_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.cb_list = cb_list

        model_exists = os.path.exists('current.h5')
        if (model_exists):
            self.model = load_model('current.h5')
            print("**************************************************")
            print("current.h5 model loaded")
        else:
            self.model = Sequential()
            #1st conv
            self.model.add(Conv3D(
                nb_filters[0],
                (conv_kernel_size[0], # depth
                conv_kernel_size[1], # rows
                conv_kernel_size[2]), # cols
                padding = "same",
                data_format = 'channels_first',
                input_shape=(s2_preprocessor.nb_bands, s2_preprocessor.window_dimension, s2_preprocessor.window_dimension, s2_preprocessor.nb_images),

#                input_shape=(s2_preprocessor.window_dimension, s2_preprocessor.window_dimension, s2_preprocessor.nb_images, s2_preprocessor.nb_bands),
                #kernel_regularizer=regularizers.l2(0.01),
                #activity_regularizer=regularizers.l1(0.01)
            ))
            self.model.add(BatchNormalization(axis=1)) #Documentation says to use axis=1 if channels_first format is used
            self.model.add(Activation("relu"))
            #1st pool
            self.model.add(MaxPooling3D(pool_size=(max_pool_size[0], max_pool_size[1], max_pool_size[2]), data_format= 'channels_first'))
            #2nd conv
            self.model.add(Conv3D(
                nb_filters[1],
                (conv_kernel_size[0], # depth
                conv_kernel_size[1], # rows
                conv_kernel_size[2]), # cols
                padding = "same",
                data_format = 'channels_first',
                input_shape=(s2_preprocessor.nb_bands, s2_preprocessor.window_dimension, s2_preprocessor.window_dimension, s2_preprocessor.nb_images),
                #input_shape=(s2_preprocessor.window_dimension, s2_preprocessor.window_dimension, s2_preprocessor.nb_images, s2_preprocessor.nb_bands),
                #kernel_regularizer=regularizers.l2(0.01),
                #activity_regularizer=regularizers.l1(0.01)
            ))
            self.model.add(BatchNormalization(axis=1))
            self.model.add(Activation("relu"))
            #2nd pool
            self.model.add(MaxPooling3D(pool_size=(max_pool_size[0], max_pool_size[1], max_pool_size[2]), data_format= 'channels_first'))
            #3rd conv
            self.model.add(Conv3D(
                nb_filters[2],
                (conv_kernel_size[0], # depth
                conv_kernel_size[1], # rows
                conv_kernel_size[2]), # cols
                padding = "same",
                data_format = 'channels_first',
                input_shape=(s2_preprocessor.nb_bands, s2_preprocessor.window_dimension, s2_preprocessor.window_dimension, s2_preprocessor.nb_images),
                #kernel_regularizer=regularizers.l2(0.01),
                #activity_regularizer=regularizers.l1(0.01)
            ))
            self.model.add(BatchNormalization(axis=1))
            self.model.add(Activation("relu"))
            #3rd pool
            self.model.add(MaxPooling3D(pool_size=(max_pool_size[0], max_pool_size[1], max_pool_size[2]), data_format= 'channels_first'))
            self.model.add(Flatten())
            self.model.add(Dense(64, activation='relu', kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.01),
                #activity_regularizer=regularizers.l1(0.01)
            ))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(s2_preprocessor.nb_classes,kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.01),
                #activity_regularizer=regularizers.l1(0.01)
            ))
            self.model.add(Activation('softmax'))
            self.model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
            print("=========== S2 CNN MODEL COMPILED ===========")

    def fit(self, X_train, Y_train, X_val, Y_val):
        hist = self.model.fit(
            X_train,
            Y_train,
            validation_data=(X_val,Y_val),
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            callbacks=self.cb_list,
            )
        return(hist)

    def save(self, filename):
        print("=========== SAVING MODEL ===========")
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

    def evaluate(self, X_val, Y_val):
        score = self.model.evaluate(
           X_val,
           Y_val,
           batch_size=self.batch_size,
           )
        return(score)

    def predict(self, input_data):
        print("=========== STARTING PREDICTION ===========")
        y_prob = self.model.predict(input_data) 
        y_predictions = y_prob.argmax(axis=-1)
        unique_classes = str(np.unique(y_predictions))
        return(y_predictions, unique_classes)

    def get_accuracy_and_empty_percent(self, y_predictions, true_values):
        count=0
        for i in range(len(y_predictions)):
            if(y_predictions[i]==np.argmax(true_values[i])):
                count+=1

        true_val_non_hot = [np.where(r==1)[0][0] for r in true_values]
        empty_percent = 1 - (np.count_nonzero(true_val_non_hot)/len(y_predictions))
        accuracy = count/(len(y_predictions))
        return(accuracy, empty_percent)

         

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras import regularizers
import keras
import tensorflow as tf
import os
import numpy as np
import signal
import time

class s2_model:

    def __init__(self, s2_preprocessor, batch_size, nb_epochs, nb_filters, max_pool_size, conv_kernel_size, loss_function, optimizer, metrics, cb_list, version):
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
        self.version = version 

        model_exists_version = False
        model_exists_0 = False

        if(version=="0"):
            model_exists_0 = os.path.exists('current.h5')
        else:
            model_exists_version = os.path.exists('Models/'+version+'.h5')

        if (model_exists_0):
            self.model = load_model('current.h5')
            print("**************************************************")
            print("current.h5 model loaded")
        elif (model_exists_version):
            self.model = load_model('Models/'+version+'.h5')
            print("**************************************************")
            print('Models/'+version+'.h5 loaded')
        else:
            print("**************************************************")
            print("Creating own model")
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
            #self.model.add(Dropout(0.5))
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

         

class SignalStopping(keras.callbacks.Callback):
    '''Stop training when an interrupt signal (or other) was received
        # Arguments
        sig: the signal to listen to. Defaults to signal.SIGTSTP.
        doubleSignalExits: Receiving the signal twice exits the python
            process instead of waiting for this epoch to finish.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
    '''
    # SBW 2018.10.15 Since ctrl-c trapping isn't working, watch for existence of file, e.g. .\path\_StopTraining.txt.
    def __init__(self, sig=signal.SIGTSTP, doubleSignalExits=False, verbose=1):
        super(SignalStopping, self).__init__()
        self.signal_received = False
        self.verbose = verbose
        self.doubleSignalExits = doubleSignalExits
        def signal_handler(sig, frame):
            self.model.stop_training = True
            #if self.signal_received and self.doubleSignalExits:
            #    if self.verbose > 0:
            #        print('') #new line to not print on current status bar. Better solution?
            #        print('Received signal to stop ' + str(sig)+' twice. Exiting..')
            #    exit(sig)
            #self.signal_received = True
            #if self.verbose > 0:
            #    print('') #new line to not print on current status bar. Better solution?
            #    print('Received signal to stop: ' + str(sig))
        signal.signal(signal.SIGTSTP, signal_handler)
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.signal_received:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("stop_training=true")

    def on_train_end(self, logs={}):
        print("on_train_end")
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: stopping due to signal' % (self.stopped_epoch)) 

import numpy as np
import keras
import skimage

import matplotlib.pyplot as plt

class data_generator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=2, dim=(8,8,5), nb_channels=22, nb_classes=28, shuffle=True, dir_path="", nb_tile_pixels=512*512, tile_dimension=512):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.shuffle = shuffle
        self.dir_path = dir_path
        self.nb_tile_pixels = nb_tile_pixels
        self.tile_dimension = tile_dimension 
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / (self.batch_size)))

    def __getitem__(self, index):
        #Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels) if 'channels last'
        # X : (n_samples, n_channels, *dim) in case of 'channels first'?

        valid_sample_count = 0
        X = np.zeros((0,self.nb_channels,*self.dim))
        Y = np.zeros((0,self.nb_classes))
        # Generate data
        for tile_nr, ID in enumerate(list_IDs_temp):
            X_np_loaded = np.load(self.dir_path+'X_' + ID + '.npy')
            Y_np_loaded = np.load(self.dir_path+'Y_' + ID + '.npy')
            #Flatten loaded 2D labels
            Y_np_loaded = np.reshape(Y_np_loaded,(self.nb_tile_pixels,self.nb_classes))
            #Get Region of Interest mask from loaded array
            ROI_mask = X_np_loaded[:,:,0,5]
            X_2D_nowindows = X_np_loaded[:,:,:,0:5]
            reshaped_ROI_mask = np.reshape(ROI_mask,(self.nb_tile_pixels))
            new_starting_index = valid_sample_count
            #print(new_starting_index)
            #valid_sample_count = np.count_nonzero(reshaped_ROI_mask)
            valid_pixels_count = np.count_nonzero(reshaped_ROI_mask)
            valid_sample_count += valid_pixels_count
            #print(valid_sample_count)
            X = np.concatenate((X,np.zeros((valid_pixels_count, self.nb_channels, *self.dim))),axis=0)
            Y = np.concatenate((Y,np.zeros((valid_pixels_count, self.nb_classes))))
            for j in range(self.dim[2]):
                for i in range(self.nb_channels):
                    padded_overpad = skimage.util.pad(X_2D_nowindows[:self.tile_dimension,:,i,j],4,'reflect')
                    padded = padded_overpad[:-1,:-1].copy() #Copy is made so that next view_as_windows wouldn't throw warning about being unable to provide views. Without copy() interestingly enough, it doesn't take extra RAM, just throws warnings.
                    windows = skimage.util.view_as_windows(padded,(self.dim[0],self.dim[0]))
                    reshaped_windows = np.reshape(windows,(self.nb_tile_pixels,self.dim[0],self.dim[0]))
                    k=0
                    l=0
                    for mask_element in reshaped_ROI_mask:
                        if(mask_element==True):
                            X[new_starting_index+k,i,:,:,j] = reshaped_windows[l]
                            Y[new_starting_index+k] = Y_np_loaded[l]
                            k+=1
                        l+=1
        print("============== DATA LOADED ===============")
        return X,Y

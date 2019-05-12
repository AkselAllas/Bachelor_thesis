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
        return int(np.floor(len(self.list_IDs) / (self.batch_size*10)))

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
        

    def gene(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels) if 'channels last'
        # X : (n_samples, n_channels, *dim) in case of 'channels first'?

        valid_sample_count = 0
        X = np.zeros((0,22,8,8,5))
        Y = np.zeros((0,self.nb_classes))
        # Generate data
        for tile_nr, loaded_str in enumerate(list_IDs_temp):
            tile_half_nr = int(loaded_str[-1])
            ID = loaded_str[:-1]

            X_np_loaded = np.load(self.dir_path+'Input_data/X_' + ID + '.npy')
            Y_np_loaded = np.load(self.dir_path+'Input_data/Y_' + ID + '.npy')
            if(tile_half_nr==0):
                Y_np_loaded = Y_np_loaded[:int(self.tile_dimension/2),:,:]
            elif(tile_half_nr==1):
                Y_np_loaded = Y_np_loaded[int(self.tile_dimension/2):,:,:]
            #Flatten loaded 2D labels
            Y_np_loaded = np.reshape(Y_np_loaded,(int(self.nb_tile_pixels/2),self.nb_classes))
            #Get Region of Interest mask from loaded array
            ROI_mask = X_np_loaded[:,:,0,5]
            X_2D_nowindows = X_np_loaded[:,:,:,0:5]
            reshaped_ROI_mask = np.reshape(ROI_mask,(self.nb_tile_pixels))
            new_starting_index = valid_sample_count
            #print(new_starting_index)
            #valid_sample_count = np.count_nonzero(reshaped_ROI_mask)
            tile_half_index = int(self.nb_tile_pixels/2)
            if(tile_half_nr==1 and tile_half_index%2==1):
                tile_half_index = tile_half_index + 1

            if(tile_half_nr==0):
                half_reshaped_ROI_mask = reshaped_ROI_mask[:tile_half_index]
            elif(tile_half_nr==1):
                half_reshaped_ROI_mask = reshaped_ROI_mask[tile_half_index:]
            half_tile_valid_pixels_count = np.count_nonzero(half_reshaped_ROI_mask)
            valid_sample_count += half_tile_valid_pixels_count
            #print(valid_sample_count)
            X = np.concatenate((X,np.zeros((half_tile_valid_pixels_count, self.nb_channels, *self.dim))),axis=0)
            Y = np.concatenate((Y,np.zeros((half_tile_valid_pixels_count, self.nb_classes))))
            for j in range(self.dim[2]):
                for i in range(self.nb_channels):
                    if(tile_half_nr==0):
                        padded_overpad = skimage.util.pad(X_2D_nowindows[:int(self.tile_dimension/2),:,i,j],4,'reflect')
                    elif(tile_half_nr==1):
                        padded_overpad = skimage.util.pad(X_2D_nowindows[int(self.tile_dimension/2):,:,i,j],4,'reflect')
                    padded = padded_overpad[:-1,:-1].copy() #Copy is made so that next view_as_windows wouldn't throw warning about being unable to provide views. Without copy() interestingly enough, it doesn't take extra RAM, just throws warnings.
                    windows = skimage.util.view_as_windows(padded,(self.dim[0],self.dim[0]))
                    reshaped_windows = np.reshape(windows,(int(self.nb_tile_pixels/2),self.dim[0],self.dim[0]))

                    k=0
                    l=0
                    for mask_element in half_reshaped_ROI_mask:
                        if(mask_element==True):
                            X[new_starting_index+k,i,:,:,j] = reshaped_windows[l]
                            Y[new_starting_index+k] = Y_np_loaded[l]
                            k+=1
                        l+=1
            #print(X.shape)
            #print(Y.shape)

        input_map  = np.zeros((512,512))
        m=0
        for i in range(512):
            for j in range(512):
                    if(reshaped_ROI_mask[512*i+j]==True):
                        input_map[i][j] = X[m][4][4][0][0]
                        m+=1

        labels_map  = np.zeros((512,512))
        m=0
        for i in range(512):
            for j in range(512):
                    if(reshaped_ROI_mask[512*i+j]==True):
                        labels_map[i][j] = np.argmax(Y[m])
                        m+=1
        f, axarr = plt.subplots(2,2)
        axarr[0][0].imshow(input_map)
        axarr[0][1].imshow(labels_map)
        plt.show()
        print("============== DATA LOADED ===============")
        return X,Y

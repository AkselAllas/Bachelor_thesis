import georaster
import skimage
import os
import numpy as np
import sys
import skimage as sk


class s2_preprocessor:

    #Rotation_augmenatation and flipping_augmentation should be 1 or 0
    def __init__(self, input_dimension, label_dir, data_dir, window_dimension, tile_dimension, nb_images, nb_bands, rotation_augmentation, flipping_augmentation):
        self.input_dimension = input_dimension
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.window_dimension = window_dimension
        self.tile_dimension = tile_dimension
        self.nb_tile_pixels= tile_dimension*tile_dimension
        self.nb_images = nb_images
        self.nb_bands = nb_bands
        self.rotation_augmentation = rotation_augmentation
        self.flipping_augmentation = flipping_augmentation
        self.nb_augmentations = (1+rotation_augmentation*3)*(1+flipping_augmentation)
        self.nb_samples = self.nb_tile_pixels
        listing = os.listdir(self.label_dir)
        self.nb_classes = len(listing) #-1 for .folder file in directory +1 for 'other' class
        self.rotations = [0,90,180,270,0,90,180,270]

    def construct_label_map(self):
        print("=========== CONSTRUCTING LABEL MAP ===========")
        label_map = np.zeros((self.input_dimension,self.input_dimension))
        listing = os.listdir(self.label_dir)
        listing.sort()
        i=1
        for filename in listing[1:]:
            path = self.label_dir+'/'+filename
            print(path)
            im = georaster.SingleBandRaster(path)
            mask = im.r!=0.0
            label_map *= (~mask) #Used so that overlapping pixels dont get summed value.
            mask = mask.astype(int)*i
            print(i)
            label_map += mask
            i+=1
        return(label_map)
 
    def tile_label_map(self, label_map):
        print("=========== TILING INPUT  ===========")
        label_map_tiles = skimage.util.view_as_windows(label_map,(self.tile_dimension,self.tile_dimension),self.tile_dimension)
        return(label_map_tiles)

    #This function also does input normalization
    def construct_input_data(self, tile_location, augmentation_nr):
        print("=========== CONSTRUCTING TRAINING SET  ===========")
        input_data = np.zeros((self.nb_samples, self.nb_bands, self.window_dimension, self.window_dimension, self.nb_images))

        listing = os.listdir(self.data_dir)
        listing.sort()
        j=0 
        for file in listing[1:]:
            Temp_array = []
            im = georaster.MultiBandRaster(self.data_dir+'/'+file, bands="all")
            for i in range(self.nb_bands):
                tile = im.r[tile_location[0]*self.tile_dimension:(tile_location[0]+1)*self.tile_dimension,tile_location[1]*self.tile_dimension:(tile_location[1]+1)*self.tile_dimension,i]                    
                tile -= np.median(tile)
                tile /= np.percentile(tile,99)
                np.putmask(tile, tile > 1, 1)
                np.putmask(tile, tile < -1, -1)

                padded_overpad = skimage.util.pad(tile,4,'reflect')
                padded = padded_overpad[:-1,:-1].copy()
                if(not(augmentation_nr==0 or augmentation_nr==4)):
                    padded = sk.transform.rotate(padded, self.rotations[augmentation_nr])
                if(augmentation_nr>3):
                    padded = padded[:, ::-1]
                windows = skimage.util.view_as_windows(padded,(self.window_dimension,self.window_dimension)).astype('float32')
                reshaped_windows = np.reshape(windows,(self.nb_tile_pixels,self.window_dimension,self.window_dimension))
                input_data[:self.nb_tile_pixels,i,:,:,j] = reshaped_windows
            j+=1
        print('X_tr size: '+str(sys.getsizeof(input_data)))
        return(input_data)

    def construct_labels(self, label_map_tiles, tile_location, augmentation_nr):
        print("=========== CONSTRUCTING LABELS ===========")
        labels = np.zeros((self.nb_samples,),dtype = int)
        label_tile = label_map_tiles[tile_location[0]][tile_location[1]]
        if(not(augmentation_nr==0 or augmentation_nr==4)):
            label_tile = sk.transform.rotate(label_tile, self.rotations[augmentation_nr])
        if(augmentation_nr>3):
            label_tile = label_tile[:, ::-1]
        for i in range(self.tile_dimension):
            for j in range(self.tile_dimension):
                labels[i*self.tile_dimension+j] = label_tile[i][j]
        return labels


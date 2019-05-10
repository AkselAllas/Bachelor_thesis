import georaster
import skimage
import os
import numpy as np
import sys
import skimage as sk

from rasterio.mask import raster_geometry_mask
import shapefile
from shapely.geometry import shape
import rasterio
import numpy.ma as ma


import matplotlib.pyplot as plt



class s2_preprocessor:

    #Rotation_augmenatation and flipping_augmentation should be 1 or 0
    def __init__(self, input_dimension, label_dir, data_dir, input_data_dir, region_of_interest_shapefile, window_dimension, tile_dimension, nb_images, nb_bands, nb_steps, rotation_augmentation, flipping_augmentation):
        self.input_dimension = input_dimension
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.input_data_dir = input_data_dir
        self.region_of_interest_shapefile = region_of_interest_shapefile
        self.window_dimension = window_dimension
        self.tile_dimension = tile_dimension
        self.nb_tile_pixels= tile_dimension*tile_dimension
        self.nb_images = nb_images
        self.nb_bands = nb_bands
        self.nb_steps = nb_steps
        self.rotation_augmentation = rotation_augmentation
        self.flipping_augmentation = flipping_augmentation
        self.nb_augmentations = (1+rotation_augmentation*3)*(1+flipping_augmentation)
        self.nb_samples = self.nb_tile_pixels
        listing = os.listdir(self.label_dir)
        class_amount = 0
        for filename in listing[1:]:
            if(not("0-0" in filename)):
                continue
            class_amount=class_amount+1
        self.nb_classes = class_amount+1 #+1 for 'other' class
        self.rotations = [0,90,180,270,0,90,180,270]

    def construct_label_map(self, selected_big_tile):
        print("=========== CONSTRUCTING LABEL MAP ===========")
        label_map = np.zeros((self.input_dimension,self.input_dimension))
        listing = os.listdir(self.label_dir)
        listing.sort()
        i=1
        for filename in listing[1:]:
            if(not(selected_big_tile in filename)):
                continue
            path = self.label_dir+'/'+filename
            print(path)
            if(os.stat(path).st_size != 0):
                im = georaster.SingleBandRaster(path)
                mask = im.r!=0.0
                label_map *= (~mask) #Used so that overlapping pixels dont get summed value.
                mask = mask.astype(int)*i
                label_map += mask
            i+=1
        print("Total number of classes in labels is: "+str(i)+" And S2_preprocessor has: "+str(self.nb_classes))
        return(label_map)
 
    def tile_label_map(self, label_map):
        print("=========== TILING INPUT  ===========")
        label_map_tiles = skimage.util.view_as_windows(label_map,(self.tile_dimension,self.tile_dimension),self.tile_dimension)
        return(label_map_tiles)

    #This function also does input normalization
    def construct_input_data(self, tile_location, selected_big_tile):
        print("=========== CONSTRUCTING TRAINING SET  ===========")
        input_data = np.zeros((self.tile_dimension, self.tile_dimension, self.nb_bands, self.nb_images+1)) #last dimension (images) also holds Region of interest true/false mask.

        listing = os.listdir(self.data_dir)
        listing.sort()
        j=0 
        for file in listing[1:]:
            if(not(selected_big_tile in file)):
                continue
            data_source = rasterio.open(self.data_dir+'/'+file)
            for i in range(-1, self.nb_bands):
                if(i==-1):
                    #Read and add ROI shapefile to input_data tensor for masking out ROI in batch_generator function in order to avoid label mismatch.
                    Region_of_interest = shapefile.Reader(self.region_of_interest_shapefile)
                    ROI_geometry = shape(Region_of_interest.shapes())
                    masked = raster_geometry_mask(data_source, ROI_geometry)
                    ROI_mask = ~ np.array(masked[0][tile_location[0]*self.tile_dimension:(tile_location[0]+1)*self.tile_dimension,tile_location[1]*self.tile_dimension:(tile_location[1]+1)*self.tile_dimension])
                    input_data[:self.tile_dimension,:self.tile_dimension,0,self.nb_images] =  ROI_mask
                else:
                    #Preprocess band, then tile it into 8x8 windows and add to input_data.
                    band = data_source.read(1+i)
                    
                    tile = band[tile_location[0]*self.tile_dimension:(tile_location[0]+1)*self.tile_dimension,tile_location[1]*self.tile_dimension:(tile_location[1]+1)*self.tile_dimension]
                    tile -= np.median(tile)
                    tile /= np.percentile(tile,99)
                    np.putmask(tile, tile > 1, 1)
                    np.putmask(tile, tile < -1, -1)

                    input_data[:self.tile_dimension,:self.tile_dimension,i,j] = tile
            j+=1
        print('X_tr size: '+str(sys.getsizeof(input_data)))
        return(input_data)

    def construct_labels(self, label_map_tiles, tile_location):
        print("=========== CONSTRUCTING LABELS ===========")
        label_tile = label_map_tiles[tile_location[0]][tile_location[1]]
        return label_tile


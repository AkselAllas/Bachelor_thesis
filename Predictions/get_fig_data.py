#!/usr/bin/env python
# coding: utf-8

import numpy as numpy
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_fig_data(version, tile):
    fig = numpy.load("fig_"+version+"_"+tile+".npy").item()
    
    #ax = fig.gca()
    
    #xy_data = fig.axes[1].images[0].get_array()
    #xy_data = ax.get_lines().get_data()
    #print(type(xy_data))
    #print(xy_data)
    #print(type(xy_data))
    #print(xy_data.shape)
    
    #xy_data = line.get_xydata()
    
    #plt.imshow(xy_data, vmax=28)
    #plt.show()
    
    Y_predictions = np.array(fig.axes[1].images[0].get_array())
    Y_ground_truth = np.array(fig.axes[0].images[0].get_array())
    #plt.show()
    plt.close()
    del fig
    #plt.imshow(Y_ground_truth, vmax=28)
    return Y_ground_truth, Y_predictions

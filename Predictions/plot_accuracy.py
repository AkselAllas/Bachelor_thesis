#!/usr/bin/env python
# coding: utf-8

import numpy as numpy
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
 
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

fig = numpy.load("fig_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".npy").item()
data = numpy.load("prediction_stats_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".npy")

print("accuracy: "+ str(data[0]))
print("empty_percent: "+ str(data[1]))
print("unique: "+ str(data[2]))

#pilt = np.array(fig2img(fig))
#print(type(pilt))
#print(pilt)
#plt.imshow(pilt, vmax=28)
#plt.legend()
plt.show()

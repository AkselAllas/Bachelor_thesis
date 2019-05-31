#!/usr/bin/env python
# coding: utf-8

import numpy as numpy
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from get_fig_data import *

class_names = np.array([
"Mitte-põllukultuur"
,"Aedmaasikas"
,"Astelpaju"
,"Heintaimed, kõrrelised"
,"Heintaimed, liblikõielised"
,"Kaer"
,"Kanep"
,"Karjatamine"
,"Kartul"
,"Mais"
,"Mustkesa"
,"Muu"
,"Põldhernes"
,"Põlduba"
,"Peakapsas"
,"Porgand"
,"Punapeet"
,"Rukis"
,"Sööti jäetud maa"
,"Suvinisu ja speltanisu"
,"Suvioder"
,"Suviraps ja -rüps"
,"Suvitritikale"
,"Talinisu"
,"Talioder"
,"Taliraps ja -rüps"
,"Talitritikale"
,"Tatar"
])

classes=np.array(range(0,28))
#print(type(classes[0]))

version = str(sys.argv[1])
#version_tile = str(sys.argv[2])
list_tiles = ['191','192', '0','1']

final_map = np.zeros((1024,1024))

q=0
for i in range(2):
    for j in range(2):
        Y_ground_truth, Y_predictions = get_fig_data(version,list_tiles[q])
        for k in range(512):
            for l in range(512):
                final_map[i*512+k][j*512+l] = Y_ground_truth[k][l]
                #final_map[i*512+k][j*512+l] = Y_predictions[k][l]
        q+=1

#axarr[0][0].imshow()
#axarr[0][1].imshow(labels_map)
#axarr[1][0].imshow(input_map)
#axarr[1][1].imshow(labels_map)

#plt.imshow(Y_predictions, vmax=28)
##plt.imshow(Y_ground_truth, vmax=28)
#plt.legend()
#plt.show()

values = np.unique(final_map.ravel())

plt.figure(figsize=(8,4))
im = plt.imshow(final_map,vmax=27)

# get the colors of the values, according to the 
# colormap used by imshow
colors = [ im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=str(int(values[i]))+"-"+class_names[int(values[i])]) ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

#plt.grid(True)
plt.show()

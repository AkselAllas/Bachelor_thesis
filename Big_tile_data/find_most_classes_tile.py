#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
from os import listdir
from os.path import isfile, join
mypath="./"
list_IDs = [f for f in listdir(mypath) if isfile(join(mypath, f))]

most_classes=['10','10']

i=0
for ID in list_IDs:
    if(ID[0]=="Y"):
        Y_np_loaded = np.load(ID)
        Y_np_loaded = np.reshape(Y_np_loaded,(512*512,28))
        Y = np.array([np.where(r==1)[0][0] for r in Y_np_loaded])
        unique = np.unique(Y)
        nb_unique = len(unique)
        if(nb_unique >= (int(most_classes[1][-2:]))):
            str_unique = str(nb_unique)
            inserted_string = ID+str_unique
            if(i==0):
                most_classes[1] = inserted_string
                most_classes[0] = most_classes[1]
            else:
                most_classes[0] = most_classes[1]
                most_classes[1] = inserted_string
                
            print(unique)
            print(most_classes)
            i+=1

print(most_classes)
np.save("most_classes.npy",most_classes)

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
nb_class_pixels = np.load("class_counts.npy")

print(nb_class_pixels)
print(np.sum(nb_class_pixels))

#classes = np.unique(Vd)
classes = np.zeros((28))
for i in range(28):
    classes[i]=i    

total_pixels=0
for i in range(28):
    total_pixels += nb_class_pixels[i]
weights = total_pixels/(28*nb_class_pixels)

for i in range(28):
    print(weights[i])
    print(nb_class_pixels[i])
#np.save("class_weights.npy",weights)

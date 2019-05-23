#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

data_1 = np.load("prediction_stats_"+str(sys.argv[1])+"_"+str(sys.argv[3])+".npy")
data_2 = np.load("prediction_stats_"+str(sys.argv[2])+"_"+str(sys.argv[3])+".npy")

print(str(sys.argv[1]))
print("accuracy: "+ str(data_1[0]))
print("empty_percent: "+ str(data_1[1]))
print("unique: "+ str(data_1[2]))

print(str(sys.argv[2]))
print("accuracy: "+ str(data_2[0]))
print("empty_percent: "+ str(data_2[1]))
print("unique: "+ str(data_2[2]))

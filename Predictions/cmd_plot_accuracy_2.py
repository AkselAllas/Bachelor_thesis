#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

fig = np.load("fig_"+str(sys.argv[1])+"_"+str(sys.argv[3])+".npy").item()
fig.suptitle(str(sys.argv[1]))
data_1 = np.load("prediction_stats_"+str(sys.argv[1])+"_"+str(sys.argv[3])+".npy")
fig_2 = np.load("fig_"+str(sys.argv[2])+"_"+str(sys.argv[3])+".npy").item()
fig_2.suptitle(str(sys.argv[2]))
data_2 = np.load("prediction_stats_"+str(sys.argv[2])+"_"+str(sys.argv[3])+".npy")

print(str(sys.argv[1]))
print("accuracy: "+ str(data_1[0]))
print("empty_percent: "+ str(data_1[1]))
print("unique: "+ str(data_1[2]))

print(str(sys.argv[2]))
print("accuracy: "+ str(data_2[0]))
print("empty_percent: "+ str(data_2[1]))
print("unique: "+ str(data_2[2]))

plt.show()

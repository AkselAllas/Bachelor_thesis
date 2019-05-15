#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

fig = np.load("fig_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".npy").item()
data = np.load("prediction_stats_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".npy")

print("accuracy: "+ str(data[0]))
print("empty_percent: "+ str(data[1]))
print("unique: "+ str(data[2]))

plt.show()

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

tiles = ['0', '1','120', '2', '10', '20', '80', '100',  '160', '170']
x = range(len(tiles))
y = []

avg_1 = 0
avg_2 = 0

version = str(sys.argv[1])

for tile in tiles:
    data_1 = np.load("prediction_stats_v50_"+tile+".npy")
    data_2 = np.load("prediction_stats_"+version+"_"+tile+".npy")
    avg_1 += float(data_1[0])
    avg_2 += float(data_2[0])

avg_1 = avg_1/10
avg_2 = avg_2/10
#    asi1=round(float(data_1[0]),4)
#    asi2=round(float(data_2[0]),4)
#    asi3=round(float(data_1[1]),4)
#    print("Vana kogutäpsus :"+str(asi1))
#    print("Uus kogutäpsus: "+str(asi2))
#    print("0-klassi protsent: "+str(asi3))
#    print("")




print(avg_1)
print(avg_2)

#plt.figure(1,figsize=(7,5))
#plt.plot(xc,train_loss)
#plt.show()

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

x= [6193,10657, 15066,19304,23938,27163, 29432,33914, 38902]
y1= [2.0831,1.9607,1.8690,1.7813,1.7173,1.6874, 1.6556,1.5981, 1.5601]

plt.figure(1,figsize=(7,5))
plt.xlabel('Epohhide arv')
plt.ylabel('Kadu')
plt.plot(x,y1)
plt.show()

#[10657, 15066, 29432, 38902]
# [1.9607, 1.8690, 1.6556, 1.5601]

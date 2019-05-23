#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys

big_tile = str(sys.argv[1])
small_tile = str(sys.argv[2])

Y_np_loaded = np.load("Y_"+big_tile+"_("+small_tile+").npy")
Y_np_loaded = np.reshape(Y_np_loaded,(512*512,28))
Y = np.array([np.where(r==1)[0][0] for r in Y_np_loaded])
unique = np.unique(Y)

print(unique)

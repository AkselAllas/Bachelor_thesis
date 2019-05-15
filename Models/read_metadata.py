#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys

version = str(sys.argv[1])

metadata_dict = np.load("metadata"+version+".npy")

print(metadata_dict)

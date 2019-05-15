#!/usr/bin/env python
# coding: utf-8

import numpy as np

def read_processed_tiles():
    cool_list = []
    processed_tile_dict = np.load("processed_tile_dict.npy")
    dictionary = processed_tile_dict.item()
    blacklist = ['1-1_(9,3)0','1-1_(9,3)1','1-1_(9,4)0','1-1_(9,4)1']
    for big_tile in dictionary:
        for tile in dictionary[big_tile]:
            for i in range(2):
                cool_string = big_tile+"_"+"("+str(tile[0])+","+str(tile[1])+")"+str(i)
                if(not(cool_string in blacklist)):
                    cool_list.append(cool_string)

    return(cool_list)

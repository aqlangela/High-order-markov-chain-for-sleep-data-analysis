#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:02:15 2018

@author: Alicia
"""

import numpy as np
import math  
import copy  
import matplotlib.pyplot as plt  
import data

class EM_MTD:
    def __init__(self):
        d = data.Data()
        self.data = d.get_d()        
        self.order = 2
        self.epsilon = 0.0001
        self.dim = int(1/d.states)
        self.occurence = self.compute_occurence()          
        self.phi = np.zeros(shape=(1, self.dim)) 
        self.matrix = np.zeros(shape=(self.dim, self.dim)) 
    
    
    def Expectation_step(self):
        pass
    def Maximization_step(self):
        pass
    def compute_loglikelihood(self):
        pass
    def compute_occurence(self): #assume it's 2nd order
        occurence = np.zeros(shape=(self.dim**self.order, self.dim))

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]-self.order):
                if self.data[i][j+2] > 0:
                    last2 = int(self.data[i][j])
                    last1 = int(self.data[i][j+1])
                    last0 = int(self.data[i][j+2])
                    row = last1 - 1 + (last2-1) * self.dim
                    col = last0 - 1
                    occurence[row][col] += 1
        return occurence
    
# Xt-2 Xt-1     Xt    
# 1    1       [[ 695.  172.   18.    0.    0.]
# 1    2        [   6.   87.   60.   12.   12.]
# 1    3        [   1.    1.    7.    9.    5.]
# 1    4        [   0.    0.    0.    0.    2.]
# 1    5        [   0.    0.    0.    0.    1.]
# 2    1        [ 177.    3.    0.    0.    0.]
# 2    2        [  96.  298.  132.    0.    0.]
# 2    3        [   0.    1.  160.   38.    0.]
# 2    4        [   0.    1.    1.    4.    6.]
# 2    5        [   0.    0.    0.    0.   12.]
# 3    1        [   5.    0.    0.    0.    0.]
# 3    2        [  80.  141.    7.    0.    0.]
# 3    3        [   3.  171.  586.  178.    0.]
# 3    4        [   0.    0.    8.  184.   38.]
# 3    5        [   0.    0.    0.    0.    5.]
# 4    1        [   0.    0.    0.    0.    0.]
#.....        
       
if __name__ == "__main__": 
    em = EM_MTD()
    print(em.occurence)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:43:57 2018

@author: Alicia
"""

import numpy as np
import math  as mt
import copy  
import matplotlib.pyplot as plt  


sigma = 6  
miu_1 = 40  
miu_2 = 20  

N = 4000  
X = np.zeros((1, N))  
for i in range(N):  
    if random.random() > 0.5:
        X[0, i] = np.random.randn() * sigma + miu_1  
    else:  
        X[0, i] = np.random.randn() * sigma + miu_2  
  
k = 2  
miu = np.random.random((1, k))  
 
Expectations = np.zeros((N, k))  

  
for step in range(100000):#设置迭代次数  
    #步骤1，计算期望  
    for i in range(N):  
        #计算分母  
        denominator = 0  
        for j in range(k):  
            denominator = denominator + mt.exp(-1 / (2 * sigma ** 2) * (X[0, i] - miu[0, j]) ** 2)  
          
        #计算分子  
        for j in range(k):  
            numerator = mt.exp(-1 / (2 * sigma ** 2) * (X[0, i] - miu[0, j]) ** 2)  
            Expectations[i, j] = numerator / denominator  
      
    #步骤2，求期望的最大  
    #oldMiu = miu  
    oldMiu = np.zeros((1, k))  
    for j in range(k):  
        oldMiu[0, j] = miu[0, j]  
        numerator = 0  
        denominator = 0  
        for i in range(N):  
            numerator = numerator + Expectations[i, j] * X[0, i]  
            denominator = denominator + Expectations[i, j]  
        miu[0, j] = numerator / denominator  
          
    #判断是否满足要求  
    epsilon = 0.00001  
    if abs(miu - oldMiu)[0,0]+abs(miu - oldMiu)[0,1] < epsilon:  
        break  
      
    print( step ) 
    print( miu ) 
      
  
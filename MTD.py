#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 14:56:17 2018

@author: chenzhengyi
"""

import data
import math
import numpy as np


class MTD:
    
    def __init__(self):
        self.dim = 5 #dimension as states

        d = data.Data()
        self.d = d.get_d()

        self.q1 = np.zeros(shape=(self.dim, self.dim))
        self.c1 = np.zeros(shape=(self.dim, self.dim))

        self.c2 = np.zeros(shape=(self.dim, self.dim))
        self.q2 = np.zeros(shape=(self.dim, self.dim))
        
    def cal_ci(self,list):
        #calculate C(i,.)
        d_ci = {}
        for i in range(self.dim):
            d_ci[i] = 0
            for j in range(self.dim):
                d_ci[i] += list[i][j]
        return d_ci

    def cal_cj(self,list):
        #calculate C(.,j)
        d_cj = {}
        for j in range(self.dim):
            d_cj[j] = 0
            for i in range(self.dim):
                d_cj[j] += list[i][j]
        return d_cj
    
    def cal_TCg(self,list):
        TCg = 0
        for i in range(self.dim):
            for j in range(self.dim):
                TCg += list[i][j]
        return TCg
    
    def cal_u(self,list):
        #calculate C(i,.)
        d_ci = self.cal_ci(list)
        #calculate C(.,j)                
        d_cj = self.cal_cj(list)
        #calculate TCg
        TCg = self.cal_TCg(list)
                        
        numerator = 0
        for i in range(self.dim):
            for j in range(self.dim):
                if list[i][j]!= 0:
                    numerator += list[i][j]*math.log2((d_ci[i]*d_cj[j])/(list[i][j]*TCg))                    
                else:
                    pass
                
        denominator = 0
        for j in range(self.dim):
            denominator += d_cj[j]*math.log2(d_cj[j]/TCg)
            
        u = numerator/denominator
        return u
                

    def init_value (self,data_set):
        #calculate c1 and q1
        self.q1 = np.zeros(shape=(self.dim, self.dim))
        self.c1 = np.zeros(shape=(self.dim, self.dim))
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]-1):
                if data_set[i][j+1] > 0:
                    row = int(data_set[i][j]) - 1
                    col = int(data_set[i][j+1]) - 1
                    self.c1[row][col] += 1
                    self.c1[row][col] = int(self.c1[row][col])

        for i in range(self.dim):
            Sum = sum(self.c1[i,:])
            for j in range(self.dim):
                self.q1[i][j] = self.c1[i][j] / Sum
#        print(self.c1)
#        print(self.q1)

        #below is C2 and Q2
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]-2):
                if data_set[i][j+2] > 0:
                    row = int(data_set[i][j]) - 1
                    col = int(data_set[i][j+2]) - 1
                    self.c2[row][col] += 1
                    self.c2[row][col] = int(self.c2[row][col])
                          
        for i in range(self.dim):
            Sum = sum(self.c1[i,:])
            for j in range(self.dim):
                self.q2[i][j] = self.c2[i][j] / Sum
        
#        print(self.c2)
#        print(self.q2)
        u1 = self.cal_u(self.c1)
        u2 = self.cal_u(self.c2)
        lambda1 = u1/(u1+u2)
        lambda2 = u2/(u1+u2)

        print("lambda:\n",lambda1,lambda2)
        if lambda1 > lambda2:
            init_Q = self.q1
        else:
            init_Q = self.q2
            
        print("init_Q:\n",init_Q)

        return lambda1,lambda2,init_Q
    

if __name__ == "__main__": 
    d = data.Data()
    fd = d.get_d()
    mtd = MTD()
    mtd.init_value(fd)
 
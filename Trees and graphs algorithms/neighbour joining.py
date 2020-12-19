#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:58:17 2018

@author: 3602786
"""

import numpy as np
import time 

D = np.array([[0,2,4,6,6,8],[2,0,4,6,6,8],[4,4,0,6,6,8],[6,6,6,0,4,8],[6,6,6,4,0,8],[8,8,8,8,8,0]])
def NJ(D):
    
    
    n = len(D[0])
    U = []
    for i in range(n):
        U.append(np.sum(D[i])/(n-2))
    Q = np.zeros((n,n))
    for i in range(1,n):
        for j in range(0,i):
            Q[i,j] = D[i,j] - U[i] - U[j]
    min_D = np.where(D==(np.eye(n)*D.max()+D).min())
    col_min,row_min = min_D[1][0],min_D[1][1]
    
    for k in range(n):
        if(k!=row_min and k!=col_min):
            D[row_min,k] = (D[row_min,k]+D[k,col_min]-D[row_min,col_min])/2
    
    D = np.delete(D,col_min,1)
    D = np.delete(D,row_min,0)
    
    
    
    
    
   
    
    
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:23:02 2020

@author: space
"""
import numpy as np

def random( matrix,h,w):
    temp_matrix = matrix.copy()
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            row = matrix[i,:]
            row = row[:j+1]
            temp_matrix[i][j] = temp_matrix[i-1][j] + np.sum(row)
    print(temp_matrix)
        
            
            
    
matrix = np.array([[0,0,0,0],
                   [0,1,2,3],
                   [0,4,5,6],
                   [0,7,8,9]])
random(matrix,2,2)
#[1,3,6
# 5,9,24
# 12,27,45
#
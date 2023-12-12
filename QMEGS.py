""" Main routines for QMEGS

Goal: Given signal, output estimatation of dominant frequencies

-Input:

Z_est: np.array of signal

d_x: space step

t_list: np.array of time points

K: number of dominant frequencies

alpha: interval constant

T: maximal time

-Output:

Dominant_freq: np.array of estimation of dominant frequencies (up to adjustment when there is no gap)

Last revision: 10/23/2023
"""

import numpy as np
from matplotlib import pyplot as plt

def QMEGS(Z_est, d_x, t_list, K, alpha, T):
    """
    QMEGS algorithm
    """
    N = len(Z_est)
    num_x=int(2*np.pi/d_x)
    x=np.arange(0,num_x)*d_x-np.pi
    G=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x)))/len(Z_est)) #Gaussian filter function
    Dominant_freq=np.zeros(K,dtype='float')
    for k in range(K):
        max_idx = np.argmax(G)
        Dominant_freq[k]=x[max_idx]
        interval_max=x[max_idx]+alpha/T
        interval_min=x[max_idx]-alpha/T
        G=np.multiply(G,x>interval_max)+np.multiply(G,x<interval_min) #eliminate interval
    return Dominant_freq

def QMEGS_new(Z_est, d_x, t_list, K, alpha, T):
    """
    QMEGS new algorithm
    
    Note: This code is slightly different from the algorithm in the paper. 
    
    To avoid long classical running time, we first do a rough search 
    then do a detailed search around the rough maximal point.
    """
    N = len(Z_est)
    num_x=int(2*np.pi/(d_x*10))
    num_x_detail=int(2*alpha/d_x/T)
    x_rough=np.arange(0,num_x)*d_x*10-np.pi
    G=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x_rough)))/len(Z_est)) #Gaussian filter function
    Dominant_freq=np.zeros(K,dtype='float')
    for k in range(K):
        max_idx_rough = np.argmax(G)
        Dominant_potential=x_rough[max_idx_rough]
        x=np.arange(0,num_x_detail)*d_x+Dominant_potential-alpha/T
        G_detail=np.abs(Z_est.dot(np.exp(1j*np.outer(t_list,x)))/len(Z_est))
        max_idx_detail = np.argmax(G_detail)
        Dominant_freq[k]=x[max_idx_detail]
        interval_max=x[max_idx_detail]+alpha/T
        interval_min=x[max_idx_detail]-alpha/T
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min) #eliminate interval
    return Dominant_freq


""" Main routines for ESPRIT

Goal: Given signal, output estimatation of dominant frequencies

-Input:

f_values: np.array of signal

K: number of dominant frequencies (guess)

M: number of rows in hankel
    
N: number of columns in hankel

-Output:

Dominant_freq: np.array of estimation of dominant frequencies (up to adjustment when there is no gap)

Last revision: 10/23/2023
"""

import numpy as np
import scipy.linalg as la

def form_hankel(f_values, M, N):
    """
    Generate Hankel matrix
    """
    hankel_matrix = np.zeros((M+1, N-M), dtype=complex)
    for i in range(M+1):
        for j in range(N-M):
            # For a Hermitian Toeplitz matrix, matrix[i, j] is determined by f_values[abs(i - j)]
            hankel_matrix[i, j] = f_values[i+j]
            
    return hankel_matrix

def ESPRIT(f_values, M, N, K, dt=1):
    """
    -Input:

    f_values: np.array of signal

    K: number of dominant frequencies
    
    M: number of rows in hankel
    
    N: number of columns in hankel
    
    dt: time step in ESPRIT (default dt=1)

    -Output:

    Dominant_freq: np.array of estimation of dominant frequencies (up to adjustment when there is no gap)
    """
    N = len(f_values)
    hankel_matrix = form_hankel(f_values, M, N)
    U_hankel, S_hankel, _ = la.svd(hankel_matrix)
    # Find indices of first r columns
    U_hankel_trunc = U_hankel[:,:K]
    S_hankel_trunc = S_hankel[:K]
    U1_matrix = U_hankel_trunc[:-1,:]
    U2_matrix = U_hankel_trunc[1:,:]
    res = np.linalg.lstsq(U1_matrix, U2_matrix, rcond=None)
    psi_matrix = res[0]
    S_psi, V_psi = la.eig(psi_matrix)
    Dominant_freq = np.sort(-np.angle(S_psi)/dt)
    return Dominant_freq



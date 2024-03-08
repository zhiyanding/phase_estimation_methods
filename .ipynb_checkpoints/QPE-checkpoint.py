""" Main routines for QPE

Goal: Use QPE to estimate ground state

-Input:

spectrum: np.array of eigenvalues

population: np.array of overlap

T: maximal runtime of QPE

N: number of repetitions of QPE

-Output:

ground_state_energy: estimation of ground state energy

Last revision: 10/23/2023
"""

import numpy as np


    
def eval_Fejer_kernel(T,x):
    """
    Generate the kernel of QPE
    """
    x_avoid = np.abs(x % (2*np.pi))<1e-8
    numer = np.sin(0.5*T*x)**2
    denom = np.sin(0.5*x)**2
    denom += x_avoid
    ret = numer / denom
    ret = (1-x_avoid)*ret + (T**2)*x_avoid
    return ret/T

def generate_QPE_distribution(spectrum,population,T):
    """
    Generate the index distribution of QPE
    """
    T=int(T)
    N = len(spectrum)
    dist = np.zeros(T)
    j_arr = 2*np.pi*np.arange(T)/T - np.pi
    for k in range(N):
        dist += population[k] * eval_Fejer_kernel(T,j_arr-spectrum[k])/T
    return dist

def draw_with_prob(measure,N):
    """
    Draw N indices independently from a given measure
    """
    L = measure.shape[0]
    cdf_measure = np.cumsum(measure)
    normal_fac = cdf_measure[-1]
    U = np.random.rand(N) * normal_fac
    index = np.searchsorted(cdf_measure,U)
    return index


def QPE(spectrum,population,T,N):
    """
    QPE Main routine
    """
    discrete_energies = 2*np.pi*np.arange(T)/(T) - np.pi 
    index_dist = generate_QPE_distribution(spectrum,population,T) #Generate QPE samples
    index_samp = draw_with_prob(index_dist,N)
    index_min = index_samp.min()
    ground_state_energy = discrete_energies[index_min]
    return ground_state_energy


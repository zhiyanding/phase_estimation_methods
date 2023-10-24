""" Main routines for MM-QCELS 

Goal: Using Multi-level QCELS to estimate the multiple dominant eigenvalues.

Input:

spectrum: np.array of eigenvalues

population: np.array of overlap

N_0: initial number of samples
 
T_0: initial depth for QCELS

N: iteration number of samples

T: last depth for QCELS

gamma: truncation constant

K: number of dominant eigenvalues

lambda_prior: initial guess of multiple dominant eigenvalues

Output: 

Dominant_eign: an estimation of multiple dominant eigenvalues

T_max: maximal evolution time 

T_total: total evolution time 

Last revision: 10/23/2023
"""

import scipy.io as sio
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import cmath
from scipy.stats import truncnorm

def generate_Hadamard_test_data_QCELS(spectrum,population,t_list,N_list):
    """ -Input:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlap
    t_list: np.array of time points
    N_list: np.array of numbers of samples
   
    -Ouput:
    
    Z_Had: np.array of the output of Hadamard test (row)
    T_max: maximal Hamiltonian simulation time
    T_total: total Hamiltonian simulation time

    """
    if len(t_list)!=len(N_list):
       print('list error')
    t_list=np.array(t_list)
    N_list=np.array(N_list)
    N_list=N_list.flatten()
    N=len(t_list)
    Nsample=int(max(N_list))
    #generate true expectation
    z=population.dot(np.exp(-1j*np.outer(spectrum,t_list)))
    Re_true=(1+np.real(z))/2
    Im_true=(1+np.imag(z))/2
    #construct check matrix for different Nsample
    N_check=np.arange(Nsample).reshape([Nsample, 1])
    N_check=N_check*np.ones((1,N))
    Sign_check=np.ones((Nsample, 1))*(N_list-0.5)
    Re_check=(np.sign(N_check-Sign_check)-1)/(-2)
    Im_check=(np.sign(N_check-Sign_check)-1)/(-2)
    Re_true=np.multiply(Re_check,np.ones((Nsample, 1)) * Re_true)
    Im_true=np.multiply(Im_check,np.ones((Nsample, 1)) * Im_true)
    #simulate Hadamard test
    Re_random=np.random.uniform(0,1,(Nsample,N))
    Im_random=np.random.uniform(0,1,(Nsample,N))
    Re=np.sum(Re_random<Re_true,axis=0)/N_list
    Im=np.sum(Im_random<Im_true,axis=0)/N_list
    Z_Had = (2*Re-1)+1j*(2*Im-1)
    T_max = max(np.abs(t_list))
    T_total = sum(np.multiply(np.abs(t_list),N_list))
    return Z_Had, T_max, T_total

def generate_ts_distribution_QCELS(T,N,gamma):
    """ Generate time samples from truncated Gaussian
    Input:
    
    T : variance of Gaussian
    gamma : truncated parameter
    N : number of samples
    
    Output: 
    
    t_list: np.array of time points
    """
    t_list=truncnorm.rvs(-gamma, gamma, loc=0, scale=T, size=N)
    return t_list

def generate_Z_QCELS(spectrum,population,T,N,gamma):
    """ Generate Z samples for a given T,N,gamma
    Input:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlap
    T : variance of Gaussian
    gamma : truncated parameter
    N : number of time samples
    
    Output: 
    
    t_list: np.array of time points
    """
    t_list = generate_ts_distribution_QCELS(T,N,gamma)
    N_list = np.ones(len(t_list))
    T_max = max(np.abs(t_list))
    T_total = sum(np.multiply(np.abs(t_list),N_list))
    Z_est, _ , _ =generate_Hadamard_test_data_QCELS(spectrum,population,t_list,N_list)
    return Z_est, t_list, T_max, T_total

def qcels_opt_fun(x, t_list, Z_est):
    "Calculate the QCELS loss function"
    N_T = t_list.shape[0]
    N_x=int(len(x)/3)
    Z_fit = np.zeros(N_T,dtype = 'complex_')
    for n in range(N_x):
       Z_fit = Z_fit + (x[3*n]+1j*x[3*n+1])*np.exp(-1j*x[3*n+2]*t_list)
    return (np.linalg.norm(Z_fit-Z_est)**2/N_T)

def qcels_opt_fun_coeff(x, t_list, Z_est, x0):
    "Calculate the QCELS loss function with different overlap"
    N_T = t_list.shape[0]
    N_x=int(len(x0)/3)
    Z_fit = np.zeros(N_T,dtype = 'complex_')
    for n in range(N_x):
       Z_fit = Z_fit + (x[2*n]+1j*x[2*n+1])*np.exp(-1j*x0[3*n+2]*t_list)
    return (np.linalg.norm(Z_fit-Z_est)**2/N_T)

def qcels_opt_multimodal(t_list, Z_est, x0, bounds = None, method = 'SLSQP'):
    "MM-QCELS minimization step"
    fun = lambda x: qcels_opt_fun(x, t_list, Z_est)
    N_x=int(len(x0)/3)
    bnds=np.zeros(6*N_x,dtype = 'float')
    for n in range(N_x):
       bnds[6*n]=-1
       bnds[6*n+1]=1
       bnds[6*n+2]=-1
       bnds[6*n+3]=1
       bnds[6*n+4]=-np.inf
       bnds[6*n+5]=np.inf
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    return res

def qcels_opt_coeff_first(t_list, Z_est, x0, bounds = None, method = 'SLSQP'):
    "MM-QCELS minimize coefficient"
    N_x=int(len(x0)/3)
    coeff=np.zeros(N_x*2)
    bnds=np.zeros(4*N_x,dtype = 'float')
    for n in range(N_x):
       bnds[4*n]=-1
       bnds[4*n+1]=1
       bnds[4*n+2]=-1
       bnds[4*n+3]=1
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    for n in range(N_x):
       coeff[2*n]=x0[3*n]
       coeff[2*n+1]=x0[3*n+1]
    fun = lambda x: qcels_opt_fun_coeff(x, t_list, Z_est, x0)    
    res=minimize(fun,coeff,method = 'SLSQP',bounds=bnds)
    x_out=x0
    for n in range(N_x):
       x_out[3*n]=res.x[2*n]
       x_out[3*n+1]=res.x[2*n+1]
    return x_out


def qcels_multimodal(spectrum, population, T_0, T, N_0, N, gamma, K, lambda_prior):        
    """Multi-level QCELS main routine.
    
    """
    T_total = 0.
    T_max = 0.
    N_level=int(np.log2(T/T_0))#number of layers
    x0=np.zeros(3*K,dtype = 'float')#store dominant eigenvalues and overlaps
    ###----First step----####
    Z_est, t_list, max_time, total_time=generate_Z_QCELS(
        spectrum,population,T_0,N_0,gamma) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
    T_total += total_time
    T_max = max(T_max, max_time)
    N_initial=10 #number of random initial search
    lambda_prior_collect=np.zeros((N_initial,len(lambda_prior)),dtype = 'float')
    lambda_prior_collect[0,:]=lambda_prior
    for n in range(N_initial-1):
        lambda_prior_collect[n+1,:]=np.random.uniform(spectrum[0],spectrum[-1],K)
    #Step up and solve the optimization problem
    Residue=np.inf
    for p in range(N_initial):#try different initial to make sure find global minimal
        lambda_prior_new=lambda_prior_collect[p,:]
        for n in range(K):
           x0[3*n:3*n+3]=np.array((np.random.uniform(0,1),0,lambda_prior_new[n]))
        x0 = qcels_opt_coeff_first(t_list, Z_est, x0)
        res = qcels_opt_multimodal(t_list, Z_est, x0)#Solve the optimization problem
        if res.fun<Residue:
            x0_fix=np.array(res.x)
            Residue=res.fun
    #Update initial guess for next iteration
    #Update the estimation interval
    x0=x0_fix
    bnds=np.zeros(6*K,dtype = 'float')
    for n in range(K):
       bnds[6*n]=-np.infty
       bnds[6*n+1]=np.infty
       bnds[6*n+2]=-np.infty
       bnds[6*n+3]=np.infty
       bnds[6*n+4]=x0[3*n+2]-np.pi/T_0
       bnds[6*n+5]=x0[3*n+2]+np.pi/T_0
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    ###----Iteration step----####
    for n_QCELS in range(N_level):
        T=T_0*2**(n_QCELS+1)
        Z_est, t_list, total_time, max_time=generate_Z_QCELS(
            spectrum,population,T,N,gamma) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        T_total += total_time
        T_max = max(T_max, max_time)
        #Step up and solve the optimization problem
        res = qcels_opt_multimodal(t_list, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        x0=np.array(res.x)
        #Update the estimation interval
        bnds=np.zeros(6*K,dtype = 'float')
        for n in range(K):
           bnds[6*n]=-np.infty
           bnds[6*n+1]=np.infty
           bnds[6*n+2]=-np.infty
           bnds[6*n+3]=np.infty
           bnds[6*n+4]=x0[3*n+2]-np.pi/T
           bnds[6*n+5]=x0[3*n+2]+np.pi/T
        bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    Dominant_eign=x0
    return x0, T_max, T_total


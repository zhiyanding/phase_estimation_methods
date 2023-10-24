""" Data generator: 

-Compatible model: TFIM, Hubbard, Random spectrum with a given gap

Goal:

1. Given p_0, p_1, output initial spectrum and population. 

2. Given t_list, N_list, output data from Hadamard test

Note: For every model, the Hamiltonian H is normalized so that \|H\|\leq 1

-Parameters

p_list: np.array of required overlap (assuming has the form [p_0,p_1])
 
t_list: np.array of time points

N_list: np.array of number of sample points at each t_list (often set=1)

-Outputs

spectrum: np.array of eigenvalues

population: np.array of overlap

Z_Had: np.array of the output of Hadamard test (row)

T_max: maximal Hamiltonian simulation time

T_total: total Hamiltonian simulation time

Last revision: 10/23/2023
"""

import numpy as np 
import scipy.sparse
import scipy.linalg as la
import scipy.io
from matplotlib import pyplot as plt

from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel

def generate_random_Hamiltonian(gap_list,p_list,d,verbose=0):
    """ -Input:
    
    gap_list: (\lambda_1-\lambda_0, \lambda_2-\lambda_1)
    p_list: required overlap
    d: dimension
    
    -Ouput:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlaps
    
    Note: original initial state is uniformly drawn from unit circle.
    """
    gap0 = gap_list[0]
    gap1 = gap_list[1]
    ##----build random Hamiltonian----##
    H = np.random.rand(d,d)+1j*np.random.rand(d,d)
    H = (H*H.conj().T)/2 #make Hermitian and positive definite
    eigenenergies_raw, eigenstates = la.eig(H)
    eigenenergies_raw=np.sort(np.abs(eigenenergies_raw))
    eigenenergies=eigenenergies_raw/np.max(np.abs(eigenenergies_raw))*(1-gap0-gap1)
    eigenenergies[1:]=eigenenergies[1:]-(eigenenergies[1]-eigenenergies[0])+gap0
    eigenenergies[2:]=eigenenergies[2:]-(eigenenergies[2]-eigenenergies[1])+gap1
    
    ##----Random initial state----##
    initial_states = np.random.normal(0,1,d)+1j*np.random.normal(0,1,d)
    initial_states = initial_states/la.norm(initial_states)
    spectrum_raw = eigenenergies
    population_raw = np.abs(np.dot(eigenstates.conj().T, initial_states))**2
    
    ##----Reorganize spectrum and population----##
    spectrum, population=organize_spectrum_population(spectrum_raw, population_raw, p_list)
    plt.plot(spectrum,population,'b-o');plt.show()
    
    return spectrum, population
    
    
def generate_initial_Hubbard(L,J,U,mu,N_up,N_down,p_list,d,verbose=0):
    """ -Input:
    
    L: number of sites
    J,U,mu,N_up,N_down: parameters of Hubbard
    p_list: required overlap
    d: final number of basis
    
    -Ouput:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlaps
    
    Note: original initial state is uniformly drawn from unit circle.
    """
    ##----build Hubbard Hamiltonian----##
    basis = spinful_fermion_basis_1d(L,Nf=(N_up,N_down))
    if verbose > 0:
        print(basis)
    
    # define site-coupling lists
    hop_right=[[-J,i,i+1] for i in range(L-1)]
    hop_left= [[+J,i,i+1] for i in range(L-1)]
    pot=[[-mu,i] for i in range(L)] # -\mu \sum_j n_{j \sigma}
    interact=[[U,i,i] for i in range(L)] # U/2 \sum_j n_{j,up} n_{j,down}
    # define static and dynamic lists
    static=[
            ['+-|',hop_left],  # up hops left
            ['-+|',hop_right], # up hops right
            ['|+-',hop_left],  # down hops left
            ['|-+',hop_right], # down hops right
            ['n|',pot],        # up on-site potention
            ['|n',pot],        # down on-site potention
            ['z|z',interact]   # up-down interaction with z=c^{\dag} c-1/2
           ]
    dynamic=[]
    # build Hamiltonian
    if verbose == 0:
        no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
    else: 
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
    ##----Random initial state----##
    eigenenergies, eigenstates = H.eigsh(k=d,which="SA")
    dim=len(eigenstates[:,0])
    initial_states = np.random.normal(0,1,dim)+1j*np.random.normal(0,1,dim)
    initial_states = initial_states/la.norm(initial_states)
    spectrum_raw = eigenenergies
    population_raw = np.abs(np.dot(eigenstates.conj().T, initial_states))**2
    
    ##----Reorganize spectrum and population----##
    spectrum, population=organize_spectrum_population(spectrum_raw, population_raw, p_list)
    plt.plot(spectrum,population,'b-o');plt.show()
    
    return spectrum, population

def generate_initial_TFIM(L,J,g,p_list,d,verbose=0):
    """ -Input:
    
    L: number of qubits
    J,g: parameters of TFIM
    p_list: required overlap
    d: final number of basis
    
    -Ouput:
    
    spectrum: np.array of eigenvalues
    population: np.array of overlaps
    
    Note: original initial state is uniformly drawn from unit circle.
    """
    ##----build TFIM Hamiltonian----##
    basis = spin_basis_1d(L=L)
    if verbose > 0:
        print(basis)
    
    # define site-coupling lists
    h_field=[[-g,i] for i in range(L)]
    J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC
    static =[["zz",J_zz],["x",h_field]] # static part of H
    dynamic=[]
    # build Hamiltonian
    if verbose == 0:
        no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
    else: 
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
    ##----Random initial state----##
    eigenenergies, eigenstates = H.eigsh(k=d,which="SA")
    dim=len(eigenstates[:,0])
    initial_states = np.random.normal(0,1,dim)+1j*np.random.normal(0,1,dim)
    initial_states = initial_states/la.norm(initial_states)
    spectrum_raw = eigenenergies
    population_raw = np.abs(np.dot(eigenstates.conj().T, initial_states))**2
    ##----Reorganize spectrum and population----##
  
    spectrum, population=organize_spectrum_population(spectrum_raw, population_raw, p_list)
    plt.plot(spectrum,population,'b-o');plt.show()
    
    return spectrum, population

def organize_spectrum_population(spectrum_raw, population_raw, p_list):
    """ -Input:
    
    spectrum_raw: np.array of original spectrum
    population_raw: np.array of original overlap
   
    -Ouput:
    
    spectrum: np.array of adjusted eigenvalues
    population: np.array of adjusted overlaps
    """
    p = np.array(p_list)
    spectrum = spectrum_raw /np.max(np.abs(spectrum_raw))#normalize the spectrum
    q = population_raw
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def generate_Hadamard_test_data(spectrum,population,t_list,N_list):
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
#installation:
#https://pylians3.readthedocs.io/_/downloads/en/master/pdf/
import numpy as np
import matplotlib.pyplot as plt
import MAS_library as MASL
import Pk_library as PKL

## Load Halo from: https://quijote-simulations.readthedocs.io/en/latest/halos.html
# catalog file
#'/home/fvillaescusa/Quijote/Halos/fiducial/0/out_4_pid.list'
f_catalog = '/scratch/network/vk9342/hlist_1.00000.list'
# read the halo catalog
data = np.loadtxt(f_catalog)
# we can now get the different properties of the halos
Mvir = data[:,10]
sam_Mvir = data[:,9]
# Vmax = data[:,3]
# PID  = data[:,41]
x = data[:,17]
y = data[:,18]
z = data[:,19]


## Compute density field delta
# https://pylians3.readthedocs.io/en/master/construction.html

# number of particles
Np = 128**3

# density field parameters
grid    = 128*4    #the 3D field will have grid x grid x grid voxels
BoxSize = 1000.0 #Mpc/h ; size of box
MAS     = 'TSC'  #mass-assigment scheme   
verbose = False   #print information on progress

## particle positions in 3D
#pos = np.random.random((Np,3)).astype(np.float32)*BoxSize 
pos = np.vstack((x, y, z)).T  
pos = pos.astype(np.float32)                

# define 3D density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)

# construct 3D density field
MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

# at this point, delta contains the effective number of particles in each voxel
# now compute overdensity and density constrast
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

## Make power spectrum 
# compute power spectrum
axis = 0  #Axis along which compute the quadrupole, hexadecapole and the 2D power spectrum.
threads = 1  #Number of openmp threads to be used
Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose) 

# Pk is a python class containing the 1D, 2D and 3D power spectra, that can be retrieved as

# 1D P(k)
k1D      = Pk.k1D
Pk1D     = Pk.Pk1D
Nmodes1D = Pk.Nmodes1D

# 2D P(k)
kpar     = Pk.kpar
kper     = Pk.kper
Pk2D     = Pk.Pk2D
Nmodes2D = Pk.Nmodes2D

# 3D P(k)  #WHAT IS THE DIFFERENCE BETWEEN EACH ONE
k       = Pk.k3D
Pk0     = Pk.Pk[:,0] #monopole
Pk2     = Pk.Pk[:,1] #quadrupole
Pk4     = Pk.Pk[:,2] #hexadecapole
Pkphase = Pk.Pkphase #power spectrum of the phases
Nmodes  = Pk.Nmodes3D


## Plot power spectrum

plt.loglog(k, Pk0, 'ro', alpha = 0.1)
plt.xlabel('k [h/Mpc]')
plt.ylabel('Pk [(Mpc/h)^3]')

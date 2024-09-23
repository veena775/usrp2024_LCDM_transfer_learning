import numpy as np
import sys, os
import matplotlib.pyplot as plt
from sbi_functions import *
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'

def get_error (k_max, num_folders):
    std = []
    mean = []
    for num in num_folders:
        cosmo_param_samples = np.load('samples/fiducial_samples/2000_fiducial_cosmo_param_samples_'+str(num)+'_'+str(k_max)+'.npy') 
        std.append(np.std(cosmo_param_samples, axis = 0))
        mean.append(np.mean(cosmo_param_samples, axis = 0))
    std = np.vstack(std)
    mean = np.vstack(mean)    
    return (std)


num_folders = [20, 50, 100, 200, 400, 1000, 2000, 4000, 8000, 16000, 32000]
k_max = 0.1
std_1 = get_error(k_max, num_folders)
k_max = 0.5
std_5 = get_error(k_max, num_folders)
print(std_1.shape)
print(std_5.shape)


num_cosmo_params = 5
fig, axes = plt.subplots(1, num_cosmo_params, figsize=(25, 5), sharey = True)
param_names = [r'$\Omega_m$', r'$\Omega_b$', r'h', r'$n_s$', r'$\sigma_8$']

for i in range (num_cosmo_params):
    ax = axes[i]
    ax.plot(num_folders, std_1[:, i], color = 'tomato')
    # ax.plot(num_folders, std_5[:, i], color = 'royalblue')
    ax.set_xscale('log')
    ax.set_title (param_names[i], size = 20)
    ax.set_xlabel ('$N_{sim}$', size = 20)
axes[0].set_ylabel('$\sigma$', size = 20)

plt.suptitle ('$k_{max}=0.1$,    fiducial', size = 20)      ### CHANGE
fig.tight_layout(rect=[0, 0, 1, 0.99])
#plt.savefig('fiducial_errorbar_sigma_0.1_log.png')     ### CHANGE             
%matplotlib inline
plt.show()


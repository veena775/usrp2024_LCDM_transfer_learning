import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm


num_cosmo_params = 5
num_samples = 1000
param_names = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']

num_Pk_folders = 1000
real_params_txt = np.loadtxt ('real_params/latin_hypercube_params_nwLH.txt')
num_bins = 3  # Define the number of bins
k_max = 0.5


fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey = True)  # 5 rows, 1 column

for n in range(num_cosmo_params):  # Loop through 5 cosmo params
    w = []
    M_nu = []
    mu = []
    sigma  = []
    
    for i in range(num_Pk_folders):  
        real_params = real_params_txt[i][:num_cosmo_params]  # First 5 real params for the ith sim
        samples_file = '/scratch/network/vk9342/nwLH_samples/1_'+str(i)+'_nwLH_cosmo_param_samples_32000_'+str(k)+'.npy'   ### CHANGE
        
        if os.path.exists(samples_file):
            w.append(real_params_txt[i][6])  # w for one Pk file
            M_nu.append(real_params_txt[i][5])  # Neutrino mass for one Pk file
            
            samples = np.load(samples_file)  # Should be 1000 x 5
            std = np.std(samples, axis=0)  # Std of those 1000 samples for one Pk (array length 5)
            residual = [(real_params[n] - samples[j][n])/std[n] for j in range(num_samples)]  # residuals for one cosmo param of one file (length 1000)
            mu.append(np.mean(residual)) # List of length num_Pk_folders with the mean of the residual of one Pk file
            sigma.append (np.std(residual))
            
        else:
            print('File ' + samples_file + ' not found')

    # Convert lists to numpy arrays
    w = np.array(w)
    M_nu = np.array(M_nu)
    mu = np.array(mu)
    sigma = np.array(sigma)
    

    # Create 2D histogram to bin the data
    z, xedges, yedges = np.histogram2d(w, M_nu, bins=num_bins, weights=sigma, density=False)     ### CHANGE
    counts, _, _ = np.histogram2d(w, M_nu, bins=num_bins)

    # Calculate average mu in each bin
    z_avg = z / counts
    z_avg = np.nan_to_num(z_avg)  # Replace NaNs with 0
    z_avg = abs(z_avg)
    
    # Plot the heatmap
    ax = axes[n]
    ax.set_xlabel('$w$', size =20)          
    ax.set_title(param_names[n], size = 20)
    
    #colorbar specs
    pcm = ax.pcolormesh(xedges, yedges, z_avg.T, cmap='Reds', vmin = 0.99999995949001, vmax = 1.0000000879776834) #, norm = nrom)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('$\sigma$', size = 20)      ### CHANGE

# cbar = fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
# cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', fraction=0.04, pad=1.4)
# cbar.set_label('$\mu$', size = 20)             ### CHANGE

axes[0].set_ylabel(r'$M_{\nu}$', size = 20)

fig.tight_layout(rect=[0, 0, 1, 0.92])        
fig.suptitle('nwLH $\sigma$,  $k_{max}$ = '+str(k_max), size = 20)        ### CHANGE
plt.savefig('pcolormesh_std_1000_nwLH_32000_'+str(k_max)+'_red.png', dpi=300)        ### CHANGE
%matplotlib inline
plt.show()


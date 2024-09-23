import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'


num_cosmo_params = 5
param_names = [r'$\Omega_m$', r'$\Omega_b$', r'h', r'$n_s$', r'$\sigma_8$']
num_samples = 1000 ## num samples per power spectum (set in sampling code)
num_Pk_folders = 1000  ## number of Pk files to sample from
real_params_txt = np.loadtxt ('latin_hypercube_params_nwLH.txt')  ## 2000 x 7


#### CALCULATE RESIDUALS
z = []
for n in range(num_cosmo_params): 
    residuals = []  # all residuals for a specific cosmological parameter
    for i in range(num_Pk_folders):  
        real_params = real_params_txt[i][:num_cosmo_params]  # First 5 real params for the ith sim; len 5
        ## sample of params for ith simulation (Pk file) out of num_Pk_folders:
        samples_file = 'nwLH_samples/1_'+str(i)+'_nwLH_cosmo_param_samples_32000_0.1.npy'    ##CHANGE LAST NUMBER FOR KMAX 0.1 OR 0.5 
        
        if os.path.exists(samples_file):
            samples = np.load(samples_file)  # Should be 1000 x 5
            std = np.std(samples, axis=0)  # Std of those 1000 samples for one Pk (length 5)
            mean = np.mean(samples, axis=0)  # Mean of those 1000 samples for one Pk (length 5)
            
            residual = [(real_params[n] - samples[j][n])/std[n] for j in range(num_samples)]  # residuals for one cosmo param of one file (length 1000)
            residuals.append(residual)  #residuals of one cosmo param for all Pk files (length (num_Pk_foldersx1000))
        else:
            print('File '+ samples_file+ ' not found')
            
    residuals  = np.hstack(residuals)
    print(residuals.shape)
    z.append(residuals)  
    
z = np.vstack(z) 
z = np.array(z)
print(z.shape)  ## z should be 5 x (num Pk foldersx1000)



#### PLOT HISTORGRAM
lim = 10
fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 5 rows, 1 column

for i in range(num_cosmo_params):
    ax = axes[i]
    ax.hist(z[i], bins=50, density=True, histtype = 'step', color = 'tomato',
            label = '$k_{max} = 0.1$ \n $\mu$ = '+ str(np.round(np.mean(z[i]), 5)) + '\n $\sigma$ = '+ str(np.round(np.std(z[i]), 5)), 
            alpha = 0.7)         ### CHANGE LABEL FOR DIFFERENT KMAX
    
    ax.set_title(param_names[i], size = 20)
    ax.axline((0, 0), (0, 0.4), linewidth=1, color='grey')
    ax.set_xlabel('z', size = 15)
    ax.legend()
    
    ## PLOT GAUSSIAN
    mu_gauss, std_gauss = 0, 1
    x = np.linspace(-lim, lim, 1000)
    gaussian_curve = stats.norm.pdf(x, mu_gauss, std_gauss)
    ax.plot(x, gaussian_curve, '--', linewidth=2, color = 'tomato')
    
axes[0].set_ylabel('counts', size = 15)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.suptitle ('$k_{max}$ = 0.1,  nwLH', size = 20)    ### CHANGE FOR DIFFERENT KMAX
# plt.savefig('residuals_1000_nwLH_32000_0.1_0.5.png', dpi=300)   ### CHANGE FOR DIFFERENT KMAX
%matplotlib inline
plt.show()



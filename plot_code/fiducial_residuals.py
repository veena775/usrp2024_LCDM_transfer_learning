import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib as mpl
from astropy.modeling.models import Gaussian1D
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'


samples_5 = np.load('samples/fiducial_samples/2000_fiducial_cosmo_param_samples_32000_0.5.npy')
samples_1 = np.load('samples/fiducial_samples/2000_fiducial_cosmo_param_samples_32000_0.1.npy')
param_names = [r'$\Omega_m$', r'$\Omega_b$', r'h', r'$n_s$', r'$\sigma_8$']
real_params = [0.3175, 0.049, 0.6711, 0.9624, 0.834]
print(real_params)

samples_5 = np.vstack(samples_5)
std_5 = np.std(samples_5, axis = 0)
mean_5 = np.mean(samples_5, axis = 0)

samples_1 = np.vstack(samples_1)
std_1 = np.std(samples_1, axis = 0)
mean_1 = np.mean(samples_1, axis = 0)

print(samples_5.shape)
print(samples_1.shape)
print(std_5, std_1)
print(mean_5, mean_1)

lim = 4
num_samples = 2000000
fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 5 row, 1 columns
for i in range(5):
    z_1 = [(real_params[i] - samples_1[j][i]) / std_1[i] for j in range(num_samples)]
    z_5 = [(real_params[i] - samples_5[j][i]) / std_5[i] for j in range(num_samples)]
    print(len(z_1))
    print(len(z_5))
    ax = axes[i]
    ax.hist(z_1, bins=50, density=True, range=(-lim, lim), 
            label = '$k_{max} = 0.1$ \n $\mu$ = '+ str(np.round(np.mean(z_1), 5)) + '\n $\sigma$ = '+ str(np.round(np.std(z_1), 5)), 
            color = 'tomato', histtype = 'step')
    ax.hist(z_5, bins=50, density=True, range=(-lim, lim), 
        label = '$k_{max} = 0.5$ \n $\mu$ = '+ str(np.round(np.mean(z_5), 5)) + '\n $\sigma$ = '+ str(np.round(np.std(z_5), 5)), 
            color = 'royalblue', histtype = 'step')
    mu_gauss, std_gauss = 0, 1
    x = np.linspace(-lim, lim, 1000)
    gaussian_curve = stats.norm.pdf(x, mu_gauss, std_gauss)
    ax.plot(x, gaussian_curve, '--', linewidth=1, color = 'black')
    ax.set_title(param_names[i])
    ax.axline((0, 0), (0, 0.4), linewidth=1, color='grey')
    ax.set_ylabel('counts')
    ax.set_xlabel('$z$')
    ax.legend()
    
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.suptitle ('fiducial', size = 20)
plt.savefig('residuals_fiducial_32000_(2000)_0.1_0.5_final.png', dpi=300)
plt.show()
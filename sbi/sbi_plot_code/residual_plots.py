import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib as mpl
from astropy.modeling.models import Gaussian1D
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'


param_names = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$', r'$M_{\nu}$', r'$w$']
num_samples = 1000
num_cosmo_params = 7
real_params_txt = np.loadtxt ('real_params/latin_hypercube_params_nwLH.txt')
num_Pk_folders = 200  #num sims for testing
k_max = 0.5
testing_nums = np.load('testing_nums_nwLH.npy')  #indecies of testing sims

devide_by_std = True


#### CALCULATE RESIDUALS
def get_z_values(num_cosmo_params, num_samples, num_Pk_folders, k_max, real_params_txt):
    z = []
    all_samples = []
    for n in range(num_cosmo_params):  # Loop through 5 cosmo params
        residuals = []  # all residuals for a specific cosmological parameter
        for i in testing_nums:  
            real_params = real_params_txt[i][:num_cosmo_params]  # First 5 real params for the ith sim
            samples_file = '/scratch/network/vk9342/nwLH_samples_nwLH_trained/nwLH_samples_nwLH_trained' + str(i)+'_cosmo_param_samples_200_'+str(k_max)+'.npy'

            if os.path.exists(samples_file):
                samples = np.load(samples_file)  # Should be 1000 x 5
                std = np.std(samples, axis=0)  # Std of those 1000 samples for one Pk (array length 5)
                mean = np.mean(samples, axis=0)  # Mean of those 1000 samples for one Pk (array length 5)
                
                if devide_by_std == False:
                    residual = [(real_params[n] - samples[j][n]) for j in range(num_samples)]  # residuals for one cosmo param of one file (length 1000)
                else:
                    residual = [(real_params[n] - samples[j][n])/std[n] for j in range(num_samples)]
                    
                residuals.append(residual)  #residuals of one cosmo param for all Pk files (length (num_Pk_foldersx1000))

                if n==0:
                    all_samples.append(samples)  # should be 1000 000 x 5
            else:
                print('File '+ samples_file+ ' not found')

        residuals  = np.hstack(residuals)
        z.append(residuals)  

    z = np.vstack(z) 
    z = np.array(z)  ## array of size (num Pk folders x 1000) x 5

    all_samples = np.vstack(all_samples)   
    all_samples = np.array(all_samples)
    std = np.std(all_samples, axis = 0)
    if devide_by_std == False:
        for i in range (num_cosmo_params):
            z[i] = z[i]/std[i]
    print(z.shape)

    return (z) # z should be 5 x (num folders x1000)


k_max = 0.1
z_1 = get_z_values(num_cosmo_params, num_samples, num_Pk_folders, k_max, real_params_txt)
k_max = 0.5
z_5 = get_z_values(num_cosmo_params, num_samples, num_Pk_folders, k_max, real_params_txt)


#### PLOT HISTORGRAM
lim = 10
fig, axes = plt.subplots(1, num_cosmo_params, figsize=(30, 5))  # 5 rows, 1 column
for i in range(num_cosmo_params):
    ax = axes[i]
    ax.hist(z_1[i], bins=30, density=True, histtype = 'step', color = 'tomato',
            label = 
            '$k_{max} = 0.1$ \n $\mu$ = '+ str(np.round(np.mean(z_1[i]), 5)) + 
            '\n $\sigma$ = '+ str(np.round(np.std(z_1[i]), 5)) 
            ) #range=(-lim, lim)
   
    ax.hist(z_5[i], bins=30, density=True, histtype = 'step', color = 'royalblue',
            label = 
            '$k_{max} = 0.5$ \n $\mu$ = '+ str(np.round(np.mean(z_5[i]), 5)) + 
            '\n $\sigma$ = '+ str(np.round(np.std(z_5[i]), 5)) 
            )
    #Gaussian
    mu_gauss, std_gauss = 0, 1
    x = np.linspace(-lim, lim, 1000)
    gaussian_curve = stats.norm.pdf(x, mu_gauss, std_gauss)
    ax.plot(x, gaussian_curve, '--', linewidth=1, color = 'black', alpha =0.7)
    ax.axline((0, 0), (0, 0.4), linewidth=1, color='grey')
    
    ax.set_title(param_names[i], size = 20)
    ax.set_xlabel('$z$', size = 20)
    axes[0].set_ylabel('normalazied counts', size = 20)
    ax.legend()
    ax.set_xlim(-5, 5)

    
       
# axes[4].set_xlim(-1.2, 1.2)
# axes[3].set_xlim(-4, 4)
# axes[2].set_xlim(-4, 4)
# axes[1].set_xlim(-4, 4)
# axes[0].set_xlim(-2.2, 2.2)
# axes[5].set_xlim(-5, 5)
# axes[6].set_xlim(-5, 5)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.suptitle ('nwLH trained and tested, devided by individual std', size = 20)

# plt.savefig('residuals_1000_nwLH_32000_0.1_0.5.png', dpi=300)
%matplotlib inline
plt.show()



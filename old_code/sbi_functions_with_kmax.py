import numpy as np
import sbi.inference
from sbi import utils as utils
from sbi import analysis as analysis
import torch
import sys, os
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython


def import_Pks (file_path, file_name, num_folders, k_max):
    Pk = []
    k = []
    for i in range(num_folders):
        k_i, Pk_i = np.loadtxt(file_path+str(i)+file_name, usecols=(0, 1), unpack=True)
        Pk_filtered = []
        for j in range (len(Pk_i)):
            if k_i[j] <= k_max:
                Pk_filtered.append(Pk_i[j])
        Pk.append(Pk_filtered)   
    Pk = np.array(Pk)
    Pk = torch.tensor(Pk, dtype=torch.float32)
    return Pk ## returns array of Pks for given kmax


def import_cosmo_params (file_path, num_folders):
    cosmo_params = np.loadtxt(file_path)
    cosmo_params = cosmo_params[:num_folders, :]
    cosmo_params = torch.tensor(cosmo_params, dtype=torch.float32)
    return cosmo_params



def train (cosmo_params, Pk_data, posterior_file_name):
   
    num_dim = cosmo_params.shape[1]  
    min_vals = torch.min(cosmo_params, dim=0).values
    max_vals = torch.max(cosmo_params, dim=0).values
    
    prior = utils.BoxUniform(low=min_vals, high=max_vals)
    inference = sbi.inference.SNPE(prior=prior)
    _ = inference.append_simulations(cosmo_params, Pk_data) 
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    
    torch.save(posterior, posterior_file_name)
    
    return posterior 


def infer_cosmological_parameters(posterior, test_Pk):
    samples = posterior.sample((1000,), x=test_Pk)
    return samples




def sample (posterior, test_Pk_file_path, num_Pk_files, k_max):
    cosmo_param_samples =[]
    Pk = []
    for i in range (num_Pk_files):
        test_file = test_Pk_file_path + str(i)+'/Pk_m_z=0.txt'
        test_k, test_Pk = np.loadtxt(test_file, unpack=True)
        test_Pk = torch.tensor(test_Pk, dtype=torch.float32)
        filtered_Pk = []
        for j in range (len(test_k)):
            if test_k[j] <= k_max:
                filtered_Pk.append(test_Pk[j])  ## array of Pks from one Pk file
        cosmo_param_sample = infer_cosmological_parameters(posterior, filtered_Pk) #cosmo params from one Pk file (1000 x 5 params)
        cosmo_param_samples.append(cosmo_param_sample.numpy()) ## cosmo params for each Pk file 
    cosmo_param_samples = np.vstack(cosmo_param_samples) 
    return (cosmo_param_samples)  ## returns array of size num_Pk_file x 1000 (samples 1000 from posterior) by 5 (5 cosmo params)





def corner_plot (real_params, all_cosmo_param_samples, plot_title, colors, line_colors, label):
    
    param_names = ['\Omega_m', '\Omega_b', 'h', 'n_s', '\sigma_8']
    names = param_names
    labels =  param_names
    all_samples = []  
    for i in range (len(all_cosmo_param_samples)): #i = number of diff posteriors 
        MCS_samples = MCSamples(samples=all_cosmo_param_samples[i],names = names, labels = labels, label = label[i])
        all_samples.append (MCS_samples)  ##array of samples of each posterior sample to be plotted
    
    # Triangle plot
    g = plots.get_subplot_plotter()
    g.triangle_plot(all_samples, filled=True, colors= colors, title_limit=1,
                    markers={'\Omega_m':float(real_params[0]), '\Omega_b':float(real_params[1]), 
                             'h':float(real_params[2]), 'n_s':float(real_params[3]), 
                             '\sigma_8':float(real_params[4])},
                   line_args=[{'color': color} for color in line_colors])
    
    for i, (name, value) in enumerate(zip(param_names, real_params)):
        ax = g.subplots[i, i]
        ax.annotate(str(value), xy=(0.1, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=10)

    plt.suptitle(plot_title)
    return (g)

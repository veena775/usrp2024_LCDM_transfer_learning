import numpy as np
import sbi.inference
from sbi import utils as utils
from sbi import analysis as analysis
import torch
# %matplotlib inline
%config InlineBackend.figure_format = 'retina'
import sys, os
# sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython



posterior = torch.load('posterior.pth')
 
test_k, test_Pk = np.loadtxt('/scratch/network/vk9342/latin_hypercube/0/Pk_m_z=0.txt', unpack=True)
test_Pk = torch.tensor(test_Pk, dtype=torch.float32) # Convert the test power spectrum data to a tensor

def infer_cosmological_parameters(posterior, test_Pk):
    samples = posterior.sample((1000,), x=test_Pk)   # Sample from the posterior distribution
    return samples

cosmo_param_samples = infer_cosmological_parameters(posterior, test_Pk)



#DEFAULT SBI PLOT
mean_params = torch.mean(cosmo_param_samples, dim=0)
std_params = torch.std(cosmo_param_samples, dim=0)
param_names = ['Omega_m ', 'Omega_b', 'h', 'n_s', 'sigma_8']
formatted_labels = [f"{name}\n${mean:.2f}_{{-{std:.2f}}}^{{+{std:.2f}}}$" for name, mean, std in zip(param_names, mean_params, std_params)]

_ = analysis.pairplot(
    cosmo_param_samples,
    labels=formatted_labels,
)


#CORNER PLOT 2

cosmo_param_samples = cosmo_param_samples.numpy()
param_names = ['\Omega_m ', '\Omega_b', 'h', 'n_s', '\sigma_8']

# Get the getdist MCSamples objects for the samples, specifying same parameter
# names and labels; if not specified weights are assumed to all be unity
names = param_names
labels =  param_names
samples = MCSamples(samples=cosmo_param_samples,names = names, labels = labels)

# Triangle plot
g = plots.get_subplot_plotter()
g.triangle_plot([samples], filled=True, colors=['green'])

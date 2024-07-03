import numpy as np
import sbi.inference
from sbi import utils as utils
from sbi import analysis as analysis
import torch

cosmo_params = np.loadtxt('latin_hypercube_params.txt')  #file with cosmological parameters
cosmo_params = torch.tensor(cosmo_params, dtype=torch.float32)

Pk = []
for i in range(2000):
    k_i, Pk_i = np.loadtxt('/scratch/network/vk9342/latin_hypercube/'+str(i)+'/Pk_m_z=0.txt', unpack=True)#files with power spectra
    Pk.append(Pk_i) 
Pk = np.array(Pk)
Pk = torch.tensor(Pk, dtype=torch.float32)  #power spectra data

num_dim = cosmo_params.shape[1]  
min_vals = torch.min(cosmo_params, dim=0).values
max_vals = torch.max(cosmo_params, dim=0).values


prior = utils.BoxUniform(low=min_vals, high=max_vals)

inference = sbi.inference.SNPE(prior=prior)

_ = inference.append_simulations(cosmo_params, Pk) 

density_estimator = inference.train()

posterior = inference.build_posterior(density_estimator)

torch.save(posterior, 'posterior.pth')
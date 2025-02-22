import numpy as np
import sbi.inference
from sbi import utils as utils
from sbi import analysis as analysis
import torch
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython
from sbi_functions import *
import random




####################### Sample random training and testing files #######################
# min_n = 0
# max_n = 2000
# total_n = int(0.9 * 2000)
# print(total_n)
# training_nums = random.sample(range(min_n, max_n), total_n)
# np.save('training_nums_nwLH', training_nums)

# testing_nums = []
# for i in range (2000):
#     if i not in random_nums:
#         testing_nums.append (i)
# np.save('testing_nums_nwLH', testing_nums)




####################### IMPORT PKS #######################
k_max = 0.1
training_nums = np.load('training_nums_nwLH.npy')
num_folders = training_nums   ## randomly drawn indecies 
file_path = '/scratch/network/vk9342/latin_hypercube_nwLH/'
file_name = '/Pk_m_z=0.txt'


def import_Pk (file_path, file_name, i, k_max):
    k, Pk = np.loadtxt(file_path+str(i)+file_name, unpack=True) #load file of index i
    Pk_filtered = []
    for j in range (len(Pk)):  #loop through Pk values from one Pk file
        if k[j] <= k_max:
            Pk_filtered.append(Pk[j])  #get Pks with corresponding k values less than k_max
    return Pk_filtered ## return filtered Pk values 

Pk = []
k = []
for i in (training_nums):
    Pk_filtered = import_Pk (file_path, file_name, i, k_max)  #Pks from one Pk file
    Pk.append(Pk_filtered)   
Pk = np.array(Pk)   #Pks from all Pk files
Pk = torch.tensor(Pk, dtype=torch.float32)
print(Pk.shape)


####################### IMPORT COSMO PARAMS #######################
cosmo_params_file = 'real_params/latin_hypercube_params_nwLH.txt'
all_cosmo_params = np.loadtxt(cosmo_params_file)

cosmo_params = []
for i in (training_nums):
    cosmo_params.append (all_cosmo_params[i])
cosmo_params = torch.tensor(cosmo_params, dtype=torch.float32)
print(cosmo_params.shape)


####################### TRAIN MODEL #######################
def train (cosmo_params, Pk, posterior_file_name):
    
    num_dim = cosmo_params.shape[1]  
    min_vals = torch.min(cosmo_params, dim=0).values
    max_vals = torch.max(cosmo_params, dim=0).values
    
    prior = utils.BoxUniform(low=min_vals, high=max_vals)
    inference = sbi.inference.SNPE(prior=prior)
    _ = inference.append_simulations(cosmo_params, Pk) 
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator, sample_with='mcmc', mcmc_method='nuts', max_sampling_batch_size=200)
    
    torch.save(posterior, posterior_file_name)
    
    return posterior

posterior_file_name = 'posterior_nwLH_1800_0.1.pth'
posterior = train(cosmo_params, Pk, posterior_file_name)
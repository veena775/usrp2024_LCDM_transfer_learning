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


"""
TRAIN

num_folders = 2000
k_max = 0.1
file_path = '/scratch/network/vk9342/latin_hypercube/'
Pk = import_Pks(file_path, num_folders, k_max)
cosmo_params_file = 'latin_hypercube_params.txt'
cosmo_paras = import_cosmo_params (file_path, num_folders)
posterior_file_name = 'posterior_2000_0.1.pth'
posterior = train(cosmo_params_file, posterior_file_name)
"""
k_max = 0.1
num_folders_lh = 2000
file_path_lh = '../Data/Pk/matter/latin_hypercube/'
file_name_lh = '/Pk_m_z=0.txt'
Pk_lh= import_Pks(file_path_lh, file_name_lh, num_folders_lh, k_max)

num_folders_BSQ = 30000
file_path_BSQ = '../Data/Pk/matter/BSQ/'
file_name_BSQ = '/Pk_CDM_z=0.000.dat'
Pk_BSQ = import_Pks(file_path_BSQ, file_name_BSQ, num_folders_BSQ, k_max)

Pk_data = torch.vstack((Pk_lh, Pk_BSQ))

cosmo_params_file_lh = 'latin_hypercube_params.txt'
cosmo_params_file_BSQ = 'BSQ_params.txt'
cosmo_params_lh = import_cosmo_params (cosmo_params_file_lh, num_folders_lh)
cosmo_params_BSQ = import_cosmo_params(cosmo_params_file_BSQ, num_folders_BSQ)

cosmo_params = torch.vstack((cosmo_params_lh, cosmo_params_BSQ))


posterior_file_name = 'posterior_32000_0.1.pth'
posterior = train(cosmo_params, Pk_data, posterior_file_name)


"""
SAMPLE
"""
posterior = torch.load(posterior_file_name)
test_Pk_file_path = '/scratch/network/vk9342/fiducial/'
num_Pk_folders = 100
k_max = 0.1
cosmo_param_samples = sample (posterior, test_Pk_file_path, num_Pk_folders, k_max)




"""
CORNER PLOTS
"""
all_cosmo_param_samples = [cosmo_param_samples_2000, cosmo_param_samples_32000]  
fig_name = 'fiducial_corner_plot_20000_32000_copy.png'
real_params = [0.3175, 0.049, 0.6711, 0.9624, 0.834]  #fiducial 
plot_title = ''
colors = ['green', ('#F7BAA6', '#E03424')]
line_colors =['green', 'red']
label = ['2000', '32000']
g = corner_plot(real_params, all_cosmo_param_samples, plot_title, colors, line_colors, label)
# plt.savefig(fig_name, dpi = 300)    
%matplotlib inline
plt.show()

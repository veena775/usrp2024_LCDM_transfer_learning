import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, f_Pk, f_Pk_norm, f_params, cosm_type, log, shuffle_all):  #make name of model and then if statement 

        # read data, scale it, and normalize it
        if log == False:
            Pk = np.load(f_Pk)
        else:
            Pk = np.log10(np.load(f_Pk))
        
        if f_Pk_norm is None:
            mean, std = np.mean(Pk, axis=0), np.std(Pk, axis=0)
        else:
            Pk_norm = np.log10(np.load(f_Pk_norm))
            mean, std = np.mean(Pk_norm, axis=0), np.std(Pk_norm, axis=0)
        Pk = (Pk - mean)/std
        
        # read the value of the cosmological & astrophysical parameters; normalize them 
        params  = np.loadtxt(f_params)
        if cosm_type == 'nwLH':
            params = params[:, :6]
            minimum = np.array([0.1, 0.02, 0.50, 0.80, 0.60, 0.01])
            maximum = np.array([0.5, 0.08, 0.90, 1.20, 1.00, 1.0])
        else:
            params = params[:, :5]
            minimum = np.array([0.1, 0.02, 0.50, 0.80, 0.60])
            maximum = np.array([0.5, 0.08, 0.90, 1.20, 1.00])
           
        params  = (params - minimum)/(maximum - minimum)

        
        # get the size and offset depending on the type of dataset
        sims = Pk.shape[0]
        if   mode=='train':  size, offset = int(sims*0.70), int(sims*0.00)
        elif mode=='valid':  size, offset = int(sims*0.15), int(sims*0.70)
        elif mode=='test':   size, offset = int(sims*0.15), int(sims*0.85)
        elif mode=='all':    size, offset = int(sims*1.00), int(sims*0.00)
        else:                raise Exception('Wrong name!')

        ## randomly shuffle the sims. Instead of 0 1 2 3...999 have a 
        ## random permutation. E.g. 5 9 0 29...342
        if shuffle_all == True:
            np.random.seed(seed)
            indexes = np.arange(sims) #only shuffle realizations, not rotations
            np.random.shuffle(indexes)
            indexes = indexes[offset:offset+size] #select indexes of mode
    
            # select the data in the considered mode
            Pk     = Pk[indexes]
            params = params[indexes]
        

        ## define size, input and output matrices
        self.size   = size
        self.input  = torch.tensor(Pk,     dtype=torch.float)
        self.output = torch.tensor(params, dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
# mode ---------------> 'train', 'valid', 'test' or 'all'
# seed ---------------> random seed to split data among training, validation and testing
# f_Pk ---------------> file containing the power spectra
# f_Pk_norm ----------> file containing the power spectra to normalize data
# f_params -----------> files with the value of the cosmological + astrophysical params
# batch_size ---------> batch size
# shuffle ------------> whether to shuffle the data or not
# workers --------> number of CPUs to load the data in parallel
def create_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, batch_size, shuffle, workers, cosm_type, log, shuffle_all):
    data_set = make_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, cosm_type, log, shuffle_all)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers)
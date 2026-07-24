import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
import optuna


class Objective(object):
    def __init__(self, input_size, output_size, max_neurons_layers, device,
                 epochs, seed, batch_size): #maybe remove max neurons

        self.input_size         = input_size  #power spectrum size
        self.output_size        = output_size  #n cosmo params 
        self.max_neurons_layers = max_neurons_layers #max n neurons per layer
        self.device             = device  #gpu or cpu
        self.epochs             = epochs  #n epochs
        self.seed               = seed
        self.batch_size         = batch_size
        self.mother = '/scratch/network/vk9342/USRP2024_scratch/pytorch/MPk_nwLH/test_MPk/'
        os.makedirs(self.mother+'losses', exist_ok=1)
        os.makedirs(self.mother+'models', exist_ok=1)

    def __call__(self, trial):

        # name of the files that will contain the losses and model weights
        # loss values are written to a file
        # These are hyperparameters being tuned by the optimizer.
        fout   = self.mother+'losses/loss_%d.txt'%(trial.number)
        fmodel = self.mother+'models/model_%d.pt'%(trial.number)


        # get the weight decay and learning rate values
        #Optuna suggests values for the learning rate and weight decay
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e0,  log=True)

        h1 = trial.suggest_int("h1", 10, 32)
        dr = trial.suggest_float("dr", 0, 0.5,  log=False)


        # generate the architecture
        model = architecture.model_1hl(self.input_size, h1, self.output_size, 
                            dr).to(self.device)

        # define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                      weight_decay=wd)      
        # define loss function
        # Mean Squared Error
        criterion = nn.MSELoss() 

        # get the data
        train_loader = data.create_dataset('train', self.seed, f_Pk, f_Pk_norm, 
                                           f_params, self.batch_size, shuffle=True, workers=1)
        valid_loader = data.create_dataset('valid', self.seed, f_Pk, f_Pk_norm, 
                                           f_params, self.batch_size, shuffle=False, workers=1)

        # train/validate model
        min_valid = 1e40
        # loop through number of epochs
        for epoch in range(self.epochs):

            # TRAINING
                #computed loss
                #backpropagation
                #optimizer updates weights
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_NN = model(x)
                loss = criterion(y_NN, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # VALIDATION
                # prevents overfitting?
                #loss calculated on validation set
                #validation loss normalized
            valid_loss, points = 0.0, 0
            model.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_NN = model(x)
                    valid_loss += (criterion(y_NN, y).item())*x.shape[0]
                    points     += x.shape[0]
            valid_loss /= points

            #if validation loss is lowest, weights saved
            if valid_loss<min_valid:  
                min_valid = valid_loss
                torch.save(model.state_dict(), fmodel)

            #save results
            f = open(fout, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, valid_loss, min_valid))
            f.close()

            # Handle pruning based on the intermediate value
            # comment out these lines if using prunning
            #trial.report(min_valid, epoch)
            #if trial.should_prune():  raise optuna.exceptions.TrialPruned()

        return min_valid

##################################### INPUT ##########################################
# data parameters
f_Pk      = 'all_MPk_nwLH.npy'
f_Pk_norm = None
f_params  = '../real_params/latin_hypercube_params_nwLH.txt' 
seed      = 1

# architecture parameters
input_size         = 79
output_size        = 7
max_neurons_layers = None

# training parameters
batch_size = 32
epochs     = 300  #100

# optuna parameters
study_name       = 'MPk_nwLH_params_test_1hl_300'
n_trials         = 100
storage          = 'sqlite:///nwLH.db'
n_jobs           = 1
n_startup_trials = 20 #random sample the space before using the sampler
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# define the optuna study and optimize it
objective = Objective(input_size, output_size, max_neurons_layers, 
                      device, epochs, seed, batch_size)
sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
if study_name in optuna.study.get_all_study_names(storage=storage):
        optuna.delete_study(study_name=study_name, storage=storage)   # fixme remove in general, but for rusty just one run run for now
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage,
                            load_if_exists=False)
study.optimize(objective, n_trials, n_jobs=n_jobs)







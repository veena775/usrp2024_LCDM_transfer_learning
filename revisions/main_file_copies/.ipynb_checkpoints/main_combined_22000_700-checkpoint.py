import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data_combined, architecture
import optuna
from torch.utils.data import random_split, DataLoader
import argparse


class Objective(object):
    def __init__(self, input_size, output_size, max_layers, max_neurons_layers, device, 
                 epochs, seed, batch_size, final_hidden_layer_size, log, model_type, max_epochs_no_improvement, n_sims):
        
        self.input_size         = input_size  #power spectrum size
        self.output_size        = output_size  #n cosmo params 
        self.max_layers         = max_layers  # max hidden layers = 3
        self.max_neurons_layers = max_neurons_layers #max n neurons per layer
        self.device             = device  #gpu or cpu
        self.epochs             = epochs  #n epochs
        self.seed               = seed
        self.batch_size         = batch_size
        self.max_epochs_no_improvement = max_epochs_no_improvement  # Early stopping patience
        self.last_hyperparams = None  # To store last set of hyperparameters
        self.no_change_count = 0  # Count how many trials have the same hyperparams
        self.final_hidden_layer_size = final_hidden_layer_size
        self.log = log
        self.max_epochs_no_improvement = max_epochs_no_improvement
        self.model_type = model_type
        self.mother = '/scratch/network/vk9342/USRP2024_scratch/pytorch/'+str(Pk_type)+'_'+str(cosm_type)+'/'+str(name)+'/' 
        self.n_sims = n_sims
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
        wd = trial.suggest_float("wd", 1e-8, 1e0, log=True)

        n_layers = trial.suggest_int("n_layers", 1, max_layers)
        p = trial.suggest_float("dropout_l", 0.2, 0.8)
        
        # # generate the architecture
        # model = architecture.dynamic_model_fixed_final(trial, self.input_size, 
        #                                    self.output_size, final_hidden_layer_size,
        #                                    n_layers, self.max_neurons_layers).to(self.device)

        if model_type == 'dynamic_fixed_final':
            out_fs = [trial.suggest_int( f"n_units_l{i}", 4, max_neurons_layers) for i in range(n_layers-1)]
            model = architecture.dynamic_model_fixed_final(trial, self.input_size, 
                                                           self.output_size, final_hidden_layer_size,
                                                           n_layers, p, out_fs, self.max_neurons_layers
                                                          ).to(self.device)
        elif model_type == 'dynamic':
            out_fs = [trial.suggest_int( f"n_units_l{i}", 4, max_neurons_layers) for i in range(n_layers)]
            model = architecture.dynamic_model(trial, self.input_size, 
                                               self.output_size,
                                               n_layers, p, out_fs,
                                               self.max_neurons_layers
                                              ).to(self.device)  
        else:
            print('not a valid model')
        
        
        # define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                      weight_decay=wd)
        # define loss function
        # Mean Squared Error
        criterion = nn.MSELoss() 
        
        
        # get the data
        train_loader = data_combined.create_dataset('train', self.seed, f_Pk_train, f_Pk_norm, 
                                           f_params_train, self.batch_size, shuffle=False, 
                                           workers=1, cosm_type = cosm_type_data_range, log=log)
        
        valid_loader = data_combined.create_dataset('valid', self.seed, f_Pk_valid, f_Pk_norm, 
                                           f_params_valid, self.batch_size, shuffle=False,
                                           workers=1, cosm_type = cosm_type_data_range, log=log)


        # Early stopping variables
        best_valid_loss = float('inf')
        epochs_no_improvement = 0

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

            # Early stopping logic
            tolerance = 1e-6
            if  (best_valid_loss - valid_loss) < tolerance:
                epochs_no_improvement += 1
                
            else:
                best_valid_loss = valid_loss
                epochs_no_improvement = 0
                # torch.save(model.state_dict(), fmodel)
                

            #save results
            f = open(fout, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, valid_loss, min_valid))
            f.close()

            if epochs_no_improvement >= self.max_epochs_no_improvement:
                print(f"Early stopping at epoch {epoch+1} with validation loss {valid_loss:.5e}")
                break


            # Handle pruning based on the intermediate value
            # comment out these lines if using prunning
            trial.report(min_valid, epoch)
            if trial.should_prune():  raise optuna.exceptions.TrialPruned()

        return min_valid


##################################### INPUT ##########################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n_sims",
        type=int,
        default=200,
        help="Number of training samples to draw from the full train set",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    n_sims = args.n_sims



    # data parameters
    cosm_type = 'EQ' #'nwLH' or 'LH' or 'LC'
    cosm_type_data_range = 'EQ'
    Pk_type = 'Pk'
    additional_extension = '_combined_train_val_revisions'
    n_sims_BSQ = 22000
    log = True
    input_size         = 79
    final_hidden_layer_size = 'None'
    model_type = 'dynamic'
    
    name = 'transfer10_network2_'+str(n_sims)+'_'+cosm_type+'_'+str(n_sims_BSQ)+'_BSQ' + additional_extension
    study_name  = str(Pk_type)+'_'+str(cosm_type)+'_params_'+str(name)   #+str(epochs)

    if cosm_type == 'nwLH':
        output_size = 6
    elif cosm_type == 'nwLH_BSQ_combined':
        output_size = 6
    elif cosm_type == 'EQ':
        output_size = 6
    elif cosm_type == 'LC':
        output_size = 6
    elif cosm_type == 'fR':
        output_size = 7
    else:
        print('Error cosm type')

    if cosm_type == 'fR':
        params_ext = '_lcdm' ## for fR '' otheriwse
    else:
        params_ext = ''
    

    f_Pk_train      = 'Pk_files/'+Pk_type+'_'+ str(n_sims) +'_'+ cosm_type +'_'+ str(n_sims_BSQ)+'_BSQ_combined_train.npy'
    f_params_train  =    'real_params/'+Pk_type+'_'+str(n_sims)+'_'+ cosm_type+'_'+ str(n_sims_BSQ)+'_BSQ_combined_params_train.txt'

    f_Pk_valid      = 'Pk_files/Pk_'+str(n_sims)+'_'+cosm_type+'_'+str(n_sims_BSQ)+'_BSQ_combined_valid.npy'
    f_params_valid  = 'real_params/Pk_'+str(n_sims)+'_'+cosm_type+'_'+str(n_sims_BSQ)+'_BSQ_combined_params_valid.txt'


    
    
    f_Pk_norm = None
    seed      = 42
    # architecture parameters
    max_layers = 3
    max_neurons_layers = 500  #None
    max_epochs_no_improvement=50
    
    
    # training parameters
    batch_size = 32 #16 from neutrino and 16 from not neutrino - inbalanced data
    epochs     = 1000  #100
    
    # optuna parameters
    # study_name       = 'Pk_nwLH_params_dynamic'   #+str(epochs)
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
    objective = Objective(input_size, output_size, max_layers, max_neurons_layers, 
                          device, epochs, seed, batch_size, final_hidden_layer_size, 
                          log, model_type, max_epochs_no_improvement, n_sims)
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    if study_name in optuna.study.get_all_study_names(storage=storage):
            optuna.delete_study(study_name=study_name, storage=storage)   # fixme remove in general, but for rusty just one run run for now
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage,
                                load_if_exists=False)
    study.optimize(objective, n_trials, n_jobs=n_jobs)
    
    
    

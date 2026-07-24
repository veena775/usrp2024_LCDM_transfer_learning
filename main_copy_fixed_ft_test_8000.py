import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
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
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-1, log=True)

        # generate architecture
        if model_type == 'dynamic_fixed_final':
            model = architecture.dynamic_model_fixed_final(trial, self.input_size, 
                                                           self.output_size, final_hidden_layer_size,
                                                           BSQ_n_layers, BSQ_p, BSQ_out_fs, self.max_neurons_layers
                                                          ).to(self.device)
        elif model_type == 'dynamic':
            model = architecture.dynamic_model(trial, self.input_size, 
                                               self.output_size,
                                               BSQ_n_layers, BSQ_p, BSQ_out_fs,
                                               self.max_neurons_layers
                                              ).to(self.device)  
            

        BSQ_mother = '/scratch/network/vk9342/USRP2024_scratch/pytorch/Pk_BSQ/'+str(BSQ_name)+'/models/'
        BSQ_model_path = BSQ_mother +'model_%d.pt'%BSQ_best_trial.number
        state_dict = torch.load(BSQ_model_path, map_location=torch.device(device))
        
        output_layer_index = 3 * BSQ_n_layers
        weight_key = f"{output_layer_index}.weight"
        bias_key   = f"{output_layer_index}.bias"
        state_dict.pop(weight_key, None)
        state_dict.pop(bias_key, None)
        model.load_state_dict(state_dict, strict=False)
        
        # define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                      weight_decay=wd)
        criterion = nn.MSELoss() 
        
        
        # get the data
        full_train_dataset = data.create_dataset('train', self.seed, f_Pk, f_Pk_norm, 
                                           f_params, self.batch_size, shuffle=False, 
                                           workers=1, cosm_type = cosm_type, log=log, 
                                           shuffle_all=True).dataset
        train_subset, _ = torch.utils.data.random_split(full_train_dataset,
                                                        [self.n_sims, len(full_train_dataset) - self.n_sims],
                                                        generator=torch.Generator().manual_seed(self.seed)
                                                       )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        valid_loader = data.create_dataset('valid', self.seed, f_Pk, f_Pk_norm, 
                                           f_params, self.batch_size, shuffle=False,
                                           workers=1, cosm_type = cosm_type, log=log, 
                                           shuffle_all = True)


        # Early stopping variables
        best_valid_loss = float('inf')
        epochs_no_improvement = 0

        # train/validate model
        min_valid = 1e40
        for epoch in range(self.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_NN = model(x)
                loss = criterion(y_NN, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # VALIDATION
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

    n_trials         = 100
    storage          = 'sqlite:///nwLH.db'
    n_jobs           = 1
    n_startup_trials = 20 #random sample the space before using the sampler

    # data parameters
    cosm_type = 'fR' #'nwLH' or 'LH' or 'LC'
    Pk_type = 'Pk'
    n_sims_BSQ = 8000
    log = True
    final_hidden_layer_size = 10
    model_type = 'dynamic'
    input_size = 79
    # additional_extension = '_fixed_ft_test'
    additional_extension = '_node_v2'
    
    name = 'transfer10_network2_'+str(n_sims)+'_'+cosm_type+'_'+str(n_sims_BSQ)+'_BSQ' + additional_extension
    study_name  = str(Pk_type)+'_'+str(cosm_type)+'_params_'+str(name)   #+str(epochs)

    ################### OPTUNA BEST STUDY LOAD #################
    if cosm_type == 'fR':
        add_ext = '_fR'
    else:
        add_ext = ''
    BSQ_name = 'transfer10_network1_'+str(n_sims_BSQ)+'_BSQ'+additional_extension+add_ext
    BSQ_study_name = 'Pk_BSQ_params_'+BSQ_name
    BSQ_study = optuna.load_study(study_name=BSQ_study_name, storage=storage)
    values = np.zeros(len(BSQ_study.trials))
    completed = 0
    for i,t in enumerate(BSQ_study.trials):
        values[i] = t.value
        if t.value is not None:  completed += 1
    indexes = np.argsort(values)
    for i in [0]:  #choose the best-model here, e.g. [0], or [1]
        BSQ_best_trial = BSQ_study.trials[indexes[i]]
        BSQ_lr       = BSQ_best_trial.params['lr']
        BSQ_wd       = BSQ_best_trial.params['wd']
        BSQ_n_layers       = BSQ_best_trial.params['n_layers']
        BSQ_p       = BSQ_best_trial.params['dropout_l']
        if model_type == 'dynamic_fixed_final':
            BSQ_out_fs = [BSQ_best_trial.params[f'n_units_l{i}'] for i in range(BSQ_n_layers - 1)]
            BSQ_out_fhl = 10
        elif model_type == 'dynamic':
            BSQ_out_fs = [BSQ_best_trial.params[f'n_units_l{i}'] for i in range(BSQ_n_layers)]
        


    ###########################################################
    
    
    if cosm_type == 'nwLH':
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
    

    f_Pk      = 'Pk_files/'+'all_'+str(Pk_type)+'_'+str(cosm_type)+params_ext+'.npy'
    f_params  = '../real_params/'+'all_' + str(cosm_type)+'_params'+params_ext+'.txt' 
    
    f_Pk_norm = None
    seed      = 42
    # architecture parameters
    max_layers = 3
    max_neurons_layers = 500  #None
    max_epochs_no_improvement=50
    
    
    # training parameters
    batch_size = 32 #16 from neutrino and 16 from not neutrino - inbalanced data
    epochs     = 1000  #100
    

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
    
    
    

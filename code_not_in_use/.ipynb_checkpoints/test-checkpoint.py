import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
import optuna

################################### INPUT ############################################
# data parameters
f_Pk_norm = None #file with Pk to normalize Pk
seed      = 1                                 #seed to split data in train/valid/test
mode      = 'valid'   #'train','valid','test' or 'all'


if type == 'nwLH':
    f_Pk      = 'all_nwLH_Pks.npy'        #file with Pk
    f_params  = '../real_params/latin_hypercube_params_nwLH.txt'       #file with parameters
    study_name = 'Pk_nwLH_params_dynamic'    # optuna parameters
    fout = 'Results_valid_Pk_nwLH_params_dynamic.txt'      # output file name
    mother = '/scratch/network/vk9342/USRP2024/pytorch/Pk_nwLH/dynamic/models/'
    output_size = 7    #dimensions of output data
    

elif type == 'LH':
    f_Pk      = 'all_LH_Pks.npy'        #file with Pk
    f_params  = '../real_params/latin_hypercube_params.txt'       #file with parameters
    study_name = 'Pk_LH_params_dynamic'    # optuna parameters
    fout = 'Results_valid_Pk_LH_params_dynamic.txt'      # output file name
    mother = '/scratch/network/vk9342/USRP2024/pytorch/Pk_LH/dynamic/models/'
    output_size = 5    #dimensions of output data
    


# architecture parameters
input_size  = 79   #dimensions of input data


# training parameters
batch_size = 32

# optuna parameters
storage    = 'sqlite:///nwLH.db'

######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# load the optuna study
study = optuna.load_study(study_name=study_name, storage=storage)

# get the scores of the study trials
values = np.zeros(len(study.trials))
completed = 0
for i,t in enumerate(study.trials):
    values[i] = t.value
    if t.value is not None:  completed += 1

# get the info of the best trial
indexes = np.argsort(values)
for i in [0]:  #choose the best-model here, e.g. [0], or [1]
    trial = study.trials[indexes[i]]
    print("\nTrial number {}".format(trial.number))
    print("Value: %.5e"%trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    lr       = trial.params['lr']
    wd       = trial.params['wd']
    h1       = trial.params['h1']
    dr       = trial.params['dr']
    fmodel = mother +'model_%d.pt'%trial.number

# generate the architecture
model = architecture.model_1hl(input_size, h1, output_size, dr)
model.to(device)    

# load best-model, if it exists
if os.path.exists(fmodel):  
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
else:
    raise Exception('model doesnt exists!!!')

# define loss function
criterion = nn.MSELoss() 

# get the data
test_loader = data.create_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, 
                                  batch_size, shuffle=False, workers=1)
test_points = 0
for x,y in test_loader:  test_points += x.shape[0]

# define the arrays containing the true and predicted value of the parameters
params  = output_size
results = np.zeros((test_points, 2*params), dtype=np.float32)

# test the model
test_loss, points = 0.0, 0
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        bs   = x.shape[0]  #batch size
        x, y = x.to(device), y.to(device)
        y_NN = model(x)
        test_loss += (criterion(y_NN, y).item())*bs
        results[points:points+bs,0*params:1*params] = y.cpu().numpy()
        results[points:points+bs,1*params:2*params] = y_NN.cpu().numpy()
        points    += bs
test_loss /= points
print('Test loss:', test_loss)

# denormalize results here
#
#

# save results to file
np.savetxt(fout, results)
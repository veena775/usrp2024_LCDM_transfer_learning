import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna

######## 1 hidden layer ##########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_1hl(nn.Module):
    
    def __init__(self, inp, h1, out, dr):
        super(model_1hl, self).__init__()

        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.fc2(out)         
        return out
##################################



######## 1 hidden layer with sigmoid ##########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_1hl_sigmoid(nn.Module):
    
    def __init__(self, inp, h1, out, dr):
        super(model_1hl_sigmoid, self).__init__()

        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.fc2(out) 
        out = self.sigmoid(out)
        return out
##################################

######## 2 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_2hl_sigmoid(nn.Module):
    
    def __init__(self, inp, h1, h2, out, dr):
        super(model_2hl_sigmoid, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

        self.sigmoid = nn.Sigmoid()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.fc3(out)  
        out = self.sigmoid(out)
        return out
##################################

####################################################################
####################################################################

# This routine returns an architecture that is built inside the routine itself
# It can have from 1 to max_layers hidden layers. The user specifies the size of the
# input and output together with the maximum number of neurons in each layers
# trial -------------> optuna variable
# input_size --------> size of the input
# output_size -------> size of the output
# max_layers --------> maximum number of hidden layers to consider (default=3)
# max_neurons_layer -> the maximum number of neurons a layer can have (default=500)
def dynamic_model(trial, input_size, output_size, n_layers, p, out_fs, max_neurons_layers=500): #nlayers

    # define the tuple containing the different layers
    layers = []
    in_features = input_size
    for i in range(n_layers):
        out_features = out_fs[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(p))
        in_features = out_features

    # get the last layer
    layers.append(nn.Linear(out_features, output_size))
    layers.append(nn.Sigmoid())

    # return the model
    return nn.Sequential(*layers)



def dynamic_model_fixed_final(trial, input_size, output_size, final_hidden_layer_size, n_layers, p, out_fs, max_neurons_layers=500):
    
    # define the tuple containing the different layers
    layers = []
    
    # get the hidden layers
    in_features = input_size
    for i in range(n_layers - 1):  # One fewer loop to leave space for the final hidden layer
        out_features = out_fs[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(p))
        in_features = out_features

    # Add the final hidden layer
    layers.append(nn.Linear(in_features, final_hidden_layer_size))
    layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.Dropout(p))

    # Add the output layer
    layers.append(nn.Linear(final_hidden_layer_size, output_size))
    layers.append(nn.Sigmoid())

    # return the model
    return nn.Sequential(*layers)


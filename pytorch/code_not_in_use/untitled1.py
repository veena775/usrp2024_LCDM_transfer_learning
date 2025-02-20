import os
import torch
import torch.nn as nn
import data, architecture


class TransferLearningPipeline:
    def __init__(self, device, input_size, output_size, max_layers, max_neurons_layers, 
                 pretrained_path, save_path, batch_size, epochs, lr, wd):
        
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.pretrained_path = pretrained_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.wd = wd

        # Output directories
        self.mother = '/scratch/network/vk9342/USRP2024/transfer_learning/'
        os.makedirs(self.mother + 'models', exist_ok=True)
        self.fmodel = self.mother + f'model_transfer_learning.pt'

    def load_pretrained_model(self):
        pretrained_model = architecture.dynamic_model(None, self.input_size, 5, 
                                                      self.max_layers, self.max_neurons_layers).to(self.device)
        pretrained_model.load_state_dict(torch.load(self.pretrained_path))
        for param in pretrained_model.parameters():
            param.requires_grad = False
        return nn.Sequential(*list(pretrained_model.children())[:-1])  # Remove output layer

    def inference_network(self, input_size, output_size):
        # Define the new inference network
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)
        ).to(self.device)

    def train(self, feature_extractor, inference_net, train_loader, valid_loader):
        model = nn.Sequential(
            feature_extractor,
            inference_net
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(inference_net.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.MSELoss()

        best_valid_loss = float('inf')

        for epoch in range(self.epochs):
            # Training
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            valid_loss, points = 0.0, 0
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = model(x)
                    valid_loss += criterion(y_pred, y).item() * x.shape[0]
                    points += x.shape[0]
            valid_loss /= points

            # Save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), self.fmodel)

            print(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {valid_loss:.5e}")
        print("Training complete. Best validation loss:", best_valid_loss)


##################################### INPUT ##########################################
# Data parameters
cosm_type = 'nwLH'  # 'nwLH' or 'LH'
if cosm_type == 'nwLH':
    f_Pk = 'all_Pk_nwLH.npy'
    f_params = '../real_params/latin_hypercube_params_nwLH.txt'
    output_size = 7
elif cosm_type == 'LH':
    f_Pk = 'all_Pk_LH.npy'
    f_params = '../real_params/latin_hypercube_params.txt'
    output_size = 5
else:
    print('specify type')
    sys.exit()

f_Pk_norm = None
seed = 1

# Architecture parameters
input_size = 79
max_layers = 3
max_neurons_layers = 500

# Training parameters
batch_size = 32
epochs = 100
lr = 1e-4
wd = 1e-6

# Pretrained model path
pretrained_path = '/path/to/pretrained/LH_model.pt'
save_path = 'transfer_learning_model.pt'

# Use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print("CUDA Not Available")
    device = torch.device('cpu')

# Load datasets
train_loader = data.create_dataset('train', seed, f_Pk, f_Pk_norm, f_params, 
                                   batch_size, shuffle=True, workers=1, cosm_type=cosm_type)
valid_loader = data.create_dataset('valid', seed, f_Pk, f_Pk_norm, f_params, 
                                   batch_size, shuffle=False, workers=1, cosm_type=cosm_type)

# Initialize and run the pipeline
pipeline = TransferLearningPipeline(device, input_size, output_size, max_layers, 
                                    max_neurons_layers, pretrained_path, save_path, 
                                    batch_size, epochs, lr, wd)

feature_extractor = pipeline.load_pretrained_model()
inference_net = pipeline.inference_network(feature_extractor[-3].out_features, output_size)
pipeline.train(feature_extractor, inference_net, train_loader, valid_loader)

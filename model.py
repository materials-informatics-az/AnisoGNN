import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, global_mean_pool, MessagePassing, aggr, SAGEConv
import time
from utils import *
from tqdm import tqdm

'''
This script trains and evaluates Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric.
It includes utility functions for setting random seeds, defines custom message-passing convolution layers,
and implements two GNN classes: a simple GNN using custom convolutions and a more complex SAGE model.
The main script handles data loading, model initialization, training, testing, saving models,
and visualizing the results. The script is structured to support various configurations and datasets,
ensuring comprehensive evaluation of the models.
'''

def seed_all(seed):
    '''
    Set random seeds for reproducability
    '''
    if not seed:
        seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyConv(MessagePassing):
    def __init__(self, N1, N2):
        super().__init__(aggr='mean')
        self.lin = Linear(N1, N2)
    def forward(self, x, edge_index):
        x = F.relu(self.lin(x))
        x = self.propagate(edge_index, x=x)
        return x

class GNN(torch.nn.Module):
    '''
    Graph Neural Network
    '''
    def __init__(self, N, N_fl):
        super(GNN, self).__init__()
        self.conv = MyConv(N, N_fl)
        self.out = Linear(N_fl, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x= self.conv(x,edge_index)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x

class SAGE(torch.nn.Module):
    '''
    Graph Neural Network
    '''
    def __init__(self, N_fl1, N_mpl, N_fl2, N_fl3):
        super(SAGE, self).__init__()
        self.pre = Linear(5, N_fl1)
        self.conv1 = SAGEConv(N_fl1, N_mpl, normalize=True)
        self.conv2 = SAGEConv(N_mpl, N_mpl, normalize=True)
        self.post1 = Linear(N_mpl, N_fl2)
        self.post2 = Linear(N_fl2, N_fl3)
        self.out = Linear(N_fl3, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Pre Processing Linear Layer
        x = F.relu(self.pre(x))
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 2. Readout layer
        x = global_mean_pool(x, batch)
        # 3. Apply Fully Connected Layers
        x = F.relu(self.post1(x))
        x = F.relu(self.post2(x))
        x = self.out(x)
        return x

def init_model_Aniso(N, N_fl):
    '''
    Initialize model
    '''
    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(N, N_fl).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=l_rate, weight_decay=w_decay)
    return model, optimizer

def init_model_SAGE(N_fl1, N_mpl, N_fl2, N_fl3):
    '''
    Initialize model
    '''
    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(N_fl1, N_mpl, N_fl2, N_fl3).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=l_rate, weight_decay=w_decay)
    return model, optimizer

def train(model, optimizer, train_loader, val_loader, n_epoch):
    '''
    Train GNN
    '''
    filename = f'{output_dir}/case-{case}_eval-{eval}_loss_history.txt'
    output = open(filename, "w")

    print('Epoch Training_MSE Validation_MSE', file=output, flush=True)
    seed_all(42)
    
    for epoch in tqdm(range(n_epoch)):
        model.train()
        # Train batches
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            train_pred = model(train_batch)
            train_true = getattr(train_batch, 'y')
            train_loss = F.mse_loss(train_pred, train_true)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluate
        val_pred, val_true = test(model, val_loader)
        val_loss = F.mse_loss(val_pred, val_true)
        print(f'{epoch:d}, {train_loss:e}, {val_loss:e}', file=output, flush=True)
    return epoch

def test(model, data_loader):
    '''
    Test GNN
    '''
    seed_all(42)
    model.eval()
    data = next(iter(data_loader)).to(device)
    pred = model(data)
    true = getattr(data, 'y')
    return pred, true


parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default = 1)
parser.add_argument('--output_type', type=str, default='png')
parser.add_argument('--config', type=int, default = 0)

args = parser.parse_args()
output_type = args.output_type
case = args.case
config = args.config

cases = ['O-SAGE_Ni_E' , 'C-Aniso_Ni_E', 'O-SAGE_Al_E' , 'C-Aniso_Al_E' , 'O-SAGE_Al_YS' , 'S-Aniso_Al_YS']
case = cases[case-1]

config_dir = './config/'
output_dir = './output/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

config_name = config_dir + str(config) + '.json'
with open(config_name, 'r') as h:
    params = json.load(h)

l_rate = params['l_rate']
w_decay = params['w_decay']
b_size = params['b_size']

if case[0] == 'O':
    n_epoch = params['n_epoch_SAGE']
    N_fl1 = params['N_fl1']
    N_mpl = params['N_mpl']
    N_fl2 = params['N_fl2']
    N_fl3 = params['N_fl3']
elif case[0] == 'C':
    n_epoch = params['n_epoch_Aniso']
    N_C = params['N_C']
    N_C_fl = params['N_C_fl']
else:
    n_epoch = params['n_epoch_Aniso']
    N_S = params['N_S']
    N_S_fl = params['N_S_fl']

# Set seeds for complete reproducability
seed_all(42)

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

evals = ['test_45_90_deg', 'test_45_deg', 'test_ran_30%', 'test_90_deg']

scaler=scalers(case) # load scalers

for eval in evals:
    train_loader, test_loader = get_data(eval, b_size, case)
    # Define model and optimizer
    if case[0] == 'O':
        model, optimizer = init_model_SAGE(N_fl1, N_mpl, N_fl2, N_fl3)
    elif case[0] == 'C':
        model, optimizer = init_model_Aniso(N_C, N_C_fl)
    else:
        model, optimizer = init_model_Aniso(N_S, N_S_fl)
    
    print('\n====== Configuration ======')
    print(f'Case:\t{case}')
    print(f'Evaluation:\t{eval}')

# *************************************************************************** #
    print('\n====== Training / Testing ======')
    start = time.time()
    
    # Train model
    train(model, optimizer, train_loader, test_loader,
          n_epoch)
    # Test Model
    preds_test, trues_test = test(model, test_loader)
    
    # Save model
    torch.save(
        model, f"{output_dir}/case-{case}_eval-{eval}_checkpoint.pth")
    
    print(f'Processing time: {time.time()-start:.2f} seconds')
    # *************************************************************************** #
    # Report and Visualize predictions
    preds_test = scaler['y'].inverse_transform(
        preds_test.detach().detach().cpu().numpy())
    trues_test = scaler['y'].inverse_transform(
        trues_test.detach().detach().cpu().numpy())
    meanARE, maxARE = mean_maxARE(preds_test, trues_test)
    
    print(f'(MeanARE, MaxARE):\t({meanARE}, {maxARE})')
    
    train_loader, test_loader= get_data(eval, 300, case)
    
    preds, trues = test(model, train_loader) # Training set
    
    preds = scaler['y'].inverse_transform(
        preds.detach().detach().cpu().numpy())
    trues = scaler['y'].inverse_transform(
        trues.detach().detach().cpu().numpy())
    
    plot_results(preds, trues,case,eval,preds_test,trues_test,output_type,output_dir)

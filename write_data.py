import glob
import pickle
import re

import h5py
import networkx as nx
import numpy as np
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

'''
This script processes raw data and prepares it for training Graph Neural Networks (GNNs).
It includes functions for natural sorting and handles data loading, feature extraction, and scaling.
The script processes data from HDF5 files for different material cases (e.g., Ni and Al) and textures,
constructs graphs, scales features and properties, and saves the processed data and scalers.
The processed data is saved in a format suitable for training GNNs, ensuring that the features and
properties are properly normalized and ready for model input.
'''

def natural_sort(l):
    """Sort list in Natural order.

    Parameters
    ----------
    l : {list}
        List of strings.

    Returns
    -------
    l : {ndarray}
        Sorted list of strings.
    """
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


Ni_raw_data = h5py.File('Ni_data.hdf5', mode='r')
Al_raw_data = h5py.File('Al_data.hdf5', mode='r')

cases = ['O-SAGE_Ni_E' , 'O-SAGE_Al_E' , 'O-SAGE_Al_YS' , 'C-Aniso_Ni_E' , 'C-Aniso_Al_E' , 'S-Aniso_Al_YS']
textures = ['comp','uni','shear','psc','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90','comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']

for case in tqdm(cases):
    prop_data = np.empty((0, 1))
    graph = case[0]+"-"+case.split("_")[1]
    if case[0] == "O":
        feature_data = torch.zeros((0, 5))
    elif case[0] == "C":
        feature_data = torch.zeros((0, 22))
    else:
        feature_data = torch.zeros((0, 109))

    if case.split("_")[2] == "E":
        prop_type = "E_modulus"
    else:
        prop_type = "yield_strength"

    for texture in textures:
        class_name = 'equi_%s' %texture
        graph_files = glob.glob(f'./graphs/{graph}/{class_name}/*')
        graph_files = natural_sort(graph_files)
        if case.split("_")[1] == "Ni":
            prop=np.expand_dims(Ni_raw_data[texture]['_prop'][prop_type][()],1)
        else:
            prop=np.expand_dims(Al_raw_data[texture]['_prop'][prop_type][()],1)
        prop_data= np.concatenate([prop_data, prop])
        
        for i, file in enumerate(graph_files):
            G = nx.read_gpickle(file)
            data = torch_geometric.utils.from_networkx(G)
            feature_data = torch.cat([feature_data, data.x])

    # Scale data
    y_scaler = StandardScaler()
    prop = y_scaler.fit_transform(prop_data)
    prop = torch.from_numpy(prop)
    prop = prop.to(torch.float32)
    
    x_scaler = StandardScaler()
    x_scaler.fit(feature_data.numpy())
    
    # Save scalers
    with open(f'./graph_data/{case}/y_scaler.pickle', 'wb') as handle:
        pickle.dump(y_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./graph_data/{case}/x_scaler.pickle', 'wb') as handle:
        pickle.dump(x_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for j, texture in enumerate(textures):
        datalist = []
        class_name = 'equi_%s' %texture
        graph_files = glob.glob(f'./graphs/{graph}/{class_name}/*')
        graph_files = natural_sort(graph_files)
    
        for i, file in enumerate(graph_files):
            G = nx.read_gpickle(file)
            data = torch_geometric.utils.from_networkx(G)
    
            # assign scaled features
            x = x_scaler.transform(data.x.numpy())
            x = torch.from_numpy(x)
            data.x = x.to(torch.float32)
            data.y = torch.unsqueeze(prop[i+25*j], 1)
            datalist.append(data)
    
        with open(f'./graph_data/{case}/{texture}.pickle', 'wb') as handle:
            pickle.dump(datalist, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
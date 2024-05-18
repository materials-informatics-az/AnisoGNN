import networkx as nx
import numpy as np
from tqdm import tqdm
import h5py
from utils import *

"""
This script reads data from HDF5 files containing grain orientations, 'C_tensor', and IDs for Nickel (Ni) and Aluminum (Al).
It then processes the data to create graphs representing grains and their properties using the NetworkX library.

For each texture and microstructural volume element (MVE), the script:
1. Extracts grain orientation, 'C_tensor', and grain ID data.
2. Uses neighboring grain data to structure the graphs.
3. Constructs graphs using grain features and neighboring information.
4. Saves the graphs into different directories based on the element and texture.

The graphs provide a structured representation of the material's microstructure, which can be further analyzed or visualized.
"""

n_MVEs = 25
Ni_raw_data = h5py.File('Ni_data.hdf5', mode='r')
Al_raw_data = h5py.File('Al_data.hdf5', mode='r')

textures = ['comp','uni','shear','psc','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90','comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']

# create graphs for Ni
for texture in tqdm(textures):
    class_name = 'equi_%s' %texture
    for i in range(1,n_MVEs+1):
        grain_oris=Ni_raw_data[texture]['{:02d}'.format(i)]['quaternions'][()]
        grain_C_tensor=Ni_raw_data[texture]['{:02d}'.format(i)]['C_tensor'][()]
        grain_ids= Ni_raw_data[texture]['{:02d}'.format(i)]['grain_id'][()]
        nbr_dict = get_nbrs(grain_ids, periodic=True)
        # Create graph
        G = nx.Graph()
        for feat, ori in enumerate(grain_oris):
            size = np.sum(grain_ids == feat+1)
            G.add_nodes_from(
                [(feat+1, {"x": np.hstack([ori, size])})]
            )
        for feat, (nbrs, areas) in nbr_dict.items():
            for nbr, area in zip(nbrs, areas):
                G.add_edge(feat, nbr, edge_weight=area)
        nx.write_gpickle(
            G, f'./graphs/O-Ni/{class_name}/{class_name}'+'_{:02d}.O'.format(i))
        
        G_C = nx.Graph()
        for feat, ori in enumerate(grain_C_tensor):
            size = np.sum(grain_ids == feat+1)
            G_C.add_nodes_from(
                [(feat+1, {"x": np.hstack([ori, size])})]
            )
        for feat, (nbrs, areas) in nbr_dict.items():
            for nbr, area in zip(nbrs, areas):
                G_C.add_edge(feat, nbr, edge_weight=area)
        nx.write_gpickle(
            G_C, f'./graphs/C-Ni/{class_name}/{class_name}'+'_{:02d}.C'.format(i))

# create graphs for Al
for texture in tqdm(textures):
    class_name = 'equi_%s' %texture
    for i in range(1,n_MVEs+1):
        grain_oris=Al_raw_data[texture]['{:02d}'.format(i)]['quaternions'][()]
        grain_C_tensor=Al_raw_data[texture]['{:02d}'.format(i)]['C_tensor'][()]
        grain_S_tensor=Al_raw_data[texture]['{:02d}'.format(i)]['S_tensor'][()]
        grain_ids= Al_raw_data[texture]['{:02d}'.format(i)]['grain_id'][()]
        nbr_dict = get_nbrs(grain_ids, periodic=True)
        # Create graph
        G = nx.Graph()
        for feat, ori in enumerate(grain_oris):
            size = np.sum(grain_ids == feat+1)
            G.add_nodes_from(
                [(feat+1, {"x": np.hstack([ori, size])})]
            )
        for feat, (nbrs, areas) in nbr_dict.items():
            for nbr, area in zip(nbrs, areas):
                G.add_edge(feat, nbr, edge_weight=area)
        nx.write_gpickle(
            G, f'./graphs/O-Al/{class_name}/{class_name}'+'_{:02d}.O'.format(i))
        
        G_C = nx.Graph()
        for feat, ori in enumerate(grain_C_tensor):
            size = np.sum(grain_ids == feat+1)
            G_C.add_nodes_from(
                [(feat+1, {"x": np.hstack([ori, size])})]
            )
        for feat, (nbrs, areas) in nbr_dict.items():
            for nbr, area in zip(nbrs, areas):
                G_C.add_edge(feat, nbr, edge_weight=area)
        nx.write_gpickle(
            G_C, f'./graphs/C-Al/{class_name}/{class_name}'+'_{:02d}.C'.format(i))

        G_S = nx.Graph()
        for feat, ori in enumerate(grain_S_tensor):
            size = np.sum(grain_ids == feat+1)
            G_S.add_nodes_from(
                [(feat+1, {"x": np.hstack([ori, size])})]
            )
        for feat, (nbrs, areas) in nbr_dict.items():
            for nbr, area in zip(nbrs, areas):
                G_S.add_edge(feat, nbr, edge_weight=area)
        nx.write_gpickle(
            G_S, f'./graphs/S-Al/{class_name}/{class_name}'+'_{:02d}.S'.format(i))
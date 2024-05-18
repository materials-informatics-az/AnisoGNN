# Aniso-GNN
 
## Setup
Install dependencies. Please check these links to install [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorchGeometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
```bash
conda create --name Aniso python=3.9
conda activate Aniso
pip install -r requirements.txt
```
## Dream3D scripts
run_equi.py: generate 25 MVEs as dream3d files based on equiaxed.json for each texture.

run_equi.m: create orientation data (in euler angles) as txt files based on ODF data of each texture.

yaml_gen.py: generate material.yaml for DAMASK analysis and create orientation data (in quaternions).

grain_ids.m: generate grain ids as numpy array.

C_tensor.m: create elasticity stiffness tensors and save 21 elements in txt files.

S_tensor.m: create Schmid tensors and save 108 elements in txt files.

## DAMASK scripts
tensionX_Al.yaml: loading file for Al, including 3 increments, elasticity deformation at the 2nd increment, yield point at the 3rd increment.

tensionX_Ni.yaml: loading file for Ni, including 2 increments, elasticity deformation at the 2nd increment.

post_processing.py: analyze DAMASK output hdf5 files and collect Young's modulus from the 2nd increment and yield strength from the 3rd increment

## HDF5 Files
Grain IDs, euler angles for each grain and the mechanical property of MVEs are stored in Ni_raw_data.hdf5 and Al_raw_data.hdf5. 

First layer: 12 textures, including
  Uniaxial Compression (0/45/90 degrees rotations),
  Uniaxial Tension (0/45/90 degrees rotations),
  Plane Strain Compression (0/45/90 degrees rotations),
  Simple Shear (0/45/90 degrees rotations);

Second layer: 25 MVEs per texture and arrays storing E_modulus and yield_strength of MVEs; 

Third layer: Grain IDs, euler angles, quaternions, C-tensor, S-tensor of each grain from each MVE. 

## Create Graphs
Microstructure graphs for all 300 MVEs have already been created in the graphs folder. The method to create those graph files (this process may take more than 20h on regular CPU, multi-processing is recommended):
```bash
python create_graphs.py
```

## Prepare Data
PyTorch datalists for graphs and MVE mechanical properties have been created as pickle files and saved in graph_data folder. The method to create those pickle files:
```bash
python write_data.py
```

## Run Model
The different cases presented in the original paper are:
  1. O-SAGE_Ni_E
  2. C-Aniso_Ni_E
  3. O-SAGE_Al_E
  4. C-Aniso_Al_E
  5. O-SAGE_Al_YS
  6. S-Aniso_Al_YS
  
For each case, four evaluations are performed and loss histories, parity plots, and model checkpoints are outputted.
Four evaluations:
  1. test_45_90_deg
  2. test_45_deg
  3. test_ran_30%
  4. test_90_deg

```
Usage: python model.py [OPTIONS]
Options:
  --cases INT                  Case number (e.g., 1 - 6, default: 1)
  --config INT                 Hyper-parameter configuration number (default: 0)
  --output_type STR            Type of output figures (png/svg, default: png)
```

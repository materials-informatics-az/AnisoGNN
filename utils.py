import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch_geometric.loader import DataLoader


def make_pdc(micro):
    '''Pads 3D array with opposing faces.

    Parameters
    ----------
    micro : {ndarray}
        3D array of grain IDs.

    Returns
    -------
    micro : {ndarray}
        3D array of grain IDs with periodic faces.
    '''
    micro = np.pad(micro, 1)
    micro[0, :, :] = micro[-2, :, :]
    micro[-1, :, :] = micro[1, :, :]
    micro[:, 0, :] = micro[:, -2, :]
    micro[:, -1, :] = micro[:, 1, :]
    micro[:, :, 0] = micro[:, :, -2]
    micro[:, :, -1] = micro[:, :, 1]
    return micro


def get_nbrs(micro, periodic=True):
    '''Get neighbors of grains.

    Parameters
    ----------
    micro : {ndarray}
        3D array of grain IDs.
        *DONT USE grain ID == 0*

    Returns
    -------
    nbr_dict : {dictionary} {grain_ID : [nbrs, shared_area]}
        Dictionary giving neighbors and shared area of each neighbor for every grain.
    '''
    if periodic:
        micro = make_pdc(micro)
    else:
        micro = np.pad(micro, 1)

    dim = micro.shape

    # structure element used to get voxel face neighbors
    s = np.zeros((3, 3, 3))
    s[1, 1, :] = 1
    s[1, :, 1] = 1
    s[:, 1, 1] = 1
    s[1, 1, 1] = 0

    nbr_dict = {}
    for feat in np.unique(micro):
        if feat == 0:
            continue
        nbr_list = []
        for x in range(1, dim[0]-1):
            for y in range(1, dim[1]-1):
                for z in range(1, dim[2]-1):
                    if micro[x, y, z] == feat:
                        nbrs = s*micro[x-1:x+2, y-1:y+2, z-1:z+2]
                        nbrs = nbrs[~np.isin(nbrs, [0, feat])]
                        for nbr in nbrs:
                            nbr_list.append(nbr)

        nbrs, counts = np.unique(np.asarray(nbr_list), return_counts=True)
        nbr_dict[feat] = [nbrs.astype(int), counts]
    return nbr_dict


def mean_maxARE(y_pred, y_true):
    '''Calcuate Mean, MaxARE
    '''
    meanARE = np.around(100*np.mean((np.abs(y_pred-y_true)/y_true)), decimals=2)
    maxARE = np.around(100*np.max((np.abs(y_pred-y_true)/y_true)), decimals=2)
    return meanARE, maxARE


def seed_worker(worker_id):
    '''Seeding for DataLoaders
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(42)
    random.seed(42)


def get_data(eval, b_size, case):
    '''Prepare data depending on Eval Case

    Parameters
    ----------
    eval : {int}
        Range 1-4, indicating type of evalution/experiment

    b_size : {int}
        Batch size

    Returns
    -------
    train_loader : {DataLoader}
    test_loader : {DataLoader} - For test or validation data
    '''
    g = torch.Generator()
    g.manual_seed(42)

    train_data = []
    test_data = []
    all_data=[]

    if eval == 'test_45_90_deg':
        for texture in ['comp','uni','shear','psc']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    elif eval == 'test_45_deg':
        for texture in ['comp','uni','shear','psc','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    elif eval == 'test_ran_30%':
        np.random.seed(42)
        random.seed(42)
        rand_ids = np.random.choice(300, 300, replace=False)
        for texture in ['comp','uni','shear','psc','comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90','comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']:
                with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                    data = pickle.load(handle)
                    all_data += data
        
        datalist_new = [all_data[i] for i in rand_ids] # shuffle
        # 70/30 split
        train_data=datalist_new[90:] 
        test_data=datalist_new[:90]

    elif eval == 'test_90_deg':
        for texture in ['comp','uni','shear','psc','comp_rot_z-45','uni_rot_z-45','shear_rot_z-45','psc_rot_z-45']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['comp_rot_z-90','uni_rot_z-90','shear_rot_z-90','psc_rot_z-90']:
            with open(f'./graph_data/{case}/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    # Data Loaders
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True, worker_init_fn=seed_worker,
                              generator=g)
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False, worker_init_fn=seed_worker,
        generator=g)

    return train_loader, test_loader

def scalers(case):
    # Re-scaling Predictions
    with open(f'./graph_data/{case}/y_scaler.pickle', 'rb') as handle:
        y_scaler = pickle.load(handle)
    with open(f'./graph_data/{case}/x_scaler.pickle', 'rb') as handle:
        x_scaler = pickle.load(handle)
    scaler = {}
    scaler['y'] = y_scaler
    scaler['x'] = x_scaler
    return scaler

def plot_results(preds, trues, case, eval, preds_test, trues_test, output_type,output_dir):
    '''Plot evaluation results
    '''
    if case.split('_')[-1]=='YS':
        trues_plot=trues_test/1e6
        preds_plot=preds_test/1e6
        trues_train=trues/1e6
        preds_train=preds/1e6
    else:
        trues_plot=trues_test/1e9
        preds_plot=preds_test/1e9
        trues_train=trues/1e9
        preds_train=preds/1e9
    fname=f"{output_dir}/{case}_{eval}"
    sns.set(font_scale=1.75)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    plt.scatter(trues_train,preds_train,s=50,color='gray', alpha=0.6,edgecolors='None')
    if eval=='test_45_90_deg':
        plt.scatter(trues_plot[:25],preds_plot[:25],s=150,color='#52B5F9', label ='Compression 45',alpha=0.7,marker="^")
        plt.scatter(trues_plot[25:50],preds_plot[25:50],s=150,color='#DF873E', label ='Tension 45',alpha=0.7,marker="^")
        plt.scatter(trues_plot[50:75],preds_plot[50:75],s=150,color='#69E374', label ='Shear 45',alpha=0.7,marker="^")
        plt.scatter(trues_plot[75:100],preds_plot[75:100],s=150,color='#EA334B', label ='PSC 45',alpha=0.7,marker="^")
    
        plt.scatter(trues_plot[100:125],preds_plot[100:125],s=150,color='#52B5F9', label ='Compression 90',alpha=0.7,marker="+")
        plt.scatter(trues_plot[125:150],preds_plot[125:150],s=150,color='#DF873E', label ='Tension 90',alpha=0.7,marker="+")
        plt.scatter(trues_plot[150:175],preds_plot[150:175],s=150,color='#69E374', label ='Shear 90',alpha=0.7,marker="+")
        plt.scatter(trues_plot[175:],preds_plot[175:],s=150,color='#EA334B', label ='PSC 90',alpha=0.7,marker="+")
    elif eval=='test_45_deg':
        plt.scatter(trues_plot[:25],preds_plot[:25],s=150,color='#52B5F9', label ='Compression 45',alpha=0.7,marker="^")
        plt.scatter(trues_plot[25:50],preds_plot[25:50],s=150,color='#DF873E', label ='Tension 45',alpha=0.7,marker="^")
        plt.scatter(trues_plot[50:75],preds_plot[50:75],s=150,color='#69E374', label ='Shear 45',alpha=0.7,marker="^")
        plt.scatter(trues_plot[75:],preds_plot[75:],s=150,color='#EA334B', label ='PSC 45',alpha=0.7,marker="^")
    elif eval=='test_90_deg':
        plt.scatter(trues_plot[:25],preds_plot[:25],s=150,color='#52B5F9', label ='Compression 90',alpha=0.7,marker="+")
        plt.scatter(trues_plot[25:50],preds_plot[25:50],s=150,color='#DF873E', label ='Tension 90',alpha=0.7,marker="+")
        plt.scatter(trues_plot[50:75],preds_plot[50:75],s=150,color='#69E374', label ='Shear 90',alpha=0.7,marker="+")
        plt.scatter(trues_plot[75:],preds_plot[75:],s=150,color='#EA334B', label ='PSC 90',alpha=0.7,marker="+")
    elif eval=='test_ran_30%':
        np.random.seed(42)
        rand_ids = np.random.choice(300, 300, replace=False)
        for i in range(90):
            if rand_ids[i] in range(300)[:25]:
                p1=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#52B5F9',alpha=0.7, label ='Compression')
            if rand_ids[i] in range(300)[25:50]:
                p2=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#DF873E',alpha=0.7, label ='Tension')
            if rand_ids[i] in range(300)[50:75]:
                p3=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#69E374',alpha=0.7, label ='Shear')
            if rand_ids[i] in range(300)[75:100]:
                p4=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#EA334B',alpha=0.7, label ='PSC')
            if rand_ids[i] in range(300)[100:125]:
                p5=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#52B5F9',alpha=0.7, label ='Compression 90',marker="+")
            if rand_ids[i] in range(300)[125:150]:
                p6=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#DF873E',alpha=0.7, label ='Tension 90',marker="+")
            if rand_ids[i] in range(300)[150:175]:
                p7=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#69E374',alpha=0.7, label ='Shear 90',marker="+")
            if rand_ids[i] in range(300)[175:200]:
                p8=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#EA334B',alpha=0.7, label ='PSC 90',marker="+")
            if rand_ids[i] in range(300)[200:225]:
                p9=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#52B5F9',alpha=0.7, label ='Compression 45',marker="^")
            if rand_ids[i] in range(300)[225:250]:
                p10=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#DF873E',alpha=0.7, label ='Tension 45',marker="^")
            if rand_ids[i] in range(300)[250:275]:
                p11=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#69E374',alpha=0.7, label ='Shear 45',marker="^")
            if rand_ids[i] in range(300)[275:]:
                p12=ax.scatter(trues_plot[i],preds_plot[i],s=150,color='#EA334B',alpha=0.7, label ='PSC 45',marker="^")
        # l = ax.legend([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], 
        #       ['Compression', 'Tension','Shear','PSC','Compression 90','Tension 90','Shear 90','PSC 90','Compression 45','Tension 45','Shear 45','PSC 45'],
        #       loc ="lower right",fontsize=10)
    if case.split('_')[2] == 'YS':
        plt.xlabel('True strength (MPa)')
        plt.ylabel('Predicted strength (MPa)')
        plt.xlim([74, 90])
        plt.ylim([74, 90])
        plt.xticks(np.arange(75, 95, 5))
        plt.yticks(np.arange(75, 95, 5))
        plt.plot([74, 90], [74, 90], 'gray', linewidth=1, zorder=1)
    elif case.split('_')[1]=='Ni':
        plt.xlabel('True modulus (GPa)')
        plt.ylabel('Predicted modulus (GPa)')
        plt.xlim([200, 280])
        plt.ylim([200, 280])
        plt.xticks(np.arange(200, 300, 20))
        plt.yticks(np.arange(200, 300, 20))
        plt.plot([200, 280], [200, 280], 'gray', linewidth=1, zorder=1)
    else:
        plt.xlabel('True modulus (GPa)')
        plt.ylabel('Predicted modulus (GPa)')
        plt.xlim([75, 82])
        plt.ylim([75, 82])
        plt.xticks(np.arange(76, 84, 2))
        plt.yticks(np.arange(76, 84, 2))
        plt.plot([75, 82], [75, 82], 'gray', linewidth=1, zorder=1)
        
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    if output_type == 'png':
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    if output_type == 'svg':
        plt.savefig(fname+".svg", dpi=300, bbox_inches="tight")
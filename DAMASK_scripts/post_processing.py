import damask
import numpy as np
import glob
import os
import math

def Mises(what,tensor):
	''' Calulate von Mises value for stress or strain tensor.
	Source: DAMASK (http://damask.mpie.de).'''

	# Get deviatoric tensor
	dev = tensor - np.trace(tensor)/3.0*np.eye(3)

	# Symmetrize
	symdev = 0.5*(dev+dev.T)

	# Get Mises value
	return math.sqrt(np.sum(symdev*symdev.T)*
      {
       'stress': 3.0/2.0,
       'strain': 2.0/3.0,
       }[what.lower()])

result_file =glob.glob('./*.hdf5')[0]
result = damask.Result(result_file)
result.add_stress_Cauchy()
result.add_strain()
result.add_equivalent_Mises('sigma')
result.add_equivalent_Mises('epsilon_V^0.0(F)')

epsilon_ave,sigma_ave=[],[]
re=result.get(['sigma','epsilon_V^0.0(F)'])
for inc in re.keys():
    epsilon_voxels=np.empty((0,3,3))
    sigma_voxels=np.empty((0,3,3))
    for i in re[inc].keys():
        epsilon=re[inc][i]['epsilon_V^0.0(F)']
        sigma=re[inc][i]['sigma']
        epsilon_voxels=np.concatenate((epsilon,epsilon_voxels),0)
        sigma_voxels=np.concatenate((sigma,sigma_voxels),0)
    epsilon_ave.append(Mises("strain",np.average(epsilon_voxels,0)))
    sigma_ave.append(Mises("stress",np.average(sigma_voxels,0)))

cwd = os.getcwd()
foldername=os.path.basename(cwd)

all=[epsilon_ave]+[sigma_ave]

# if Al, yield strength is obtained from the third increment (inc)
np.savetxt(foldername+'_out_sig_epi.txt',all)

# calculate Young's modulus at the second increment (inc)
modulus=(sigma_ave[1]-sigma_ave[0])/(epsilon_ave[1]-epsilon_ave[0])
print('modulus: '+str(modulus))

with open(foldername+'_out.txt', 'w') as f:
    f.write(str(modulus))

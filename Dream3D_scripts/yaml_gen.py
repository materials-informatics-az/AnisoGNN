import damask
import numpy as np

"""
This script processes material data in `.dream3d` format and saves it in other formats for further analysis.

Key Steps:
1. Loads a sample material configuration and modifies it to remove plastic behavior.
2. Iterates through different textures and MVEs, reading data from input files.
3. Applies rotations and configurations to generate new material data in the desired formats.
"""

n_MVEs= 25
textures = ['comp','uni','shear','psc']
paths=[]

path_example='./master/'
example=damask.ConfigMaterial.load(path_example+'material.yaml')
del example['phase']['Aluminum']['mechanical']['plastic'] # delete plastic part
qu_0=damask.Rotation.from_Euler_angles([0,0,0],degrees=True)

for tex in textures: 
    class_name = 'equi_%s' %tex
    dream3d_output='path_to_save/%s/' %class_name
    
    for i in range(1,n_MVEs+1):
        newpath = dream3d_output+class_name+'_{:02d}/'.format(i)
        grid=damask.Grid.load_DREAM3D(newpath+class_name+'_{:02d}.dream3d'.format(i),feature_IDs='FeatureIds')
        config=damask.ConfigMaterial.load_DREAM3D(newpath+class_name+'_{:02d}.dream3d'.format(i))

        grid.save(newpath+class_name+'_{:02d}.vti'.format(i))

        text=np.loadtxt(newpath+class_name+'_{:02d}.txt'.format(i),skiprows=1)

        n_grains = len(text)

        qutext=damask.Rotation.from_Euler_angles(text,degrees=True)
        np.savetxt(newpath+class_name+'_qu_{:02d}.txt'.format(i),qutext)
                
        del config['phase']['Primary']

        material=[]

        for ii in range(n_grains+1):
            ph_name = 'grain_{:03d}'.format(ii)
            config['phase'][ph_name] = example['phase']['Aluminum']
            material.append(ph_name)

        config_new = damask.ConfigMaterial().material_add(O=qu_0,phase='grain_000',homogenization='direct')
        config_new = config_new.material_add(O=qutext,phase=material[1:],homogenization='direct')

        config['homogenization']['direct']=example['homogenization']['SX']
        config['material']=config_new['material']

        config.save(newpath+'material.yaml')
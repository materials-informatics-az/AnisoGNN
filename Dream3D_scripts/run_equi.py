import json
import os
from subprocess import call
"""
This script automates the process of generating `.dream3d` files for different textures and microstructural volume elements (MVEs).

Key Steps:
1. Defines the number of MVEs and lists the textures to process.
2. Prepares the necessary folders and subdirectories for saving the results.
3. For each MVE:
   - Reads a JSON template and updates the output file path.
   - Saves the updated JSON to a new file.
   - Runs the pipeline executable with the generated JSON file to create the `.dream3d` files.
"""
n_MVEs= 25
textures = ['comp','uni','shear','psc']

for tex in textures: 
    class_name = 'equi_%s' %tex
    mainfolder='path_to_save'
    dream3_doutput='path_to_save/%s/' %class_name
    path_dream3d='dream3d_path'

    if not os.path.exists(dream3_doutput):
            os.makedirs(dream3_doutput)

    for i in range(1,n_MVEs+1):
        newpath = dream3_doutput+class_name+'_{:02d}/'.format(i)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        f= open(mainfolder+'equiaxed.json')
        data = json.load(f)

        data['7']['OutputFile'] = newpath+class_name+'_{:02d}.dream3d'.format(i)

        newfile=newpath+class_name+'_{:02d}.json'.format(i)

        with open(newfile, 'w') as f:
            json.dump(data, f)

        runPipe=path_dream3d+'PipelineRunner.exe -p '+newpath+class_name+'_{:02d}.json'.format(i)
        call(runPipe)

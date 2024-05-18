%startup_mtex
% This script processes texture data to assign orientations to grains in synthetic microstructures.

% Key Steps:
% 1. Defines paths to input ODF data and initializes texture types and the number of MVEs.
% 2. Iterates over each texture type to set up the file paths for saving results.
% 3. Loads orientation distribution function (ODF) data from .mat files.
% 4. For each MVE, reads the number of grains from the .dream3d files.
% 5. Uses the ODF to sample orientations for the specified number of grains.
% 6. Saves the assigned orientations to .txt files for further analysis.

% The script helps in generating synthetic microstructures with specific texture properties.

path_odf="./textures/";
textures = ["comp","uni","shear","psc"];
n_MVEs= 25;
for i =1:length(textures)
    tex=textures(i);
    class_name=sprintf("equi_%s", tex);
    path_dream3d="path_to_save"+class_name+"/";

    data = load(path_odf+tex+".mat"); %load odf
    odf=data.odf;
    for j=1:n_MVEs
        newpath=path_dream3d+sprintf(class_name+"_%02d/", j);
        data_h5read=h5read(newpath+sprintf(class_name+"_%02d.dream3d",j),"/DataContainers/SyntheticVolumeDataContainer/CellEnsembleData/NumFeatures");

        num_grains=data_h5read(2);%number of grains
        ori = odf.discreteSample(num_grains);
        fname = newpath+sprintf(class_name+"_%02d.txt",j);
        export(ori,fname);
    end
end

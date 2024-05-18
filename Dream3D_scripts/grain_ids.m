%This script handles the reading and processing of grain ID data from .dream3d files for different textures and microstructural volume elements (MVEs). It:

%Defines the number of MVEs and a list of textures to process.
%For each texture, it generates a class_name and constructs a path for storing data.
%Iterates through each MVE, reads the grain IDs from the .dream3d files, and saves the grain ID data into .mat files for further analysis.
n_MVEs= 25;
textures = ["comp","uni","shear","psc"];
for i =1:length(textures)
    tex=textures(i);
    class_name=sprintf("equi_%s", tex);
    path_dream3d="path_to_save/"+class_name+"/";

    for j=1:n_MVEs
        newpath=path_dream3d+sprintf(class_name+"_%02d/", j);
        grain_id=h5read(newpath+sprintf(class_name+"_%02d.dream3d",j),"/DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds");
        fname = newpath+sprintf(class_name+"_%02d.mat",j);
        save(fname,'grain_id')

    end
end
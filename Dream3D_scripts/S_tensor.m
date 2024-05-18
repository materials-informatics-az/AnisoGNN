%This script calculates and saves the rotated S-tensor for different textures and microstructural volume elements (MVEs). The key steps include:

%Setting the number of MVEs and loading the crystal symmetry and precomputed S-tensor data.
%Iterating over various textures to set up the file paths for storing data.
%For each MVE, reading grain orientation data from input files, applying rotations to the S-tensor to align it with grain orientations, and saving the rotated tensor data to files for further analysis.
n_MVEs= 25;
cs = crystalSymmetry('m-3m');
sSAll = load("sSAll.mat").sSAll;

textures = ["comp","uni","shear","psc"];
for i =1:length(textures)
    tex=textures(i);
    class_name=sprintf("equi_%s", tex);
    path_dream3d="path_to_save/"+class_name+"/";
    for j =1:n_MVEs
        newpath=path_dream3d+sprintf(class_name+"_%02d/", j);
        ori_txt = newpath+sprintf(class_name+"_%02d.txt",j); % orientation data import

        data=importdata(ori_txt); 

        r = rotation.byEuler(data.data*degree);
        S=[];
        for k=1:length(r)
            Srot = rotate(ST,r(k)); % shape: 12*1(tensor)
            Sreshape=reshape(Srot.M,1,[]);
            S=[S;Sreshape];
        end
        fname = newpath+sprintf(class_name+"_S_%02d.txt",j);
        writematrix(S,fname,'Delimiter',' '); % save S-tensor elements
    end
end
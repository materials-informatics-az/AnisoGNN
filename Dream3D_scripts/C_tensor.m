%This script processes stiffness tensors for different textures and microstructural volume elements (MVEs). The key steps include:

%Defining the stiffness tensor matrix and specifying the crystal symmetry for cubic crystals.
%Iterating through a list of textures to set up file paths for saving data.
%For each MVE, importing grain orientation data, aligning the stiffness tensor to the grain orientations, and saving the processed tensor data for further analysis.

n_MVEs= 25;
M = [[267.1e+9   170.5e+9  170.5e+9   0     0     0];...
    [  170.5e+9  267.1e+9  170.5e+9   0     0     0];...
    [  170.5e+9   170.5e+9 267.1e+9   0     0     0];...
    [   0    0   0  107.6e+9     0     0];...
    [   0    0   0   0    107.6e+9     0];...
    [   0    0   0   0     0    107.6e+9]];

cs = crystalSymmetry('m-3m');

C = stiffnessTensor(M,cs);

m  = triu(true(size(C.Voigt)));

x = vector3d(1,0,0);

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
 
        Crot = rotate(C,r); % shape: 6*6*(num of grains)

        Crot_v=Crot.Voigt;

        mm=repmat(m,1,1,length(data.data)); % shape: 6*6*(num of grains)

        v=reshape(Crot_v(mm),21,length(data.data)); % save only the upper half diag 21 elements from matrix

        vv=v'; % switch columns and rows
        vv=vv/1e9;
        fname = newpath+sprintf(class_name+"_C_%02d.txt",j);
        writematrix(vv,fname,'Delimiter',' '); % save C-tensor elements

    end
end
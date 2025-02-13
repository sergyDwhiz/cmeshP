import nibabel as nib
import numpy as np

# path
curv_file_path = '/home/sergy/cortical-mesh-parcellation/10brainsurfaces (1)/100206/surf/lh_aligned.surf'

# read .H file using nibabel
curv_data = nib.freesurfer.read_morph_data(curv_file_path)

# debugging
print("Curvature data:")
print(curv_data)

# save the curvature data in a numpy array and save it as a .npy file
curvature_array = np.array(curv_data)
np.save('curvature_array.npy', curvature_array)

# debugging
loaded_curvature_array = np.load('curvature_array.npy')
print("Loaded curvature array:")
print(loaded_curvature_array)


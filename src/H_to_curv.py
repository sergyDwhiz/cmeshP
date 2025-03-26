import nibabel as nib
import numpy as np
import os

curv_file_path = '/home/sergy/cortical-mesh-parcellation/10brainsurfaces (1)/100206/surf/lh_aligned.surf'

if not os.path.exists(curv_file_path):
    raise FileNotFoundError(f"Curvature file not found: {curv_file_path}")

curv_data = nib.freesurfer.read_morph_data(curv_file_path)
print("Curvature data:")
print(curv_data)

curvature_array = np.array(curv_data)
np.save('curvature_array.npy', curvature_array)
loaded_curvature_array = np.load('curvature_array.npy')
print("Loaded curvature array:")
print(loaded_curvature_array)


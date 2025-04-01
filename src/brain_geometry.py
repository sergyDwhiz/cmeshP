"""
Module for processing and visualizing brain mesh geometry.
"""

import trimesh
import open3d as o3d
import polyscope as ps
import gpytoolbox as gpy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Update the surface file path for correct file location
surface_path = '/Users/sergiusnyah/cmeshP/10brainsurfaces/100206/surf/lh_aligned.surf'
vertices, faces = nib.freesurfer.read_geometry(surface_path)

mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_o3d])

ps.init()
ps_mesh = ps.register_surface_mesh("brain", vertices, faces)
ps_mesh.add_vector_quantity("per-face normals", gpy.per_face_normals(vertices, faces), defined_on="faces", enabled=True)
ps.show()

curvature = gpy.angle_defect(vertices, faces)
lower_percentile = np.percentile(curvature, 1)
upper_percentile = np.percentile(curvature, 99)
curvature_clipped = np.clip(curvature, lower_percentile, upper_percentile)
curvature_normalized = (curvature_clipped-lower_percentile) / (upper_percentile-lower_percentile)

color_map = plt.get_cmap('viridis')
curvature_colors = color_map(curvature_normalized)[:,:3]
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(curvature_colors)
mesh_o3d.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_o3d], window_name='Mesh with Curvature Colors')

plt.figure()
plt.hist(curvature_normalized, bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Normalized Curvature Values")
plt.xlabel("Normalized Curvature")
plt.ylabel("Frequency")
plt.show()


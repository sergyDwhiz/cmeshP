"""
Module for processing and visualizing brain mesh geometry.
Functions include mesh creation, point cloud conversion, per-face curvature computation,
and visualization with Open3D and Polyscope.
"""

import trimesh
import open3d as o3d
import polyscope as ps
import gpytoolbox as gpy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Load FreeSurfer surface and visualize mesh using various tools
vertices, faces = nib.freesurfer.read_geometry('/home/sergius/SGI Research/cortical-mesh-parcellation/10brainsurfaces (1)/100206/surf/lh_aligned.surf')
mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_o3d])
o3d.visualization.draw_geometries([mesh_o3d],
    window_name='Mesh with Normals',
    width=1800,
    height=1600,
    left=50,
    top=50,
    mesh_show_wireframe=True,
    mesh_show_back_face=True)
point_cloud = mesh_o3d.sample_points_uniformly(number_of_points=10000)
o3d.visualization.draw_geometries([point_cloud])
N = gpy.per_face_normals(vertices, faces)
ps.init()
ps_mesh = ps.register_surface_mesh("brain", vertices, faces)
ps_mesh.add_vector_quantity("per-face normals", N, defined_on="faces", enabled=True)
ps.show()
curvature = gpy.angle_defect(vertices, faces)
print("Raw Curvature Values:")
print(curvature)
print("Curvature min:", np.min(curvature))
print("Curvature max:", np.max(curvature))
lower_percentile = np.percentile(curvature, 1)
upper_percentile = np.percentile(curvature, 99)
curvature_clipped = np.clip(curvature, lower_percentile, upper_percentile)
curvature_normalized = (curvature_clipped-lower_percentile) / (upper_percentile-lower_percentile)
print("Normalized Curvature Values:")
print(curvature_normalized)
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


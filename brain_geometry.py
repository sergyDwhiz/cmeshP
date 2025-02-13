import trimesh
import open3d as o3d
import polyscope as ps
import gpytoolbox as gpy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# Load the FreeSurfer surface
vertices, faces = nib.freesurfer.read_geometry('/home/sergy/SGI Research/cortical-mesh-parcellation/10brainsurfaces (1)/100206/surf/lh_aligned.surf')

# Now vertices and faces contain the vertex coordinates and face indices of the mesh
# extract vertices (V) and faces (F)

# Create a trimesh object
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
# convert the vertices and faces to the format expected by open3d
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# compute vertex normals
mesh.compute_vertex_normals()

# visualize the mesh
o3d.visualization.draw_geometries([mesh])

# visualize the mesh with borders
o3d.visualization.draw_geometries(
    [mesh],
    window_name='Mesh with Normals',
    width=1800,
    height=1600,
    left=50,
    top=50,
    mesh_show_wireframe=True,  # show wireframe
    mesh_show_back_face=True
)

# convert the mesh to a point cloud
point_cloud = mesh.sample_points_uniformly(number_of_points=10000)

# point cloud
o3d.visualization.draw_geometries([point_cloud])

# normals
N = gpy.per_face_normals(vertices, faces)

# register the mesh in Polyscope
ps.init()
mesh = ps.register_surface_mesh("brain", vertices, faces)
mesh.add_vector_quantity("per-face normals", N, defined_on="faces", enabled=True)
ps.show()

# compute per-face curvature using gpytoolbox (angle defect)
curvature = gpy.angle_defect(vertices, faces)

# debugging
print("Raw Curvature Values:")
print(curvature)
print("Curvature min:", np.min(curvature))
print("Curvature max:", np.max(curvature))

# percentile-based normalization
lower_percentile = np.percentile(curvature, 1)
upper_percentile = np.percentile(curvature, 99)

# clipping the curvature values to the 1st and 99th percentiles, to diminish the effect of outliers
curvature_clipped = np.clip(curvature, lower_percentile, upper_percentile)

# normalize the clipped curvature values between 0 and 1
curvature_normalized = (curvature_clipped - lower_percentile) / (upper_percentile - lower_percentile)

# debugging
print("Normalized Curvature Values:")
print(curvature_normalized)

# select color map
color_map = plt.get_cmap('viridis')
curvature_colors = color_map(curvature_normalized)[:, :3]  # ignore alpha channel

# create Open3D mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.vertex_colors = o3d.utility.Vector3dVector(curvature_colors)  # apply colors to vertices

# compute normals to improve lighting in visualization
mesh.compute_vertex_normals()

# visualize the mesh with curvature coloring
o3d.visualization.draw_geometries([mesh], window_name='Mesh with Curvature Colors')

# visualize the normalized curvature values as a histogram
plt.figure()
plt.hist(curvature_normalized, bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Normalized Curvature Values")
plt.xlabel("Normalized Curvature")
plt.ylabel("Frequency")
plt.show()


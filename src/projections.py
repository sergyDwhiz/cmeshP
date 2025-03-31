"""
Module for visualizing 3D cortical meshes and extracting features via 2D projections.
Ensure that the custom model (MultiViewCNN) is available in trainCNN.
"""

import pyrender
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import polyscope as ps
import nibabel as nib
import os
import torch
from trainCNN import MultiViewCNN  # Ensure trainCNN is in the Python path or same directory

def visualize_mesh(mesh_path, save_dir, annotations_path, curvature_path, model_path):
    """
    Visualizes a FreeSurfer mesh via multiple 2D projections, extracts CNN features,
    and registers annotations and curvature in a Polyscope visualization.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load the 3D cortical mesh using nibabel
    vertices, faces = nib.freesurfer.read_geometry(mesh_path)

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Create a pyrender scene and add the mesh
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Define camera views and positions
    camera_views = [pyrender.PerspectiveCamera(yfov=np.pi/3.0) for _ in range(6)]
    camera_positions = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]

    # Load the trained model with error handling.
    model = MultiViewCNN()
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    # Render 2D projections and extract features
    for i in range(6):
        camera = camera_views[i]
        position = camera_positions[i]
        camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(camera_node)
        renderer = pyrender.OffscreenRenderer(640, 480)
        color, depth = renderer.render(scene)

        # Make a copy of the array to avoid negative strides
        color_copy = color.copy()

        # Convert the rendered image to a tensor
        image_tensor = torch.tensor(color_copy).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Extract features using the trained model
        with torch.no_grad():
            features = model(image_tensor)

        # Display the rendered image
        plt.imshow(color)
        plt.title(f'Camera {i+1} View')
        plt.savefig(f"{save_dir}/view_{i}.png")
        plt.close()

        # Remove the camera from the scene
        scene.remove_node(camera_node)

    # Load annotations and curvature data
    annotations_3d = nib.freesurfer.read_annot(annotations_path)[0]
    curvature_3d = np.load(curvature_path)

    # Ensure curvature_3d has the correct shape
    if curvature_3d.shape[0] != vertices.shape[0]:
        curvature_3d = curvature_3d[:vertices.shape[0]]

    # Debugging: Print shapes
    print(f"Vertices shape: {vertices.shape}")
    print(f"Annotations shape: {annotations_3d.shape}")
    print(f"Curvature shape: {curvature_3d.shape}")

    # Initialize Polyscope
    ps.init()

    # Register the 3D mesh with Polyscope
    ps_mesh = ps.register_surface_mesh("annotated_brain", vertices, faces)

    # Add the annotations and curvature as scalar quantities
    ps_mesh.add_scalar_quantity("annotations", annotations_3d, defined_on="vertices", cmap="viridis")
    ps_mesh.add_scalar_quantity("curvature", curvature_3d, defined_on="vertices", cmap="coolwarm")

    # Show the visualization
    ps.show()

if __name__ == "__main__":
    # Example usage
    mesh_path = '10brainsurfaces (1)/100206/surf/lh_aligned.surf'
    annotations_path = '10brainsurfaces (1)/100206/label/lh.annot'
    curvature_path = 'curvature_array.npy'
    model_path = 'results/model.pth'
    save_dir = 'results/images'

    visualize_mesh(mesh_path, save_dir, annotations_path, curvature_path, model_path)
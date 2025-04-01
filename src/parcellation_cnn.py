import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import polyscope as ps
from sklearn.cluster import KMeans  # NEW import for unsupervised segmentation

# Updated segmentation model that produces per-vertex predictions
class CorticalSegmentationCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CorticalSegmentationCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Final convolution outputs per-vertex class scores
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # x has shape (batch, 3, num_vertices)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.conv3(x)  # shape: (batch, num_classes, num_vertices)
        return logits

def load_cortical_mesh(mesh_path):
    # Load mesh vertices and faces using nibabel
    vertices, faces, _ = nib.freesurfer.read_geometry(mesh_path, read_metadata=True)
    # Reorder vertex columns to [x, y, z]
    vertices = vertices[:, [2, 0, 1]]
    return vertices, faces

def parcellate_mesh(mesh_path, model_path, num_classes=36):  # UPDATED: default num_classes set to 36
    vertices, faces = load_cortical_mesh(mesh_path)
    # Prepare input tensor with shape (1, 3, num_vertices)
    input_tensor = torch.tensor(vertices.T, dtype=torch.float32).unsqueeze(0)
    model = CorticalSegmentationCNN(num_classes=num_classes)  # UPDATED: uses provided num_classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)  # (1, num_classes, num_vertices)
        # For per-vertex segmentation, take argmax along channel dimension
        predicted = torch.argmax(logits, dim=1).squeeze(0).numpy()  # shape: (num_vertices,)
    return predicted, faces

def kmeans_segmentation(vertices, n_clusters=10):
    """Segment the mesh by clustering vertex coordinates."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vertices)
    return kmeans.labels_

# NEW: Advanced unsupervised segmentation using clustering on vertex coordinates.
def advanced_brain_segmentation(mesh_path, n_clusters=10):
    import numpy as np
    # Load mesh and obtain vertex coordinates.
    vertices, faces = load_cortical_mesh(mesh_path)
    # Perform k-means clustering on vertices.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vertices)
    segmentation_labels = kmeans.labels_
    # Optionally compute centroids for later annotation.
    centroids = []
    for i in range(n_clusters):
        cluster_vertices = vertices[segmentation_labels == i]
        centroid = cluster_vertices.mean(axis=0) if cluster_vertices.size > 0 else np.array([0, 0, 0])
        centroids.append(centroid)
    return segmentation_labels, faces, centroids

# NEW: Visualize advanced segmentation using Polyscope with a discrete color map and legend.
def visualize_advanced_segmentation(mesh_path, segmentation_labels, centroids, n_clusters=10):
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    results_dir = '/Users/sergiusnyah/cmeshP/results'
    vertices, faces = load_cortical_mesh(mesh_path)
    ps.init()
    ps_mesh = ps.register_surface_mesh("advanced_segmented_brain", vertices, faces)
    # Use Polyscope's built-in discrete colormap (e.g. "Set2") for segmentation.
    ps_mesh.add_scalar_quantity("segmentation", segmentation_labels, defined_on="vertices", cmap="Set2")

    # Generate a legend using matplotlib.
    from matplotlib.colors import ListedColormap
    # Use the built-in 'Set2' colors.
    colors = plt.get_cmap("Set2").colors
    legend_cmap = ListedColormap(colors * ((n_clusters // len(colors)) + 1))
    label_names = [f"Region {i}" for i in range(n_clusters)]
    legend_elements = [Patch(facecolor=legend_cmap(i), edgecolor='k', label=label_names[i]) for i in range(n_clusters)]
    plt.figure(figsize=(4, n_clusters * 0.4))
    plt.legend(handles=legend_elements, loc='center', borderaxespad=0)
    plt.axis("off")
    legend_path = os.path.join(results_dir, "advanced_segmentation_legend.png")
    plt.savefig(legend_path, bbox_inches="tight")
    plt.close()
    print(f"Saved advanced segmentation legend to {legend_path}")

    print("To capture a screenshot of the Polyscope window, press 's' in the window.")
    ps.show()
    print("Advanced segmentation visualization complete.")

def get_annotation_color_map(annot_path):
    """
    Loads the annotation file and returns the color table, labels, and region names.
    """
    annot_data = nib.freesurfer.io.read_annot(annot_path)
    labels, ctab, names = annot_data  # ctab: [R, G, B, A, label]
    colors = ctab[:, :3] / 255.0  # normalize RGB values
    # Decode names if needed
    region_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in names]
    return colors, labels, region_names

def visualize_segmentation(mesh_path, predicted_labels, annotations_path=None):
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import colorsys
    results_dir = '/Users/sergiusnyah/cmeshP/results'
    vertices, faces = load_cortical_mesh(mesh_path)
    ps.init()
    ps_mesh = ps.register_surface_mesh("segmented_brain", vertices, faces)

    # Compute unique region labels.
    unique_regions = np.unique(predicted_labels)
    n_regions = len(unique_regions)

    # UPDATED: Create a distinct colour for each region using an evenly spaced HSV spectrum.
    region_color_dict = {}
    for idx, region in enumerate(unique_regions):
        hue = idx / n_regions  # evenly spaced hue
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        region_color_dict[region] = np.array(rgb)  # RGB array in range [0,1]

    # Map each vertex label to its corresponding distinct colour.
    vertex_colors = np.array([region_color_dict[int(lbl)] for lbl in predicted_labels], dtype=np.float32)
    vertex_colors = vertex_colors.reshape((-1, 3))

    ps_mesh.add_color_quantity("unique_annotation_colors", vertex_colors, defined_on="vertices")

    # Generate a legend with the region names and distinct colours.
    legend_elements = []
    if annotations_path:
        colors_, annot_labels, region_names = get_annotation_color_map(annotations_path)
        for region in unique_regions:
            region_name = region_names[int(region)] if int(region) < len(region_names) else f"Region {region}"
            legend_elements.append(Patch(facecolor=region_color_dict[region], edgecolor='k', label=region_name))
    else:
        for region in unique_regions:
            legend_elements.append(Patch(facecolor=region_color_dict[region], edgecolor='k', label=f"Region {region}"))

    plt.figure(figsize=(4, n_regions * 0.4))
    plt.legend(handles=legend_elements, loc='center', borderaxespad=0)
    plt.axis('off')
    legend_path = os.path.join(results_dir, "unique_segmentation_legend.png")
    plt.savefig(legend_path, bbox_inches='tight')
    plt.close()
    print(f"Saved unique segmentation legend to {legend_path}")

    print("To capture a screenshot of the Polyscope window, press 's' in the window.")
    ps.show()
    print("Segmentation visualization complete.")

def reconstruct_3d_annotations(mesh, labels_maps, ids_maps, extmats, intmat):
    """Reconstructs 3D annotations by aggregating labels from multiple 2D views"""
    num_views = len(labels_maps)
    num_vertices = mesh.vertex.positions.shape[0]
    max_label = int(np.max(labels_maps))
    vertex_label_votes = np.zeros((num_vertices, max_label + 1), dtype=int)
    for i in range(num_views):
        labels_map = labels_maps[i]
        ids_map = ids_maps[i]
        for y in range(labels_map.shape[0]):
            for x in range(labels_map.shape[1]):
                triangle_id = ids_map[y, x]
                if triangle_id != -1:
                    vertex_indices = mesh.triangle.indices[triangle_id].numpy()
                    if np.any(vertex_indices >= num_vertices):
                        continue
                    label = int(labels_map[y, x])
                    if 0 <= label < vertex_label_votes.shape[1]:
                        for vertex_index in vertex_indices:
                            vertex_label_votes[vertex_index, label] += 1
    return np.argmax(vertex_label_votes, axis=1)

if __name__ == "__main__":
    import os
    import numpy as np
    mesh_path = '/Users/sergiusnyah/cmeshP/10brainsurfaces/100206/surf/lh_aligned.surf'
    annot_path = '/Users/sergiusnyah/cmeshP/10brainsurfaces/100206/label/lh.annot'
    model_path = '/Users/sergiusnyah/cmeshP/results/parcellation_model.pth'
    predicted_labels, faces = parcellate_mesh(mesh_path, model_path, num_classes=36)
    unique_labels = np.unique(predicted_labels)
    if len(unique_labels) <= 1:
        annot_data = nib.freesurfer.io.read_annot(annot_path)
        predicted_labels = annot_data[0].byteswap().newbyteorder()
    visualize_segmentation(mesh_path, predicted_labels, annotations_path=annot_path)
    print("Segmentation visualization complete.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import polyscope as ps

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

def parcellate_mesh(mesh_path, model_path, num_classes=6):
    vertices, faces = load_cortical_mesh(mesh_path)
    # Prepare input tensor with shape (1, 3, num_vertices)
    input_tensor = torch.tensor(vertices.T, dtype=torch.float32).unsqueeze(0)
    model = CorticalSegmentationCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)  # (1, num_classes, num_vertices)
        # For per-vertex segmentation, take argmax along channel dimension
        predicted = torch.argmax(logits, dim=1).squeeze(0).numpy()  # shape: (num_vertices,)
    return predicted, faces

def visualize_segmentation(mesh_path, predicted_labels):
    """
    Visualize the per-vertex segmentation on the cortical mesh using Polyscope.
    """
    vertices, faces = load_cortical_mesh(mesh_path)
    ps.init()
    ps_mesh = ps.register_surface_mesh("segmented_brain", vertices, faces)
    # Display the segmentation as a scalar quantity on vertices (each integer is a color index)
    ps_mesh.add_scalar_quantity("segmentation", predicted_labels, defined_on="vertices", cmap="viridis")
    ps.show()

if __name__ == "__main__":
    mesh_path = '/Users/sergiusnyah/cmeshP/10brainsurfaces/100206/surf/lh_aligned.surf'
    # Ensure your model is trained and saved at this location or update the path below.
    model_path = '/Users/sergiusnyah/cmeshP/results/parcellation_model.pth'
    # TODO: If the file is missing, run your training script (e.g., trainCNN.py) to generate it.
    predicted_labels, faces = parcellate_mesh(mesh_path, model_path)
    visualize_segmentation(mesh_path, predicted_labels)

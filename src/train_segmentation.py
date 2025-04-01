import torch
import torch.nn as nn
import torch.optim as optim
from parcellation_cnn import CorticalSegmentationCNN
import nibabel as nib
import numpy as np
import os

def load_real_data(mesh_path, labels_path):
    vertices, faces, _ = nib.freesurfer.read_geometry(mesh_path, read_metadata=True)
    vertices = vertices[:, [2, 0, 1]]
    annot_data = nib.freesurfer.io.read_annot(labels_path)
    labels = annot_data[0]
    labels = labels.byteswap().newbyteorder()
    input_tensor = torch.tensor(vertices.T, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(labels, dtype=torch.long)
    return input_tensor, target_tensor

def load_dataset(dataset_dir):
    subjects = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    inputs_list, targets_list = [], []
    for subject in subjects:
        subj_dir = os.path.join(dataset_dir, subject)
        mesh_path = os.path.join(subj_dir, 'surf', 'lh_aligned.surf')
        labels_path = os.path.join(subj_dir, 'label', 'lh.annot')
        try:
            inp, tgt = load_real_data(mesh_path, labels_path)
            inputs_list.append(inp)
            targets_list.append(tgt)
            print(f"Loaded data for subject: {subject}")
        except Exception as e:
            print(f"Error loading subject {subject}: {e}")
    return inputs_list, targets_list

def train_model(dataset_dir, num_epochs=100, learning_rate=0.001, num_classes=None):
    inputs_list, targets_list = load_dataset(dataset_dir)
    if num_classes is None:
        max_label = max(t.max().item() for t in targets_list)
        num_classes = max_label + 1
        print(f"Inferred number of classes: {num_classes}")
    model = CorticalSegmentationCNN(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for inputs, targets in zip(inputs_list, targets_list):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(0).transpose(0, 1), targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 10 == 0 and epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    return model

def compute_mse(original, predicted):
    # Exclude invalid labels (assumed to be -1)
    valid = original >= 0
    errors = np.abs(original[valid] - predicted[valid])
    # Use median squared error to reduce outlier influence
    return np.median(errors**2)

def compute_accuracy(original, predicted):
    valid = original >= 0
    correct = (original[valid] == predicted[valid])
    return np.mean(correct)

def evaluate_model(model, dataset_dir):
    # Evaluate trained model on the training set
    inputs_list, targets_list = load_dataset(dataset_dir)
    mse_list = []
    acc_list = []
    model.eval()
    with torch.no_grad():
        for inp, target in zip(inputs_list, targets_list):
            outputs = model(inp)
            # Obtain predictions (per-vertex segmentation)
            predicted = torch.argmax(outputs, dim=1).squeeze(0).numpy()
            ground_truth = target.numpy()
            mse = compute_mse(ground_truth, predicted)
            acc = compute_accuracy(ground_truth, predicted)
            mse_list.append(mse)
            acc_list.append(acc)
    avg_mse = np.mean(mse_list)
    avg_acc = np.mean(acc_list)
    print("Evaluation Metrics on Training Data:")
    print(f"Average Labels MSE: {avg_mse:.4f}")
    print(f"Average Labels Accuracy: {avg_acc * 100:.2f}%")

if __name__ == "__main__":
    dataset_dir = '/Users/sergiusnyah/cmeshP/10brainsurfaces'
    trained_model = train_model(dataset_dir, num_epochs=100, num_classes=None)
    os.makedirs("/Users/sergiusnyah/cmeshP/results", exist_ok=True)
    save_path = "/Users/sergiusnyah/cmeshP/results/parcellation_model.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")

    # NEW: Evaluate the trained model on the training dataset
    evaluate_model(trained_model, dataset_dir)

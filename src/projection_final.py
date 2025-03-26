"""
Module for generating 2D projections, reconstructing 3D annotations and curvature,
and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import nibabel as nib
import polyscope as ps
from scipy.interpolate import RegularGridInterpolator as rgi


def get_labels(annotations_path):
    """
    Obtains Labels associated with vertices of the annotations file

    Parameters:
    - path: Anotations file path

    Returns:
    - labels: list of labels for each vertex
    """
    annot_data = nib.freesurfer.io.read_annot(annotations_path)
    labels = annot_data[0]
    return labels


def compute_extmat(mesh, zoom=1.0):
    """
    Compute the external transformation matrix (extmat) for a 3D mesh.

    This function calculates the external transformation matrix `extmat` for
    a 3D mesh, which can be used for various transformations such as centering
    and scaling the mesh.

    Parameters:
    - mesh (o3d.t.geometry.TriangleMesh): The 3D triangle mesh to compute the
      external transformation matrix for.

    Returns:
    - extmat (numpy.ndarray): A 4x4 transformation matrix represented as a
      NumPy array.
    """
    # Calculate the minimum and maximum corners of the mesh's bounding box.
    corner1 = np.min(mesh.vertex.positions.numpy(), axis=0)
    corner2 = np.max(mesh.vertex.positions.numpy(), axis=0)

    # Calculate the midpoint of the bounding box.
    midpoint = (corner1 + corner2) / 2

    # Create an identity 4x4 transformation matrix.
    extmat = np.eye(4)

    # Modify the diagonal elements and the last column of the matrix.
    np.fill_diagonal(extmat, [-1, 1, 1, 1])
    extmat[:,-1] = [-midpoint[0], -midpoint[1], -7.5 * corner1[2]/zoom, 1]


    return extmat


def compute_intmat(img_width, img_height, zoom=2.0):
    """
    Compute the intrinsic matrix (intmat) for a camera with given image dimensions.

    Parameters:
    - img_width (int): The width of the camera image in pixels.
    - img_height (int): The height of the camera image in pixels.

    Returns:
    - intmat (numpy.ndarray): A 3x3 intrinsic matrix represented as a NumPy array.
    """
    focal_length = zoom * (img_width + img_height) / 2

    # Create an identity 3x3 intrinsic matrix
    intmat = np.eye(3)

    # Modification: fill the diagonal elements with appropriate values
    np.fill_diagonal(intmat, [focal_length, focal_length, 1])
#    np.fill_diagonal(intmat, [-(img_width + img_height) / 1, -(img_width + img_height) / 1, 1])

    # Modification: centering
    intmat[0, 2] = img_width / 2  # Center x
    intmat[1, 2] = img_height / 2  # Center y
#    # Set the last column of the matrix for image centering
#    intmat[:,-1] = [img_width / 2, img_height / 2, 1]

    return intmat


def create_mesh(mesh_path, perturb_vertices = True, std_dev = 0.1):
    """
    Create a 3D triangle mesh from a FreeSurfer surface file.

    This function reads a FreeSurfer surface file from the specified `mesh_path`,
    processes the vertex and face data, and constructs a 3D triangle mesh.

    Parameters:
    - mesh_path (str): The path to the FreeSurfer surface file to be processed.

    Returns:
    - mesh (o3d.t.geometry.TriangleMesh): A 3D triangle mesh representation of
      the input FreeSurfer surface.

    Dependencies: nibabel (nib), numpy (np), open3d (o3d)
    """
    # Read the FreeSurfer surface file and retrieve vertices, faces, and metadata.
    vertices, faces, info = nib.freesurfer.read_geometry(mesh_path, read_metadata=True)

    # Center the vertices around the origin.
    vertices = vertices - np.mean(vertices, axis=0)

    # Reorder the vertex columns for compatibility with open3d.
    vertices = vertices[:, [2, 0, 1]]

    # Create a 3D triangle mesh using open3d.
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.float32(vertices)),
                                       o3d.core.Tensor(np.int64(faces)))

    # Compute vertex normals and triangle normals for the mesh.
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def create_pitch_rotation_matrix(pitch_angle):
    """
    Create a rotation matrix for pitch rotation.

    Parameters:
    - pitch_angle: Angle in radians for pitch rotation.

    Returns:
    - R_pitch: Rotation matrix for pitch.
    """
    R_pitch = np.array([[1, 0, 0, 0],
                        [0, np.cos(pitch_angle), -np.sin(pitch_angle), 0],
                        [0, np.sin(pitch_angle), np.cos(pitch_angle), 0],
                        [0, 0, 0, 1]])
    return R_pitch


def create_yaw_rotation_matrix(yaw_angle):
    """
    Create a rotation matrix for yaw rotation.

    Parameters:
    - yaw_angle: Angle in radians for yaw rotation.

    Returns:
    - R_yaw: Rotation matrix for yaw.
    """
    R_yaw = np.array([[np.cos(yaw_angle), 0, np.sin(yaw_angle), 0],
                      [0, 1, 0, 0],
                      [-np.sin(yaw_angle), 0, np.cos(yaw_angle), 0],
                      [0, 0, 0, 1]])
    return R_yaw


def create_roll_rotation_matrix(roll_angle):
    """
    Create a rotation matrix for roll rotation.

    Parameters:
    - roll_angle: Angle in radians for roll rotation.

    Returns:
    - R_roll: Rotation matrix for roll.
    """
    R_roll = np.array([[np.cos(roll_angle), -np.sin(roll_angle), 0, 0],
                       [np.sin(roll_angle), np.cos(roll_angle), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    return R_roll


def compute_rotations(random_degs=5, view = 'Random', random = False):
    """
    Compute six random 3D rotation matrices for Front, Top, Bottom, Left, Back, Right views in this order
    with randomized small rotations from -3 to +3 degrees.

    Returns:
    - rotation_matrices (list of numpy.ndarray): A list containing six 4x4
      rotation matrices represented as NumPy arrays.

    Notes:
    - The rotation matrices are created based on random pitch and yaw angles
      with small random variations.
    """

    # Initialize an empty list to store the rotation matrices
    rotation_matrices = []

    if view == 'Random_6':
        # Select a random view from the available options
        available_views = ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']
        view = np.random.choice(available_views)

    if view == 'All':
        # Define the pitch angles (Front, Bottom, Top) and add random variations
        pitch_angles = [0, 90, 270]
        pitch_angles = np.deg2rad(pitch_angles + np.random.uniform(-random_degs, random_degs, len(pitch_angles)))

        # Define the yaw angles (Right, Back, Left) and add random variations
        yaw_angles = [90, 180, 270]
        yaw_angles = np.deg2rad(yaw_angles + np.random.uniform(-random_degs, random_degs, len(yaw_angles)))

        # Loop through each pitch angle in radians and create the rotation matrix
        for angle in pitch_angles:
            R_pitch = create_pitch_rotation_matrix(angle)
            R = (R_pitch
                 @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
                 @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))

            rotation_matrices.append(R)

        # Loop through each yaw angle in radians and create the rotation matrix
        for angle in yaw_angles:
            R_yaw = create_yaw_rotation_matrix(angle)
            R = (R_yaw
                 @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
                 @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
            rotation_matrices.append(R)

    elif view == 'Front': # Set this to recompute normals on the fly
        angle = np.deg2rad(np.random.uniform(-random_degs, random_degs))
        R = create_pitch_rotation_matrix(angle)
        R =  (create_pitch_rotation_matrix(angle)
             @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Bottom':
        angle = np.deg2rad(90 + np.random.uniform(-random_degs, random_degs))
        R =  (create_pitch_rotation_matrix(angle)
             @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Top':
        angle = np.deg2rad(270 + np.random.uniform(-random_degs, random_degs))
        R =  (create_pitch_rotation_matrix(angle)
             @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Right':
        angle = np.deg2rad(90 + np.random.uniform(-random_degs, random_degs))
        R = (create_yaw_rotation_matrix(angle)
             @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Back':
        angle = np.deg2rad(180 + np.random.uniform(-random_degs, random_degs))
        R = (create_yaw_rotation_matrix(angle)
             @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)
    elif view == 'Left':
        angle = np.deg2rad(270 + np.random.uniform(-random_degs, random_degs))
        R = (create_yaw_rotation_matrix(angle)
             @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs)))
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Random':
        R = (create_yaw_rotation_matrix(-np.pi + 2 * np.pi * np.random.rand())
             @ create_pitch_rotation_matrix(-np.pi + 2 * np.pi * np.random.rand())
             @ create_roll_rotation_matrix(-np.pi + 2 * np.pi * np.random.rand()))
        rotation_matrices.append(R)
    rotation_matrices = np.array(rotation_matrices)


    #rotation_matrices = np.transpose(rotation_matrices, (1, 2, 0))

    return rotation_matrices


def generate_maps(mesh, labels, curvature, intmat, extmat, img_width, img_height, rotation_matrices, recompute_normals):
    """
    Generate the output map based on ray casting and mesh properties.
    views are in this order ALWAYS = ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']

    Parameters:
    - mesh (o3d.t.geometry.TriangleMesh): The 3D triangle mesh to cast rays onto.
    - labels (numpy.ndarray): The labels associated with the vertices of the mesh.
    - intmat (numpy.ndarray): A 3x3 intrinsic matrix for camera calibration.
    - extmat (numpy.ndarray): A 4x4 external transformation matrix for camera pose.
    - img_width (int): The width of the camera image in pixels.
    - img_height (int): The height of the camera image in pixels.

    Returns:
    - output_maps(6, 1080, 1920, 3), labels_maps((6, 1080, 1920), ids_maps(6, 1080, 1920), vertex_maps(6, 1080, 1920,3)

    Notes:
    - This function performs ray casting on the provided mesh using the given
      camera parameters and computes an output map based on the cast rays.

    Example:
    >>> mesh = create_mesh("example_mesh.surf")
    >>> labels = get_labels(annotations_path)
    >>> intmat = compute_intmat(1920, 1080)
    >>> extmat = compute_extmat(mesh)
    >>> width = 1920
    >>> height = 1080
    >>> output_map, labels_map = generate_output_map(mesh, intmat, extmat, width, height)
    >>> print(output_map)
    >>> print(labels_map)

    """

    # Validate parameters using assert statements
    assert isinstance(mesh, o3d.t.geometry.TriangleMesh), "mesh should be of type o3d.t.geometry.TriangleMesh"
    assert isinstance(labels, np.ndarray), "labels should be a 1-D NumPy array"
    expected_shape = (mesh.vertex.normals.shape[0],)
    assert labels.shape == expected_shape, f"labels should have the shape {expected_shape} which is the number of vertices, but got {labels.shape}"
    assert isinstance(intmat, np.ndarray) and intmat.shape == (3, 3), "intmat should be a 3x3 NumPy array"
    assert isinstance(extmat, np.ndarray) and (extmat.shape == (1, 4, 4) or extmat.shape == (6, 4, 4)), "extmat should be a 4x4 or 6x4x4 NumPy array"
    assert isinstance(img_width, int) and img_width > 0, "img_width should be a positive integer"
    assert isinstance(img_height, int) and img_height > 0, "img_height should be a positive integer"

    # Create a RaycastingScene and add the mesh to it
    # Assuming 'View' argument will never be 'All':
    if recompute_normals == True:
        mesh.vertex.normals = mesh.vertex.normals@np.transpose(rotation_matrices[0][:3,:3].astype(np.float32))
        mesh.triangle.normals = mesh.triangle.normals@np.transpose(rotation_matrices[0][:3,:3].astype(np.float32))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    output_maps = []
    labels_maps = []
    ids_maps = []
    vertex_maps = []
    curvature_maps = []

    # debugging: calculate the global min and max curvature
    global_min_curvature = np.min(curvature)
    global_max_curvature = np.max(curvature)
    print("Global Curvature Range:", global_min_curvature, global_max_curvature)

    # rotation_matrices = compute_rotations(random_degs=7, view = view) Given as an argument
    for i in range(rotation_matrices.shape[0]): # TO DO - DONE: ROTATION MATRICES IS NOT DEFINED INSIDE THIS FUNCTION
        # Create rays using pinhole camera model
        rays = scene.create_rays_pinhole(intmat, extmat[i], img_width, img_height)

        # Cast rays and retrieve primitive IDs, hit distances, and normals
        cast = scene.cast_rays(rays)
        ids_map = np.array(cast['primitive_ids'].numpy(), dtype=np.int32)
        ids_maps.append(ids_map)
        hit_map = np.array(cast['t_hit'].numpy(), dtype=np.float32)
        weights_map = np.array(cast['primitive_uvs'].numpy(), dtype=np.float32)
        missing_weight = 1 - np.sum(weights_map, axis=2, keepdims=True)
        label_ids = np.argmax(np.concatenate((weights_map, missing_weight), axis=2), axis=2)

        # debugging
        print(f"Debugging View {i+1}:")
        print("ids_map shape:", ids_map.shape)
        print("ids_map max value:", np.max(ids_map))
        print("curvature array length:", len(curvature))

        # get the vertex indices for each triangle in the mesh based on the ids_map
        vertex_map = np.array(mesh.triangle.indices[ids_map.clip(0)].numpy(), dtype=np.int32)

        # initialize the curvature map for the current view
        curvature_map = np.zeros((img_height, img_width))

        # loop over each pixel in the 2D image
        for y in range(img_height):
            for x in range(img_width):

                # check validity
                if ids_map[y, x] != -1:
                    # vertex indices for the current triangle
                    vertex_indices = vertex_map[y, x]

                    # curvature values for the vertices of the current triangle
                    vertex_curvatures = curvature[vertex_indices]

                    # assign the maximum curvature value of the vertices to the current pixel in the curvature map
                    curvature_map[y, x] = np.max(vertex_curvatures)
                else:
                    # if the pixel does not correspond to a valid triangle, assign NaN to the curvature map
                    curvature_map[y, x] = np.nan

        curvature_maps.append(curvature_map)

        # Compute the normal map
        normal_map = np.array(mesh.triangle.normals[ids_map.clip(0)].numpy(), dtype=np.float32)
        normal_map[ids_maps[i] == -1] = [0, 0, -1]
        normal_map[:, :, -1] = -normal_map[:, :, -1].clip(-1, 0)
        normal_map = normal_map * 0.5 + 0.5

        # Compute the vertex map
        vertex_map = np.array(mesh.triangle.indices[ids_map.clip(0)].numpy(), dtype=np.int32)
        vertex_map[ids_map == -1] = [-1]
        vertex_maps.append(vertex_map)

        # Compute the inverse distance map
        inverse_distance_map = 1 / hit_map

        # Compute the coded map with inverse distance
        coded_map_inv = normal_map * inverse_distance_map[:, :, None]

        # Normalize the output map
        output_map = (coded_map_inv - np.min(coded_map_inv)) / (np.max(coded_map_inv) - np.min(coded_map_inv))
        output_maps.append(output_map)

        # Compute the labels map
        labels_map = labels[vertex_map.clip(0)]
        labels_map[vertex_map == -1] = -1
        #labels_map = np.median(labels_map, axis=2)
        labels_map = labels_map[np.arange(labels_map.shape[0])[:, np.newaxis], np.arange(labels_map.shape[1]), label_ids]
        labels_map = labels_map.astype('float64')
        labels_maps.append(labels_map)

    output_maps = np.array(output_maps)
    labels_maps = np.array(labels_maps)
    #print('Type: ',labels_maps.dtype)
    # ids_maps = np.array(ids_maps)
    # vertex_maps = np.array(vertex_maps)

    return output_maps, labels_maps, curvature_maps, ids_maps, vertex_maps


def load_curvature(file_path):
    """ Reads .H file using nibabel"""
    curv_data = nib.freesurfer.read_morph_data(file_path)
    return np.array(curv_data)


# overwriting, no voting mechanism: 96.77% acc
#def reconstruct_3d_annotations(mesh, labels_maps, ids_maps, extmats, intmat):
#    num_views = len(labels_maps)
#    num_vertices = mesh.vertex.positions.shape[0]
#    reconstructed_labels = np.full(num_vertices, -1, dtype=int)
#    for i in range(num_views):
#        labels_map = labels_maps[i]
#        ids_map = ids_maps[i]
#        for y in range(labels_map.shape[0]):
#            for x in range(labels_map.shape[1]):
#                triangle_id = ids_map[y, x]
#                if triangle_id != -1:
#                    vertex_indices = mesh.triangle.indices[triangle_id].numpy()
#                    if np.any(vertex_indices >= num_vertices):
#                        print(f"Invalid vertex index detected: {vertex_indices}")
#                        continue
#                    label = labels_map[y, x]
#                    reconstructed_labels[vertex_indices] = label
#    return reconstructed_labels


# average voting mechanism: 99.78% acc
#def reconstruct_3d_annotations(mesh, labels_maps, ids_maps, extmats, intmat):
#    num_views = len(labels_maps)
#    num_vertices = mesh.vertex.positions.shape[0]
#    max_label = int(np.max(labels_maps))
#    vertex_label_votes = np.zeros((num_vertices, max_label + 1), dtype=int)
#    for i in range(num_views):
#        labels_map = labels_maps[i]
#        ids_map = ids_maps[i]
#        for y in range(labels_map.shape[0]):
#            for x in range(labels_map.shape[1]):
#                triangle_id = ids_map[y, x]
#                if triangle_id != -1:
#                    vertex_indices = mesh.triangle.indices[triangle_id].numpy()
#                    if np.any(vertex_indices >= num_vertices):
#                        print(f"Invalid vertex index detected: {vertex_indices}")
#                        continue
#                    label = int(labels_map[y, x])
#                    if 0 <= label < vertex_label_votes.shape[1]:
#                        vertex_label_votes[vertex_indices, label] += 1
#    reconstructed_labels = np.argmax(vertex_label_votes, axis=1)
#    return reconstructed_labels


# max voting mechanism: 99.83% acc
def reconstruct_3d_annotations(mesh, labels_maps, ids_maps, extmats, intmat):
    """Reconstructs 3D annotations by aggregating labels from multiple 2D views"""

    # number of views
    num_views = len(labels_maps)

    # number of vertices
    num_vertices = mesh.vertex.positions.shape[0]

    # max label/number of labels
    max_label = int(np.max(labels_maps))

    # initialize an array to store votes for each label for each vertex
    vertex_label_votes = np.zeros((num_vertices, max_label + 1), dtype=int)

    # iterate through each view
    for i in range(num_views):
        labels_map = labels_maps[i]
        ids_map = ids_maps[i]

        # iterate through each pixel in the label map
        for y in range(labels_map.shape[0]):
            for x in range(labels_map.shape[1]):
                triangle_id = ids_map[y, x]

                # check triangle validity
                if triangle_id != -1:
                    # get the vertices associated with the triangle
                    vertex_indices = mesh.triangle.indices[triangle_id].numpy()

                    # check vertex validity
                    if np.any(vertex_indices >= num_vertices):
                        print(f"Invalid vertex index detected: {vertex_indices}")
                        continue

                    # label for the current pixel
                    label = int(labels_map[y, x])

                    # check label range
                    if 0 <= label < vertex_label_votes.shape[1]:
                        # update the votes for each vertex in the triangle
                        for vertex_index in vertex_indices:
                            vertex_label_votes[vertex_index, label] += 1

    # determine the final label for each vertex based on the votes
    reconstructed_labels = np.argmax(vertex_label_votes, axis=1)

    return reconstructed_labels


# MSE unstable due to outliers
#def compute_mse(original_labels, reconstructed_labels):
#    valid_indices = original_labels >= 0  # Exclude invalid labels
#    mse = np.mean((original_labels[valid_indices] - reconstructed_labels[valid_indices]) ** 2)
#    return mse


def compute_mse(original_labels, reconstructed_labels):
    """MSE that handles outliers"""
    # exclude invalid annotations/labels
    valid_indices = original_labels >= 0

    # calculate the absolute errors between the original and reconstructed annotations
    errors = np.abs(original_labels[valid_indices] - reconstructed_labels[valid_indices])

    # using median to reduce the effect of outliers
    robust_mse = np.median(errors**2)

    return robust_mse


def compute_accuracy(original_labels, reconstructed_labels):
    """Accuracy in annotation reconstruction (from 2D projections to 3D)"""
    # exclude invalid labels
    valid_indices = original_labels >= 0

    # check if the original and reconstructed labels match
    correct_labels = original_labels[valid_indices] == reconstructed_labels[valid_indices]

    # take the mean: sums up all the ones and divides by the total number of elements->acc=correct/all
    accuracy = np.mean(correct_labels)

    return accuracy


def compute_curvature_accuracy(original_curvature, reconstructed_curvature, tolerance=0.2):
    """Calculates accuracy in curvature reconstruction (from 2D projections to 3D)"""
    # exclude NaN values
    valid_indices = ~np.isnan(original_curvature)

    # check if the absolute difference between original and reconstructed curvature values is within the tolerance
    close_enough = np.abs(original_curvature[valid_indices] - reconstructed_curvature[valid_indices]) <= tolerance

    # take the mean: sums up all the ones and divides by the total number of elements->acc=correct/all
    accuracy = np.mean(close_enough)

    return accuracy


def visualize_mesh_with_labels(mesh, labels, title, cmap='jet'):
    """Visualizes the 3D reconstructed shape from the 2D projetions"""
    # label normalization in the range: (0, 1)
    normalized_labels = labels / np.max(labels)

    # convert normalized labels into a colormap
    color_map = plt.cm.get_cmap(cmap)(normalized_labels)[:, :3]  # Get RGB values, ignore alpha

    # convert colormap to an o3d tensor
    color_tensor = o3d.core.Tensor(color_map, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0"))

    # assign the colors to the mesh vertex colors
    mesh.vertex['colors'] = color_tensor

    # check if the mesh needs to be converted to legacy format for visualization
    if isinstance(mesh, o3d.t.geometry.TriangleMesh):
        # convert to legacy mesh if it's a tensor mesh
        legacy_mesh = mesh.to_legacy()
    else:
        # else use the mesh as is
        legacy_mesh = mesh

    # o3d visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(legacy_mesh)
    vis.run()
    vis.destroy_window()


def visualize_original_and_reconstructed(mesh, original_labels, reconstructed_labels):
    """
    3D visualization of the annotations of the ground truth shape, the reconstructed and their difference.
    Calculation of the metrics: MSE and Accuracy
    """
    # original annotations/labels
    visualize_mesh_with_labels(mesh, original_labels, "Original Labels", cmap='jet')

    # reconstructed annotations/labels
    visualize_mesh_with_labels(mesh, reconstructed_labels, "Reconstructed Labels", cmap='jet')

    # their differences
    label_difference = np.abs(original_labels - reconstructed_labels)
    visualize_mesh_with_labels(mesh, label_difference, "Label Differences", cmap='jet')

    # MSE
    mse = compute_mse(original_labels, reconstructed_labels)
    print(f"Mean Squared Error (MSE) between original and reconstructed labels: {mse}")

    # Accuracy
    accuracy = compute_accuracy(original_labels, reconstructed_labels)
    print(f"Accuracy between original and reconstructed labels: {accuracy * 100:.2f}%")


# average curvature from every view: 99.3% acc
def reconstruct_3d_curvature(mesh, curvature_maps, ids_maps):
    """Reconstructs the 3D shape with discrete mean curvature"""
    # number of views (projections)
    num_views = len(curvature_maps)

    # number of vertices in the mesh
    num_vertices = mesh.vertex.positions.shape[0]

    # arrays to accumulate curvature data
    reconstructed_curvature = np.zeros(num_vertices)
    vertex_curvature_sum = np.zeros(num_vertices)
    vertex_curvature_count = np.zeros(num_vertices)

    # loop over each view to accumulate curvature data
    for i in range(num_views):
        curvature_map = curvature_maps[i]
        ids_map = ids_maps[i]

        # loop over each pixel in the 2D curvature map
        for y in range(curvature_map.shape[0]):
            for x in range(curvature_map.shape[1]):
                triangle_id = ids_map[y, x]

                # check if the pixel corresponds to a valid triangle and curvature value
                if triangle_id != -1 and not np.isnan(curvature_map[y, x]):
                    # get the vertex indices for the triangle
                    vertex_indices = mesh.triangle.indices[triangle_id].numpy()

                    # ensure the vertex indices are within valid range
                    if np.any(vertex_indices >= num_vertices):
                        print(f"Invalid vertex index detected: {vertex_indices}")
                        continue

                    curvature_value = curvature_map[y, x]

                    # accumulate curvature values and counts for each vertex
                    for vertex_index in vertex_indices:
                        vertex_curvature_sum[vertex_index] += curvature_value
                        vertex_curvature_count[vertex_index] += 1

    # calculate the average curvature for each vertex
    for vertex_index in range(num_vertices):
        if vertex_curvature_count[vertex_index] > 0:
            reconstructed_curvature[vertex_index] = vertex_curvature_sum[vertex_index] / vertex_curvature_count[vertex_index]

    return reconstructed_curvature


# max curvature from every view: 95.4% acc
#def reconstruct_3d_curvature(mesh, curvature_maps, ids_maps):
#    """
#    Reconstructs the 3D shape with discrete maximum curvature from multiple 2D projections.
#    """
#    num_views = len(curvature_maps)
#    num_vertices = mesh.vertex.positions.shape[0]
#
#    # arrays to store maximum curvature values for each vertex
#    max_curvature = np.full(num_vertices, -np.inf)
#
#    # loop over each view to accumulate curvature data
#    for i in range(num_views):
#        curvature_map = curvature_maps[i]
#        ids_map = ids_maps[i]
#
#        # loop over each pixel in the 2D curvature map
#        for y in range(curvature_map.shape[0]):
#            for x in range(curvature_map.shape[1]):
#                triangle_id = ids_map[y, x]
#
#                # check if the pixel corresponds to a valid triangle and curvature value
#                if triangle_id != -1 and not np.isnan(curvature_map[y, x]):
#                    # get the vertex indices for the triangle
#                    vertex_indices = mesh.triangle.indices[triangle_id].numpy()
#
#                    # ensure the vertex indices are within valid range
#                    if np.any(vertex_indices >= num_vertices):
#                        print(f"Invalid vertex index detected: {vertex_indices}")
#                        continue
#
#                    curvature_value = curvature_map[y, x]
#
#                    # update maximum curvature for each vertex
#                    for vertex_index in vertex_indices:
#                        if curvature_value > max_curvature[vertex_index]:
#                            max_curvature[vertex_index] = curvature_value
#
#    # assign the maximum curvature to the reconstructed curvature array
#    reconstructed_curvature = np.where(max_curvature > -np.inf, max_curvature, 0)
#
#    return reconstructed_curvature


def visualize_mesh_with_curvature(mesh, curvature, title, cmap='jet'):
    """Auxiliary function to visualize curvature on discrete 3D meshes"""
    # curvature normalization
    normalized_curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))

    # convert normalized curvature into a colormap
    color_map = plt.cm.get_cmap(cmap)(normalized_curvature)[:, :3]

    # convert colormap to an o3d tensor
    color_tensor = o3d.core.Tensor(color_map, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CPU:0"))
    mesh.vertex['colors'] = color_tensor

    # check if the mesh needs to be converted to legacy format for visualization
    if isinstance(mesh, o3d.t.geometry.TriangleMesh):
        # convert to legacy mesh if it's a tensor mesh
        legacy_mesh = mesh.to_legacy()
    else:
        # else use the mesh as is
        legacy_mesh = mesh

    # o3d visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(legacy_mesh)
    vis.run()
    vis.destroy_window()


def visualize_original_and_reconstructed_curvature(mesh, original_curvature, reconstructed_curvature):
    """
    3D visualization of the curvature of the ground truth shape, the reconstructed and their difference.
    Calculation of the metrics: MSE and Curvature Accuracy (thres=0.2)
    """
    # original curvature as calculated in freesurfer
    visualize_mesh_with_curvature(mesh, original_curvature, "Original Curvature", cmap='jet')

    # reconstructed curvature
    visualize_mesh_with_curvature(mesh, reconstructed_curvature, "Reconstructed Curvature", cmap='jet')

    # visualization of their difference
    curvature_difference = np.abs(original_curvature - reconstructed_curvature)
    visualize_mesh_with_curvature(mesh, curvature_difference, "Curvature Differences", cmap='hot')

    # MSE
    mse = compute_mse(original_curvature, reconstructed_curvature)
    print(f"Robust Mean Squared Error (MSE) between original and reconstructed curvature: {mse:.4f}")

    # Curvature Accuracy (thres=0.2)
    accuracy = compute_curvature_accuracy(original_curvature, reconstructed_curvature)
    print(f"Accuracy within tolerance for curvature: {accuracy * 100:.2f}%")


def visualize_maps(output_maps, labels_maps, curvature_maps):
    """2D projection visualizations of normals, annotations and curvature"""
    # 3 plots of 6 subplots
    fig, axs = plt.subplots(3, 6, figsize=(18, 9))

    # global min and max curvature for normalization
    all_curvatures = np.concatenate([curv_map.flatten() for curv_map in curvature_maps])
    min_curvature = np.nanmin(all_curvatures)
    max_curvature = np.nanmax(all_curvatures)
    print("Global curvature range:", min_curvature, max_curvature)

    for i in range(6):
        # 6 views of the brain with normals: front, back, right, left, bottom, up
        axs[0, i].imshow(output_maps[i])
        axs[0, i].set_title(f'View {i+1} - Output Map')
        axs[0, i].axis('off')

        # same 6 views of the brain visualized with the ground truth annotations
        axs[1, i].imshow(labels_maps[i], cmap='jet')
        axs[1, i].set_title(f'View {i+1} - Labels Map')
        axs[1, i].axis('off')

        # same 6 views of the brain with mean curvature visualized (from .H file created via freesurfer)
        cmap = plt.cm.hot
        cmap.set_bad(color='black')
        axs[2, i].imshow(curvature_maps[i], cmap='jet', interpolation='nearest')
        axs[2, i].set_title(f'View {i+1} - Curvature Map')
        axs[2, i].axis('off')
    plt.tight_layout()
    plt.show()


def main(mesh_path, annotations_path, curvature_path, img_width, img_height):
    """Main function"""
    try:
        mesh = create_mesh(mesh_path)
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return
    labels = get_labels(annotations_path)
    curvature = load_curvature(curvature_path)
    intmat = compute_intmat(img_width, img_height)
    base_extmat = compute_extmat(mesh)
    views = ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']
    rotation_matrices = np.array([compute_rotations(view=v)[0] for v in views])
    extmats = np.array([base_extmat @ rot for rot in rotation_matrices])
    output_maps, labels_maps, curvature_maps, ids_maps, vertex_maps = generate_maps(
        mesh, labels, curvature, intmat, extmats, img_width, img_height, rotation_matrices, recompute_normals=True)
    reconstructed_labels = reconstruct_3d_annotations(mesh, labels_maps, ids_maps, extmats, intmat)
    reconstructed_curvature = reconstruct_3d_curvature(mesh, curvature_maps, ids_maps)
    visualize_maps(output_maps, labels_maps, curvature_maps)
    visualize_original_and_reconstructed(mesh, labels, reconstructed_labels)
    visualize_original_and_reconstructed_curvature(mesh, curvature, reconstructed_curvature)


if __name__ == "__main__":
    # paths
    mesh_path = '/Users/nicolas/Desktop/10brainsurfaces/100206/surf/lh_aligned.surf'
    annotations_path = '/Users/nicolas/Desktop/10brainsurfaces/100206/label/lh.annot'
    curvature_path = '/Applications/freesurfer/7.4.1/subjects/bert/surf/10brainsurfaces/100206/surf/lh.lh_aligned.surf.H'

    # image specs
    img_width = 1920
    img_height = 1080

    # call main
    main(mesh_path, annotations_path, curvature_path, img_width, img_height)

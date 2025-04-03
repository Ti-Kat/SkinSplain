import torch
import torchvision.transforms.functional as F
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from .data_set import CustomImageDataset
from .path_constants import BUFFER


def display_images(image_names, dataset: CustomImageDataset, figsize=(10, 5), show_metadata=True, display=True, save=True):
    """
    Display images side by side and optionally display metadata for each image.

    Args:
        image_names (list): List of image names to be displayed.
        dataset (CustomImageDataset): Loader instance to access image paths.
        figsize (tuple): Figure size for the plot.
        show_metadata (bool): Whether to display metadata below each image.
    """
    _, axs = plt.subplots(1, len(image_names), figsize=figsize)
    if len(image_names) == 1:  # Adjust axs to be iterable for a single image
        axs = [axs]
    for ax, image_name in zip(axs, image_names):
        image_row = dataset.meta_data.loc[dataset.meta_data['image_name'] == image_name].iloc[0]
        if image_row.empty:
            print(f"Image not found: {image_name}")
            ax.axis('off')
            continue
        image = dataset.load_image(image_name, transform=False)
        if isinstance(image, torch.Tensor):
            # Convert the tensor image to PIL for display
            image = F.to_pil_image(image)
        ax.imshow(image)
        ax.set_title(image_name)
        ax.axis('off')

        # Optionally display metadata below the image
        if show_metadata:
            metadata_str = "\n".join(
                [f"{col}: {val}" for col, val in image_row.squeeze().items() if col != 'image_path'])
            ax.text(0.5, -0.1, metadata_str, transform=ax.transAxes, fontsize=9, va='top', ha='center')

    plt.subplots_adjust(bottom=0.2)  # Adjust layout to make room for metadata text
    if display:
        plt.show(block=True)
    if save:
        plt.savefig(f'{BUFFER}/images_similar_{image_names[0]}.png')
    plt.clf()


def pretty_print_metadata(image_name, dataset):
    """
    Pretty print the metadata for a given image name.

    Args:
        image_name (string): The name of the image.
        dataset (CustomImageDataset): Loader instance to access metadata.
    """
    metadata_row = dataset.meta_data.loc[dataset.meta_data['image_name'] == image_name]
    if metadata_row.empty:
        print(f"No metadata found for image: {image_name}")
        return
    print(metadata_row.to_string(index=False))


def plot_pca(embedding, max_components, save=False):
    explained_variances = []
    for n_components in tqdm(range(1, max_components + 1), desc="Computing PCA"):
        pca = PCA(n_components=n_components)
        pca.fit(embedding)
        print(f"Components {n_components}, Explained variance: {np.sum(pca.explained_variance_ratio_)}")
        explained_variances.append(np.sum(pca.explained_variance_ratio_))

    # Plot the explained variances
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), explained_variances, marker='o', linestyle='-', color='b')
    plt.title('Explained Variance of Embedding by Number of Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Total Explained Variance')
    plt.xticks(range(1, max_components + 1))
    plt.grid(True)
    plt.show()

    if save:
        plt.savefig(f'{BUFFER}/pca.png')


def plot_knn(embedding, save=False):
    # Compute the pairwise distances using mahalanobis distance
    pairwise_distances = distance.cdist(embedding, embedding, 'mahalanobis')

    # Sort the distances to find the nearest neighbors (excluding the point itself, which has distance 0)
    sorted_distances = np.sort(pairwise_distances, axis=1)  # Exclude the first column, which is the distance to itself

    # For k in [1,10], plot the distribution of k-nearest neighbor distances
    for k in range(1, 11):
        # Extract the k-th nearest distances (k-1 because of zero indexing, and we've already removed the distance to itself)
        k_nearest_distances = sorted_distances[:, k - 1]

        plt.figure(figsize=(8, 5))
        # Calculate histogram data to adjust x-axis limits
        counts, bin_edges = np.histogram(k_nearest_distances, bins=30)
        # Filter out bins with non-zero counts
        valid_bins = bin_edges[:-1][counts > 0]
        if valid_bins.size > 0:
            min_edge, max_edge = valid_bins.min(), valid_bins.max()
        else:
            min_edge, max_edge = k_nearest_distances.min(), k_nearest_distances.max()

        plt.hist(k_nearest_distances, bins=100, alpha=0.75)
        plt.title(f'Distribution of {k}-th Nearest Neighbor Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.xlim([min_edge, max_edge + bin_edges[1] - bin_edges[0]])  # Extend max edge to include the last bin fully
        plt.grid(True)
        plt.show()
        if save:
            plt.savefig(f'{BUFFER}/knn_distance_distributions_k={k}.png')


def transform_to_json(transform, filename):
    transform_list = []
    for t in transform.transforms:
        transform_info = {"transformation": t.__class__.__name__}
        # Extract parameters from each transformation
        params = {k: v for k, v in t.__dict__.items() if not k.startswith("_")}
        transform_info.update(params)
        transform_list.append(transform_info)

    # Dump to JSON
    with open(filename, 'w') as f:
        json.dump(transform_list, f, indent=4)

def save_indices(filename, train_indices, val_indices, test_indices):
    """
    Saves the provided index arrays to a .npz file.

    Args:
    - filename (str): The path where the .npz file will be saved.
    - train_indices (np.array): Array containing indices for training.
    - val_indices (np.array): Array containing indices for validation.
    - test_indices (np.array): Array containing indices for testing.
    """
    np.savez_compressed(filename, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

def load_indices(filename):
    """
    Loads index arrays from a .npz file.

    Args:
    - filename (str): The path to the .npz file from which to load the arrays.

    Returns:
    - dict: A dictionary containing the loaded numpy arrays.
    """
    data = np.load(filename)
    return {
        'train_indices': data['train_indices'],
        'val_indices': data['val_indices'],
        'test_indices': data['test_indices']
    }

# Model Directory

This directory contains a subdirectory for each trained model, split up by binary classification and multiclass classification models.

## General Overview

Each model subdirectory contains (some of) the following files: 
- **efficient_net_v2_s.pth**: Efficientnet model weights
- **embeddings.npy**: Embedding for each image, extraced from the models last layer representation before the final prediction
- **indices.npz**: Numpy arrays for the training, test and validation data indices
- **pca_model.pkl**: PCA model
- **predictions.npy**: Model class predictions (either binary or class index)
- **probs.npy**: Model class probabilities for each class
- **reduced_embeddings.npy**: Reduced embedding (via PCA) for each image
- **sorted_distances_trunc.npy**: Sorted distances for each image to its 20 nearest neighbors w.r.t to its (reduced) embedding 

> **Note:**
> The values in the `.npy` files are ordered by occurence of the respective image in corresponding metadata `.csv` file.

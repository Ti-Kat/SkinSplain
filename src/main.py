# Example of the classifier functions

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time

import numpy as np
    
from common.data_handling import generate_datasets, generate_dataloader, TRANSFORMS_BASE, TRANSFORMS_TRAIN
from common.modelling_base import ModelHandler, MODELNAME_BINARY, MODELNAME_MULTI

start_time = time.time()

full_dataset, train_dataset, val_dataset, test_dataset = generate_datasets(train_transform=TRANSFORMS_BASE,
                                                                           base_transform=TRANSFORMS_BASE,
                                                                           legacy_data=False) # Set this to True for old binary models
full_dataset_loader, train_dataset_loader, val_dataset_loader, test_dataset_loader \
    = generate_dataloader(full_dataset, train_dataset, val_dataset, test_dataset, weighted_sampling=False)

# Benign, Melanoma in Training, Melanoma out of Training
# image_names = ['ISIC_2637011', 'ISIC_0149568', 'ISIC_8355446']

model_handler = ModelHandler.load_models(dataset=full_dataset,
                                        #  time_stamp_path='2024_05_06-00_22_03', # Optional: Specifies model, otherwise uses most recent
                                         model_name=MODELNAME_BINARY) # Set this to MODELNAME_BINARY to load binary model model

image_name = 'ISIC_0149568'
image_label = np.array(full_dataset.meta_data['target'])[full_dataset.get_index_by_image_name(image_name)]
image = full_dataset.load_image(image_name=image_name, transform=False)
transformed_image = full_dataset.load_image(image_name=image_name, transform=True)

# Returns Melanoma Score (0-10)
# Either for already precomputed image (given image_name), or for potentially novel image
print("---------------------------------------------------------------------")
print(f'Melanoma Score for known image: {model_handler.get_melanoma_score(image_name=image_name)}')
print(f'Melanoma Score for novel image: {model_handler.get_melanoma_score(image=transformed_image)}')

# Returns Reliability Score (0-10)
# Either for already precomputed image (given image_name), or for potentially novel image 
# For 2nd case still uses image name to adjust its embedding for the current reliability score computation
print("---------------------------------------------------------------------")
print(f'Reliability Score for known image: {model_handler.get_reliability_score(image_name=image_name)}')
print(f'Reliability Score for novel image: {model_handler.get_reliability_score(image_name=image_name, image=image)}')

# Save image along with most similar with same and most similar with different label (save_plot=True is default)
# and returns the image names of both in that order as well
# May use ground truth to determine same / different label instead of model predictions (if use_ground_true=True)
print("---------------------------------------------------------------------")
model_handler.get_most_similar_images(image_name='ISIC_0052212')
model_handler.get_most_similar_images(image_name='ISIC_0149568',
                                      use_ground_truth=True)
model_handler.get_most_similar_images(image_name='ISIC_0157923',
                                      image=image)

# Save saliency map for given image in /img/saliency_map.png
# Either just overlayed image (single=True), or next to original image
# print("---------------------------------------------------------------------")
model_handler.get_saliency_map(image=image, single=True)

print(f"\nElapsed time: {round((time.time()-start_time), 3)} seconds")

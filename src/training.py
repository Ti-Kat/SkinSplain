# Trains new model and computes predictions + class probabilities + (reduced) embeddings.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time

import torch.backends.cudnn as cudnn
import torch

from common.modelling_base import ModelHandler, OODModel
from common.modelling_base import MODELNAME_BINARY, MODELNAME_MULTI
from common.data_handling import generate_datasets, generate_dataloader, TRANSFORMS_BASE, TRANSFORMS_TRAIN


def train_new_model(model_name=MODELNAME_BINARY):
    cudnn.benchmark = True

    torch.cuda.empty_cache()

    start_time = time.time()

    full_dataset, train_dataset, val_dataset, test_dataset = generate_datasets(train_transform=TRANSFORMS_TRAIN,
                                                                               base_transform=TRANSFORMS_BASE,
                                                                               split_all_vs_2018=True)
    full_dataset_loader, train_dataset_loader, val_dataset_loader, test_dataset_loader \
        = generate_dataloader(full_dataset, train_dataset, val_dataset, test_dataset, weighted_sampling=True)

    indices = {
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices,
    }

    model_handler = ModelHandler(dataset=full_dataset, indices=indices, model_name=model_name)
    model_handler.build_inference_model()
    model_handler.build_embedding_model()

    model_handler.inference_model.train_model(train_loader=train_dataset_loader, val_loader=test_dataset_loader)
    model_handler.inference_model.compute_predictions(input_data_loader=full_dataset_loader, set_attribute=True)

    model_handler.embedding_model.compute_embeddings(data_loader=full_dataset_loader, set_attribute=True)
    model_handler.embedding_model.compute_PCA(set_attribute=True)
    model_handler.embedding_model.compute_reduced_embeddings(set_attribute=True)

    model_handler.ood_model = OODModel(embeddings=model_handler.embedding_model.reduced_embeddings)

    model_handler.store_models()

    # Set train dataset to base transform for evaluation
    train_dataset.transform = TRANSFORMS_BASE

    for loader in (train_dataset_loader, test_dataset_loader):
        model_handler.inference_model.evaluate_model(test_loader=loader, write_to_csv=False)

    print(f"\nElapsed time: {round((time.time()-start_time), 3)} seconds")


if __name__ == '__main__':
    # Adjust argument to train either multiclass or binary classification model.
    train_new_model(model_name=MODELNAME_BINARY)

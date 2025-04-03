import albumentations
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .data_set import CustomImageDataset
from .path_constants import ISIC_META_DATA

IMAGE_SIZE = 448
LEGACY_DATA_SIZE = 33126 # TODO: Deprecated, for compatiblity with old binary models which used only ISIC 2020 data

TRANSFORMS_BASE = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.Normalize()
])

TRANSFORMS_TRAIN = albumentations.Compose([
    # albumentations.Transpose(p=0.2),
    albumentations.VerticalFlip(p=0.2),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    albumentations.OneOf([
        # albumentations.MotionBlur(blur_limit=5),
        # albumentations.MedianBlur(blur_limit=5),
        # albumentations.GaussianBlur(blur_limit=5),
        albumentations.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.3),

    # albumentations.OneOf([
    #     albumentations.OpticalDistortion(distort_limit=1.0),
    #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
    #     albumentations.ElasticTransform(alpha=3),
    # ], p=0.5),

    albumentations.CLAHE(clip_limit=4.0, p=0.3),
    albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
    albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.3),
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.CoarseDropout(max_height=int(IMAGE_SIZE * 0.05), max_width=int(IMAGE_SIZE * 0.05), max_holes=1, p=0.3),
    albumentations.Normalize()
])


def generate_full_dataset(transform=TRANSFORMS_BASE, legacy_data=False):
    return CustomImageDataset(csv_file=str(ISIC_META_DATA),
                              indices=None if not legacy_data else np.arange(LEGACY_DATA_SIZE),
                              transform=transform)


def generate_datasets(train_transform=TRANSFORMS_TRAIN, base_transform=TRANSFORMS_BASE, legacy_data=False, split_all_vs_2018=False):
    _full_dataset = generate_full_dataset(base_transform, legacy_data)
    dataset_size = len(_full_dataset.meta_data)

    if split_all_vs_2018 == False:
        # Get repeatable random permutation of indices
        rng = np.random.default_rng(seed=21)
        indices = rng.permutation(np.arange(dataset_size))

        # Define split sizes
        train_size = int(0.75 * dataset_size)
        val_size = int(0.15 * dataset_size)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    else:
        # Generate the indices
        indices = np.arange(dataset_size)
        split_index = 15356 # 2018 data starts here with ISIC_0024306
        split_length = int(dataset_size * 0.10) # 10% of the 2018 data with segmentation mask and meta info is used as holdout

        # Split indices
        train_indices = np.concatenate([indices[:split_index], indices[split_index + split_length:]])
        val_indices = indices[split_index:split_index+1]  # Workaround, no real validation set for final model to use all the non-test data
        test_indices = indices[split_index:split_index + split_length]

    _train_dataset = CustomImageDataset(csv_file=str(ISIC_META_DATA),
                                        transform=train_transform,
                                        indices=train_indices)

    _val_dataset = CustomImageDataset(csv_file=str(ISIC_META_DATA),
                                    transform=base_transform,
                                    indices=val_indices)

    _test_dataset = CustomImageDataset(csv_file=str(ISIC_META_DATA),
                                    transform=base_transform,
                                    indices=test_indices)

    print("Generated datasets")

    return _full_dataset, _train_dataset, _val_dataset, _test_dataset


def generate_dataloader(full, train, validate, test, weighted_sampling=False):
    sampler = RandomSampler(train)
    if weighted_sampling:
        # Compute sample weights for the training dataset to account for class imbalance
        targets = [full.meta_data.iloc[idx]['target'] for idx in train.indices]
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Prepare DataLoader
    batch_size = 64
    _full_loader = DataLoader(full, batch_size=batch_size, pin_memory=True, num_workers=8)
    _train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True, num_workers=8, sampler=sampler)
    _val_loader = DataLoader(validate, batch_size=batch_size, pin_memory=True, num_workers=8)
    _test_loader = DataLoader(test, batch_size=batch_size, pin_memory=True, num_workers=8)

    print("Generated dataloader")

    return _full_loader, _train_loader, _val_loader, _test_loader

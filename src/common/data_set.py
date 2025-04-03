import os
import cv2

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from .path_constants import ISIC_DATASET_MAPPING

class CustomImageDataset(Dataset):
    def __init__(self, csv_file: str, transform=None, indices=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Limit the number of images to be used in the dataset.
        """
        self.meta_data = pd.read_csv(csv_file)
        self.eval_mode = False
        # Filter meta_data to include only specified indices if provided
        if indices is not None:
            self.indices = indices
            self.meta_data = self.meta_data.iloc[indices]
        self.transform = transform

        self.meta_data['image_path'] = self.meta_data.apply(
            lambda row: os.path.join(ISIC_DATASET_MAPPING[row['dataset']], row['image_name']) + ".jpg",
            axis=1
        )

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_path = self.meta_data.iloc[idx, self.meta_data.columns.get_loc('image_path')]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform_image(image)
            
        target = self.meta_data.iloc[idx, self.meta_data.columns.get_loc('target')]
        target = torch.tensor(target, dtype=torch.long)

        if self.eval_mode is False:
            return {'image': image, 'target': target}
        else: # TODO: Currently does not work, and probably not needed in this form (?)
            return {
                'image': image,
                'target': target,
                'image_name': self.get_value_by_index('image_name', idx),
                'dataset': self.get_value_by_index('dataset', idx),
                'sex': self.get_value_by_index('sex', idx),
                'age_approx': self.get_value_by_index('age_approx', idx),
                'anatom_site_general': self.get_value_by_index('anatom_site_general', idx),
                'diagnosis': self.get_value_by_index('diagnosis', idx),
            }

    def transform_image(self, image):
        transformed = self.transform(image=image)
        image = transformed['image']
        return torch.from_numpy(image).float().permute(2, 0, 1)

    def set_eval_mode(self, value):
        """Sets eval_mode, which, if true, causes __getitem__ to also return metadata for each image"""
        self.eval_mode = value

    def get_image_path(self, image_name):
        idx = self.get_index_by_image_name(image_name)
        return self.meta_data.at[idx, 'image_path']

    def get_meta_data(self, image_name):
        return self.meta_data.loc[self.meta_data['image_name'] == image_name].tolist()[0]
    
    def get_value_by_index(self, attribute_name, idx, default=-1):
        value = self.meta_data.iloc[idx, self.meta_data.columns.get_loc(attribute_name)]
        return default if pd.isna(value) else value

    def get_index_by_image_name(self, image_name):
        return self.meta_data.index[self.meta_data['image_name'] == image_name].tolist()[0]
    
    def get_image_name_by_index(self, idx):
        return self.meta_data.at[idx, 'image_name']

    def __get_indices_by_attribute(self, column_name):
        result = {}
        if column_name not in self.meta_data.columns:
            raise ValueError(f"Column '{column_name}' not found in meta data")
        
        for value in self.meta_data[column_name].unique():
            result[value] = np.array(self.meta_data.index[self.meta_data[column_name] == value])
        
        return result

    def get_indices_by_dataset(self):
        """
        Get all indices from the dataset grouped by 'dataset'
        Key values: '2020_train', '2019_train', '2018_train', '2018_test', '2018_val'
        """
        return self.__get_indices_by_attribute('dataset')
    
    def get_indices_by_sex(self):
        """
        Get all indices from the dataset grouped by 'sex'
        Key values: 'male', 'female', 'unknown'
        """
        return self.__get_indices_by_attribute('sex')
    
    def get_indices_by_age(self):
        """
        Get all indices from the dataset grouped by 'age_approx'
        Key values: 0.0-90.0 (5.0 steps), 'unknown'
        """
        return self.__get_indices_by_attribute('age_approx')

    def get_indices_by_anatom_site(self):
        """
        Get all indices from the dataset grouped by 'anatom_site_general'
        Key values: 'head/neck', 'upper extremity', 'lower extremity', 'torso', 'palms/soles',
        'oral/genital', 'anterior torso', 'posterior torso', 'lateral torso', 'foot', 'hand', 'unknown'
        """
        return self.__get_indices_by_attribute('anatom_site_general')
    
    def get_indices_by_diagnosis(self):
        """
        Get all indices from the dataset grouped by 'diagnosis'
        Key values: 'UNK', 'NV', 'MEL', 'BKL', 'DF', 'SCC', 'BCC', 'VASC', 'AK'
        """
        return self.__get_indices_by_attribute('diagnosis')
    
    def get_indices_by_target(self):
        """
        Get all indices from the dataset grouped by 'target'
        Key values: 0-8
        """
        return self.__get_indices_by_attribute('target')

    def load_image(self, image_name, transform=True):
        """
        Load an individual image file by image name.
        """
        row = self.meta_data[self.meta_data['image_name'] == image_name]
        
        if row.empty:
            raise ValueError(f"No data found for image name: {image_name}")
        dataset = row['dataset'].iloc[0]  # Extract the scalar value
        
        image_path = os.path.join(ISIC_DATASET_MAPPING[dataset], image_name + '.jpg')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if transform == True and self.transform:
            return self.transform_image(image)
        else:
            return image

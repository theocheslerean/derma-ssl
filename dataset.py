'''
    Author: Theodor Cheslerean-Boghiu
    Date: May 26th 2023
    Version 1.0
'''
import os
import cv2
import numpy as np
import skimage
from PIL import Image
from typing import Union, Any

import pandas as pd

import torch
import torch.utils.data as data
import pytorch_lightning as pl
import albumentations

class SD198(data.Dataset):
    """Base implementation for the PAD UFES 20 dataset
    
    ~2400 pairs:
        - clinical image of skin lesion
        - 21 meta information points about the patient (ignored for now)
    
    Implements the data loading steps.
    
    Dataset structure:
        padufes/
            images/
                img1.png
                img2.png
            metadata.csv

    Attributes:
        diagnostic_label (list): List with class names in the order used by the model
        root_path (str): Root path where to load the pandas dataframes from 
        transform (albumentations.Compose): Composition of augmentation operations
        dataframe (pandas.Dataframe): Dataframe containing all the metadata, diagnosis and lesion image filepath information to load images and patient metadata
        targets (list): List of classification (diagnosis) labels
    """
    
    def __init__(self,
                 root_path: str,
                 transform: albumentations.BaseCompose,
                 resolution: int=224,
                 self_supervised: bool =False) -> None:
        """
            Args:
                root_path (str): _description_
                transform (albumentations.BaseCompose): _description_
                dataset_type (str, optional): _description_. Defaults to None.
        """ 
        super().__init__()
        
        self.resolution = resolution
        self.transform = transform
        self.root_path = root_path
        self.self_supervised = self_supervised
        
        # the downloaded ISIC dataset is one big folder with images and one csv file containing the meta-information        
        df_images = pd.read_csv(os.path.join(root_path, 'images.txt'), header=None, names=['img_id'], delimiter=' ')
        df_labels = pd.read_csv(os.path.join(root_path, 'image_class_labels.txt'), header=None, names=['diagnostic'], delimiter=' ')
        df_label_names = pd.read_csv(os.path.join(root_path, 'classes.txt'), header=None, names=['diagnostic'], delimiter=' ')
        
        self.dataframe = df_images.join(df_labels['diagnostic']).reset_index()
        # self.dataframe = pd.read_csv(os.path.join(root_path, 'balanced.csv'))
        
        # self.dataframe = self.dataframe[self.dataframe['diagnostic'] != 'MEL'].groupby('diagnostic').head(50).reset_index()
        # self.dataframe = self.dataframe.groupby('diagnostic').head(20).reset_index()
        
        self.diagnostic_label = df_label_names['diagnostic'].unique()
        
        # Convert diagnostic labels to unique integer numbers
        self.diagnostic_label_index = dict(zip(self.diagnostic_label, range(len(self.diagnostic_label))))
        self.targets = self.dataframe['diagnostic'].replace(self.diagnostic_label_index)
        
    def __len__(self):
        return len(self.dataframe)

    def get_image(self, row, label):
        
        # Get file path from dataframe in column 'isic_id'
        img_path = os.path.join(self.root_path, 'images', row['img_id'])

        try:
            img = skimage.io.imread(img_path, plugin='pil')[:,:,:3]
        except FileNotFoundError as err:
            print("We have an error")
            print(err)
            raise

        # Boilerplate code for resizing and converting to torch
        img = cv2.resize(img, (self.resolution, self.resolution))
        
        # Generate an image with no augmentation (not even the normalization) for visualization purposes when generating figures
        # Ignore this tensor during training
        img_orig = torch.from_numpy(np.transpose(np.array(img), (2, 0, 1)))

        # albumentations operations work directly on the numpy arrays
        if self.transform:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        label = np.array(label).astype(np.int64)

        return img_orig, img, label

    def __getitem__(self, idx):
        
        row = self.dataframe.iloc[idx]
        label = self.targets[idx]
        
        # Doing it twice means generating two agumentations for the same image
        return (self.get_image(row, label), self.get_image(row, label)) if self.self_supervised else self.get_image(row, label)

class PadUfes20(data.Dataset):
    """Base implementation for the PAD UFES 20 dataset
    
    ~2400 pairs:
        - clinical image of skin lesion
        - 21 meta information points about the patient (ignored for now)
    
    Implements the data loading steps.
    
    Dataset structure:
        padufes/
            images/
                img1.png
                img2.png
            metadata.csv

    Attributes:
        diagnostic_label (list): List with class names in the order used by the model
        root (str): Root path where to load the pandas dataframes from 
        transform (albumentations.Compose): Composition of augmentation operations
        dataframe (pandas.Dataframe): Dataframe containing all the metadata, diagnosis and lesion image filepath information to load images and patient metadata
        targets (list): List of classification (diagnosis) labels
    """
    
    def __init__(self,
                 root: str,
                 transform: albumentations.BaseCompose,
                 resolution: int=224,
                 sample_size: int=-1,
                 istest: bool=False) -> None:
        """
            Args:
                root_path (str): _description_
                transform (albumentations.BaseCompose): _description_
                dataset_type (str, optional): _description_. Defaults to None.
        """ 
        super().__init__()
        
        self.resolution = resolution
        self.transform = transform
        self.root = root
        
        # the downloaded ISIC dataset is one big folder with images and one csv file containing the meta-information
        self.dataframe = pd.read_csv(os.path.join(root, 'metadata.csv'))
        # self.dataframe = pd.read_csv(os.path.join(root_path, 'balanced.csv'))
        
        # self.dataframe = self.dataframe[self.dataframe['diagnostic'] != 'MEL'].reset_index()
        
        self.diagnostic_label = self.dataframe['diagnostic'].unique()
        # Convert diagnostic labels to unique integer numbers
        self.diagnostic_label_index = dict(zip(self.diagnostic_label, range(len(self.diagnostic_label))))
        
        if sample_size != -1:
            self.train_set = self.dataframe.groupby('diagnostic').sample(sample_size, random_state=200)
            self.val_set = self.dataframe.drop(self.train_set.index)
        
        self.dataframe = self.train_set
        if istest:
            self.dataframe = self.val_set
        
        self.dataframe.reset_index(inplace=True)
        self.targets = self.dataframe['diagnostic'].replace(self.diagnostic_label_index)
        
    def __len__(self):
        return len(self.dataframe)

    def get_image(self, row, label):
        
        # Get file path from dataframe in column 'isic_id'
        img_path = os.path.join(self.root, 'images', row['img_id'])

        try:
            img = skimage.io.imread(img_path, plugin='pil')[:,:,:3]
        except FileNotFoundError as err:
            print("We have an error")
            print(err)
            raise

        # Boilerplate code for resizing and converting to torch
        img = cv2.resize(img, (self.resolution, self.resolution))
        
        # Generate an image with no augmentation (not even the normalization) for visualization purposes when generating figures
        # Ignore this tensor during training
        img_orig = torch.from_numpy(np.transpose(np.array(img), (2, 0, 1)))

        # albumentations operations work directly on the numpy arrays
        if self.transform:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        label = np.array(label).astype(np.int64)

        return img_orig, img, label

    def __getitem__(self, idx):
        
        row = self.dataframe.iloc[idx]
        label = self.targets[idx]
        
        # Doing it twice means generating two agumentations for the same image
        return self.get_image(row, label)
    
class KUMData(data.Dataset):
    """Base implementation for the PAD UFES 20 dataset
    
    ~2400 pairs:
        - clinical image of skin lesion
        - 21 meta information points about the patient (ignored for now)
    
    Implements the data loading steps.
    
    Dataset structure:
        padufes/
            images/
                img1.png
                img2.png
            metadata.csv

    Attributes:
        diagnostic_label (list): List with class names in the order used by the model
        root (str): Root path where to load the pandas dataframes from 
        transform (albumentations.Compose): Composition of augmentation operations
        dataframe (pandas.Dataframe): Dataframe containing all the metadata, diagnosis and lesion image filepath information to load images and patient metadata
        targets (list): List of classification (diagnosis) labels
    """
    
    def __init__(self,
                 root: str,
                 transform: albumentations.BaseCompose,
                 resolution: int=224,
                 sample_size: int=-1,
                 istest: bool=False) -> None:
        """
            Args:
                root (str): _description_
                transform (albumentations.BaseCompose): _description_
                dataset_type (str, optional): _description_. Defaults to None.
        """ 
        super().__init__()
        
        self.resolution = resolution
        self.transform = transform
        self.root = root
        
        # the downloaded ISIC dataset is one big folder with images and one csv file containing the meta-information
        self.dataframe = pd.read_csv(os.path.join(root, 'metadata.csv'))
        
        self.diagnostic_label = self.dataframe['diagnostic'].unique()
        # Convert diagnostic labels to unique integer numbers
        self.diagnostic_label_index = dict(zip(self.diagnostic_label, range(len(self.diagnostic_label))))
        
        if sample_size != -1:
            self.train_set = self.dataframe.groupby('diagnostic').sample(frac=sample_size/100.0, random_state=200)
            self.val_set = self.dataframe.drop(self.train_set.index)
        
        self.dataframe = self.train_set
        if istest:
            self.dataframe = self.val_set
        
        self.dataframe.reset_index(inplace=True)
        self.targets = self.dataframe['diagnostic'].replace(self.diagnostic_label_index)
        
    def __len__(self):
        return len(self.dataframe)

    def get_image(self, row, label):
        
        # Get file path from dataframe in column 'isic_id'
        img_path = os.path.join(self.root, 'images', row['Name'])

        try:
            img = skimage.io.imread(img_path, plugin='pil')[:,:,:3]
        except FileNotFoundError as err:
            print("We have an error")
            print(err)
            raise

        # Boilerplate code for resizing and converting to torch
        img = cv2.resize(img, (self.resolution, self.resolution))
        
        # Generate an image with no augmentation (not even the normalization) for visualization purposes when generating figures
        # Ignore this tensor during training
        img_orig = torch.from_numpy(np.transpose(np.array(img), (2, 0, 1)))

        # albumentations operations work directly on the numpy arrays
        if self.transform:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
        label = np.array(label).astype(np.int64)

        return img_orig, img, label

    def __getitem__(self, idx):
        
        row = self.dataframe.iloc[idx]
        label = self.targets[idx]
        
        return self.get_image(row, label)
             
class WrapperDataset(pl.LightningDataModule):
    """Main implementation of the full dataset
    
    Generates splits at initialization. 
    Generates the DataLoaders.

    Attributes:
        data_dir (str): Directory path for the data
        batch_size (int): Integer denoting the batch size needed by the Dataloader constructor
        transform_train (albumentations.Compose): Composition of augmentation operations for the training set
        transform_test (albumentations.Compose): Composition of augmentation operations for the training set. Will only contain the normalization operation.
        train_data: Training Dataset
        val_data: Validation Dataset
        test_data: Test Dataset (#TODO: for now it is just the validation split)
        num_classes (int): Total number of classes
    """    

    def __init__(self,
                 dataset_name: str=None,
                 root: str=None,
                 batch_size: int=16,
                 resolution: int=224,
                 transforms: Union[albumentations.BaseCompose, Any]=None,
                 sample_size: int=50):
        """_summary_

        Args:
            data_dir (str, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 16.
            sampling (str, optional): _description_. Defaults to None.
            normalize_weights (bool, optional): _description_. Defaults to True.
            metadata_choice (str, optional): _description_. Defaults to 'all'.
        """        
        super().__init__()
        
        self.root = root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.transform_train = transforms

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        self.transform_test = albumentations.Compose(
            [
                albumentations.Normalize(mean=norm_mean, std=norm_std, always_apply=True),
            ])
        
        dataset_dict = {
            'kum' : KUMData,
            'pad' : PadUfes20,
            'sd198' : SD198
        }
    
        self.train_data = dataset_dict[self.dataset_name](
            self.root,
            transform=self.transform_train,
            resolution=resolution,
            sample_size=sample_size,
            istest=False)
        
        self.val_data = dataset_dict[self.dataset_name](
            self.root,
            transform=self.transform_test,
            resolution=resolution,
            sample_size=sample_size,
            istest=True)
        
        print(f"Class count in training set: {self.train_data.dataframe['diagnostic'].value_counts()}")
        print(f"Class count in validation set: {self.val_data.dataframe['diagnostic'].value_counts()}")
       
        self.diagnostic_labels = self.train_data.diagnostic_label
        self.logits_size = len(self.diagnostic_labels)
        self.class_weights = np.ones_like(self.diagnostic_labels)
        
        _, nrs = np.unique(
            self.train_data.targets, return_counts=True)
        
        nrs = nrs.astype(np.float32)
                
        self.class_weights = np.ones_like(nrs).astype(np.float32)
        self.sampling_weights = np.ones_like(nrs).astype(np.float32) #/ (nrs / np.sum(nrs))
        
        print(f'loss weights: {self.class_weights}')
        print(f'sampling weights: {self.sampling_weights}')
        
        train_samples_weight = torch.from_numpy(np.array([self.sampling_weights[
            self.train_data.targets[i]] for i in range(len(self.train_data))]))
    
        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            train_samples_weight, int(len(self.train_data)), replacement=True)
            

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=(self.batch_size if self.batch_size <= 16 else 16),
                                           sampler=None,
                                           pin_memory=True,
                                           drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           num_workers=(self.batch_size if self.batch_size <= 16 else 16),
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=True)
        
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False)

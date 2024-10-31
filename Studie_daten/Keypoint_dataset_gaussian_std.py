import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import pydicom
from skimage.exposure import match_histograms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import pandas as pd


def get_image():
    img_dir = r'/home/alex/Documents/new try/Data/losw-dose-simulation/Phantom'
    file_path = os.path.join(img_dir, '650.dcm')
    ds = pydicom.dcmread(file_path, force=True)
    ds_center = ds.WindowCenter
    ds_width = ds.WindowWidth
    dcm_img = ds.pixel_array
    image = (dcm_img - (ds_center - ds_width/2)) / ds_width
    image = np.clip(image, a_min=0, a_max=1)
    return np.uint8(255*image), dcm_img, ds_center, ds_width


class Keypoint_dataset_poisson(Dataset):
    def __init__(self, df, image_std, noise_std_ratio, transform=None):
        self.transform = transform
        self.image_std = image_std
        self.noise_std_ratio = noise_std_ratio
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df['new path'][idx]
        img = np.array(Image.open(img_path).convert('L'))
        s = np.random.normal(130, self.image_std, (2995, 2993)).astype(int)
        matched = match_histograms(img, s)
        keypoints = [(self.df['X_Th1_Original'][idx], self.df['Y_Th1_Original'][idx]),
                    (self.df['X_Th12_Original'][idx], self.df['Y_Th12_Original'][idx]),
                    (self.df['X_LeftRecess_Original'][idx], self.df['Y_LeftRecess_Original'][idx]),
                    (self.df['X_RightRecess_Original'][idx], self.df['Y_RightRecess_Original'][idx])] #T1_original, T12_original, LeftRecess_original, RightRecess_original
        if self.transform is not None:
            transformed = self.transform(image= matched, keypoints=keypoints)
            matched = transformed['image']
            gaussian_noise = torch.normal(0, self.image_std / self.noise_std_ratio, matched.shape)
            matched_gaussian = matched + gaussian_noise
            matched_gaussian = np.clip(matched_gaussian, a_min = 0, a_max= 255)
            keypoints = transformed['keypoints']
            keypoints = torch.from_numpy(np.float32(keypoints).flatten())

        return matched_gaussian.type(torch.float)/255, keypoints.unsqueeze(0)

transform_padding = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CropAndPad(px = 40, keep_size=True),
        A.Rotate(limit= 15, p = 0.2),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)

transform_test_padding = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CropAndPad(px = 40, keep_size=True),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)

def get_loader_keypoint_simulated(BATCH_SIZE, image_std, noise_std_ratio):
    df_path = r'/home/alex/Documents/new try/Data/Studie_daten/Study_data.csv'
    df = pd.read_csv(df_path, sep = ';')
    train_indices, test_indices = train_test_split(df.index, test_size=0.3, random_state=42)
    test_indices, validation_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

    train_df = df.loc[train_indices].reset_index(drop = True)
    test_df = df.loc[test_indices].reset_index(drop = True)
    validation_df = df.loc[validation_indices].reset_index(drop = True)
    
    train_keypoint_ds = Keypoint_dataset_poisson(train_df,
                        image_std, 
                        noise_std_ratio,
                        transform= transform_padding)
    test_keypoint_ds = Keypoint_dataset_poisson(test_df,
                        image_std, 
                        noise_std_ratio,
                        transform= transform_test_padding)
    validation_keypoint_ds = Keypoint_dataset_poisson(validation_df,
                        image_std, 
                        noise_std_ratio,
                        transform= transform_test_padding)

    train_loader = DataLoader(train_keypoint_ds,
                          batch_size= BATCH_SIZE,
                          num_workers= 0,
                          pin_memory= False,
                          shuffle= True)

    test_loader = DataLoader(test_keypoint_ds,
                          batch_size= BATCH_SIZE,
                          num_workers= 0,
                          pin_memory= False,
                          shuffle= False)
    
    validation_loader = DataLoader(validation_keypoint_ds,
                          batch_size= BATCH_SIZE,
                          num_workers= 0,
                          pin_memory= False,
                          shuffle= False)
    
    return train_loader, test_loader, validation_loader

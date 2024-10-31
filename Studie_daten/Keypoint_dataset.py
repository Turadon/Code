import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Keypoint_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df['new path'][idx]
        img = np.array(Image.open(img_path).convert('L'))
        keypoints = [(self.df['X_Th1_Original'][idx], self.df['Y_Th1_Original'][idx]),
                    (self.df['X_Th12_Original'][idx], self.df['Y_Th12_Original'][idx]),
                    (self.df['X_LeftRecess_Original'][idx], self.df['Y_LeftRecess_Original'][idx]),
                    (self.df['X_RightRecess_Original'][idx], self.df['Y_RightRecess_Original'][idx])] #T1_original, T12_original, LeftRecess_original, RightRecess_original
        if self.transform is not None:
            transformed = self.transform(image= img, keypoints=keypoints)
            img = transformed['image']
            keypoints = transformed['keypoints']
            keypoints = torch.from_numpy(np.float32(keypoints).flatten())

        return img.type(torch.float)/255, keypoints.unsqueeze(0)
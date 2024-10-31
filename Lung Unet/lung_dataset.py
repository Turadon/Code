import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LungDataset(Dataset):
    def __init__(self, dir_images, dir_masks, transform=None):
        self.transform = transform
        self.dir_images = dir_images
        self.dir_masks = dir_masks
        self.images = sorted(os.listdir(dir_images))
        self.masks = sorted(os.listdir(dir_masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_images, self.images[idx])
        mask_path = os.path.join(self.dir_masks, self.masks[idx])
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        if self.transform is not None:
            transform_output = (self.transform(image=image, mask=mask))
            image = transform_output["image"]
            mask = transform_output["mask"]
        return image/255, mask.unsqueeze(0)/255

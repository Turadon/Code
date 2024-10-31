import torch
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from unet import UNET
from lung_dataset import LungDataset
import torch.nn as nn

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    # save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = False  # True if DEVICE == 'cuda' else False
LOAD_MODEL = False
LOGGING = True
BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_WORKERS = 0
IMAGE_SIZE = 256
TRAIN_IMAGE_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/train/images' #r'.\data\train\images'
TRAIN_MASKS_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/train/masks' #r'.\data\train\masks'
TEST_IMAGE_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/test/images' #r'.\data\test\images'
TEST_MASKS_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/test/masks' #r'.\data\test\masks'


def train_epoch(loader, model, optimizer, loss_fn, scaler, DEVICE='cuda'):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    del predictions
    torch.cuda.empty_cache() 
    return loss


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Rotate(limit=20, p=1.0),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ],
    )

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASKS_DIR,
        TEST_IMAGE_DIR,
        TEST_MASKS_DIR,
        BATCH_SIZE,
        train_transform,
        test_transform,
        IMAGE_SIZE=IMAGE_SIZE,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(0, test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, NUM_EPOCHS+1):
        train_epoch(train_loader, model, optimizer, loss_fn, scaler)

        # check accuracy
        check_accuracy(epoch, test_loader, model, device=DEVICE)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(
        checkpoint, r'.\save_states\own_unet_test.pth.tar')

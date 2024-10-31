import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Keypoint_dataset import Keypoint_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision.ops import (
    masks_to_boxes,
    generalized_box_iou
)

def save_checkpoint(state, filename):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(model, filename):
    print('=> Loading checkpoint')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])


resize_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)

transform_test = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(mean = (150,), std= (65,), max_pixel_value=1.0),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)

transform_padding = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CropAndPad(px = 40, keep_size=True),
        A.Normalize(mean = (150,), std= (65,), max_pixel_value=1.0),
        A.Rotate(limit= 15, p = 0.2),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)

transform_test_padding = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CropAndPad(px = 40, keep_size=True),
        A.Normalize(mean = (150,), std= (65,), max_pixel_value=1.0),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)


def get_test_dataset():
    df_path = r'/home/alex/Documents/new try/Data/Studie_daten/Study_data.csv'
    df = pd.read_csv(df_path, sep = ';')
    _, test_indices = train_test_split(df.index, test_size=0.3, random_state=42)
    test_df = df.loc[test_indices].reset_index(drop = True)

    test_keypoint_ds = Keypoint_dataset(test_df,
                        transform= transform_test)  
    return test_keypoint_ds


def get_loader_keypoint(BATCH_SIZE):
    df_path = r'/home/alex/Documents/new try/Data/Studie_daten/Study_data.csv'
    df = pd.read_csv(df_path, sep = ';')
    train_indices, test_indices = train_test_split(df.index, test_size=0.3, random_state=42)
    test_indices, validation_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

    train_df = df.loc[train_indices].reset_index(drop = True)
    test_df = df.loc[test_indices].reset_index(drop = True)
    validation_df = df.loc[validation_indices].reset_index(drop = True)
    
    train_keypoint_ds = Keypoint_dataset(train_df,
                        transform= transform_padding)
    test_keypoint_ds = Keypoint_dataset(test_df,
                        transform= transform_test_padding)
    validation_keypoint_ds = Keypoint_dataset(validation_df,
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


def get_acc(model, data_loader, loss_func, DEVICE = 'cuda:0'):
    model.eval()
    loss = 0
    IoU = 0
    with torch.no_grad():
        for image, targ in data_loader:
            img = image.to(device = DEVICE)
            target = targ.to(DEVICE)
            pred = model(img)
            loss += loss_func(pred, target.squeeze(1))
            IoU += keypoints_IoU_mult(image, targ.squeeze(1), pred)
    model.train()
    print('loss:', loss.item()/len(data_loader), 'BoxIoU:', IoU/len(data_loader.dataset))
    return loss.item()/len(data_loader), IoU/len(data_loader.dataset)

def get_acc_EfficentNet(model, data_loader, DEVICE = 'cuda:0'):
    loss_func = nn.MSELoss()
    model.eval()
    loss = 0
    IoU = 0
    with torch.no_grad():
        for image, targ in data_loader:
            img = image.to(device = DEVICE).repeat(1, 3, 1, 1)
            target = targ.to(DEVICE)
            pred = model(img)
            loss += loss_func(pred, target.squeeze(1))
            IoU += keypoints_IoU_mult(image, targ.squeeze(1), pred)
    model.train()
    print('loss:', loss/len(data_loader), 'BoxIoU:', IoU/len(data_loader.dataset))
    return loss.item()/len(data_loader), IoU/len(data_loader.dataset)


def keypoints_IoU_single(test_image, true_kp, pred_kp):
    '''
    ínput:
        test_image: torch [1, 1, N, M]
        true_kp: torch [1, 8]
        pred_kp: torch [1, 8]
    output:
        IoU score: float rounded to 4 
    '''
    boxes = []
    true_kp = true_kp.detach().numpy().astype(np.uint8)
    pred_kp = pred_kp.detach().numpy().astype(np.uint8)
    for keypoints in [true_kp, pred_kp]:
        keypoint_image = torch.zeros(test_image[0,0].shape)
        for i in range(0,8,2):
            keypoint_image[keypoints[i+1],keypoints[i]] = 1
        boxes.append(masks_to_boxes(keypoint_image.unsqueeze(0)))
    test_box, pred_box = boxes[0][0], boxes[1][0]
    box_iou = generalized_box_iou(boxes[0], boxes[1])
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    return box_iou


def keypoints_IoU_mult(test_image, true_kp, pred_kp):
    '''
    ínput:
        test_image: torch [K, 1, N, M]
        true_kp: torch [K, 8]
        pred_kp: torch [K, 8]
    output:
        IoU score: float rounded to 4 
    '''
    K = test_image.shape[0]
    boxes = []
    true_kp = true_kp.detach().cpu().numpy().astype(np.uint8)
    pred_kp = pred_kp.detach().cpu().numpy().astype(np.uint8)
    for keypoints in [true_kp, pred_kp]:
        keypoint_image = torch.zeros(test_image.shape)
        for k in range(K):
            for i in range(0,8,2):
                keypoint_image[k,0,keypoints[k,i+1],keypoints[k,i]] = 1
        boxes.append(masks_to_boxes(keypoint_image.squeeze(1)))
    box_iou = generalized_box_iou(boxes[0], boxes[1])
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    return box_iou


def plot_keypoints_IoU(test_image, true_kp, pred_kp):
    '''
    ínput:
        test_image: torch [K, 1, N, M]
        true_kp: torch [K, 8]
        pred_kp: torch [K, 8]
    output:
        Show test_image wiht keypoints and bounding boxes
    '''
    fig, axs = plt.subplots(1, 1, figsize=(30, 10))
    axs.imshow(test_image[0,0], cmap ='gray', vmin = 0, vmax = 1)
    pred_box = masks_to_boxes(keypoint_image.unsqueeze(0))[0]
    boxes = []
    true_kp = true_kp.detach().numpy().astype(np.uint8)
    for keypoints in [true_kp, pred_kp]:
        keypoint_image = torch.zeros(test_image[0,0].shape)
        for i in range(0,8,2):
            keypoint_image[keypoints[i+1],keypoints[i]] = 1
        boxes.append(masks_to_boxes(keypoint_image.unsqueeze(0)))
    test_box, pred_box = boxes[0][0], boxes[1][0]
    for i in range(0,8,2):
        axs.scatter(pred_kp[i],pred_kp[i+1], color = 'C0')
        axs.scatter(true_kp[i],true_kp[i+1], color = 'C1')
    rect_pred = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
                                  pred_box[3] - pred_box[1], linewidth=4, edgecolor='C0', facecolor='none', label='predicted crop')
    rect_test = patches.Rectangle((test_box[0], test_box[1]), test_box[2] - test_box[0],
                                  test_box[3] - test_box[1], linewidth=4, edgecolor='C1', facecolor='none', label='true crop')

    #pred_rect_copy = copy(rect_pred)
    axs.add_patch(rect_pred)
    axs.add_patch(rect_test)
    box_iou = generalized_box_iou(boxes[0], boxes[1])
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    plt.title(f'Box IoU: {box_iou:.2f}')
    plt.legend()
    plt.show()


def plot_keypoints_IoU(image, kp, pred_kp):
    '''
    image: (B, C, 256, 256)
    kp:  torch.Size([B, 8]))
    pred_kp: torch.Size([B, 8])
    '''
    #keypoint_image = torch.zeros(image.shape)
    #keypoint_image = keypoint_image[0,0]
    #for i in range(0,8,2):
    #    keypoint_image[pred_kp[i+1],pred_kp[i+1]] = 1
    fig, axs = plt.subplots(1, 1, figsize=(30, 10))
    axs.imshow(image[0,0], cmap ='gray')
    #pred_box = masks_to_boxes(keypoint_image.unsqueeze(0))[0]
    boxes = []
    pred_kp = pred_kp.cpu().detach().numpy().astype(np.uint8)
    kp = kp.detach().numpy().astype(np.uint8)
    for keypoints in [kp, pred_kp]:
        keypoint_image = torch.zeros(image[0,0].shape)
        for i in range(0,8,2):
            keypoint_image[keypoints[0,i+1],keypoints[0,i]] = 1
        boxes.append(masks_to_boxes(keypoint_image.unsqueeze(0)))
    test_box, pred_box = boxes[0][0], boxes[1][0]
    for i in range(0,8,2):
        axs.scatter(pred_kp[0,i],pred_kp[0,i+1], color = 'C0')
        axs.scatter(kp[0,i],kp[0,i+1], color = 'C1')
    rect_pred = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
                                    pred_box[3] - pred_box[1], linewidth=4, edgecolor='C0', facecolor='none', label='predicted crop')
    rect_test = patches.Rectangle((test_box[0], test_box[1]), test_box[2] - test_box[0],
                                    test_box[3] - test_box[1], linewidth=4, edgecolor='C1', facecolor='none', label='true crop')

    #pred_rect_copy = copy(rect_pred)
    axs.add_patch(rect_pred)
    axs.add_patch(rect_test)
    box_iou = generalized_box_iou(boxes[0], boxes[1])
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    plt.title(f'Box IoU: {box_iou:.2f}')
    plt.legend()
    plt.show()
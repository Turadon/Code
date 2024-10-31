import os
import torch
from lung_dataset import LungDataset
from torch.utils.data import DataLoader, Subset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch.nn import BCEWithLogitsLoss
from torchvision.ops import (
    masks_to_boxes,
    generalized_box_iou
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
from copy import copy
import wandb
from boundingbox import get_largest_segments_bounding_box, metric_largest_box_iou
import numpy as npff

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = False  # True if DEVICE == 'cuda' else False
LOGGING = True
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 0
IMAGE_SIZE = 256
TRAIN_IMAGE_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/train/images' #r'.\data\train\images'
TRAIN_MASKS_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/train/masks' #r'.\data\train\masks'
TEST_IMAGE_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/test/images' #r'.\data\test\images'
TEST_MASKS_DIR = r'/home/alex/Documents/new try/Data/Lung Unet/data/test/masks' #r'.\data\test\masks'


def save_checkpoint(state, filename):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(model, filename):
    print('=> Loading checkpoint')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])


def log_test(epoch, loss, acc, dice, iou, hd):
    wandb.log(
        {
            "epoch": epoch,
            "test_loss": loss,
            "test_acc": acc,
            "test_dice": dice,
            "test_box_iou": iou,
            "test_hd": hd
        }
    )


def log_val(epoch, loss, acc, dice, iou, hd):
    wandb.log(
        {
            "epoch": epoch,
            "val_loss": loss,
            "val_acc": acc,
            "val_dice": dice,
            "val_box_iou": iou,
            "val_hd": hd
        }
    )


def create_paths(model_name, i, features, image_size):
    SAVE_IMAGE_PATH = r'./home/alex/Documents/new try/Data/Lung Unet/eval_images'
    SAVE_MODEL_PATH = r'/home/alex/Documents/new try/Data/Lung Unet/save_states'
    featur_string = '_'.join(map(str, features))
    save_image_path = os.path.join(
        SAVE_IMAGE_PATH, (model_name + '_' + str(i) + '_' + str(image_size) + '_' + featur_string))
    save_model_path = os.path.join(
        SAVE_MODEL_PATH, (model_name) + '_' + str(i) + '_' + str(image_size) + '_' + featur_string + '.pth.tar')
    return save_model_path, save_image_path


def get_bounding_box(mask):
    '''
    input: mask: Tensor[N, 1, X, Y]
    output: box: Tensor[N,4]
    '''
    try:
        box = masks_to_boxes(mask.squeeze(1))
    except:
        box = torch.zeros(mask.size()[0], 4)
    return box


def metric_box_iou(pred, mask):
    '''
    input: pred and mask of model: Tensor[N, 1, X, Y]
    output: sum of box_iou Tensor
    '''
    box_1 = get_bounding_box(pred).to(DEVICE)
    box_2 = get_bounding_box(mask).to(DEVICE)
    box_iou = generalized_box_iou(box_1, box_2)
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    return box_iou


def check_metrics(test_masks, preds):
    test_masks = test_masks.to('cpu')
    preds = preds.to('cpu')
    metric_hd = HausdorffDistanceMetric(percentile=95)
    metric_BCEL = BCEWithLogitsLoss()
    metric_dice = DiceMetric(include_background=True, reduction="mean")
    metric_dice.reset()
    metric_hd.reset()
    num_correct = (preds == test_masks).sum()
    num_pixels = torch.numel(preds)
    dice_score = metric_dice(test_masks, preds).sum()
    BCEL = metric_BCEL(test_masks, preds).sum()
    hausdorff_distance = metric_hd(test_masks, preds).sum()
    iou_score = metric_largest_box_iou(test_masks, preds)

    test_metrics = f"Loss: {BCEL:.2f}, Acc: {num_correct/num_pixels:.2f}, and Dice score: {(dice_score):.2f}, IoU: {iou_score:.2f}, hd: {hausdorff_distance:.2f}"

def dice_score_test(mask1, mask2, smooth=1e-6):
    # Flatten the masks
    mask1_flat = mask1.contiguous().view(-1)
    mask2_flat = mask2.contiguous().view(-1)
    
    # Compute intersection and sum of both masks
    intersection = (mask1_flat * mask2_flat).sum()
    total = mask1_flat.sum() + mask2_flat.sum()
    
    # Dice score calculation
    dice = (2.0 * intersection + smooth) / (total + smooth)
    
    return dice

def test_dice(loader, model, device="cuda"):
    dice_score = 0
    metric_dice = DiceMetric(include_background=True, reduction="mean")
    metric_dice.reset()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            print(dice_score_test(y, preds))


def check_accuracy(epoch, loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    hausdorff_distance = 0
    BCEL = 0
    n = len(loader.sampler)
    metric_hd = HausdorffDistanceMetric(percentile=95)
    metric_hd.reset()
    metric_dice = DiceMetric(include_background=True, reduction="mean")
    metric_dice.reset()
    metric_BCEL = BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            BCEL += metric_BCEL(y, preds).sum()
            iou_score += metric_largest_box_iou(y.cpu(), preds.cpu())
            metric_dice(y, preds)
            metric_hd(y, preds)

    dice_score = metric_dice.aggregate().item()
    hausdorff_distance = metric_hd.aggregate().item()

    print(
        f"Epoch: {epoch}, Acc: {(num_correct/num_pixels):.2f}, and Dice score: {(dice_score):.2f}, IoU: {(iou_score/n):.2f}, hd: {(hausdorff_distance):.2f}"
    )
    model.train()
    del x, y, preds, num_correct, num_pixels, dice_score, iou_score, hausdorff_distance
    torch.cuda.empty_cache()

def check_accuracy_val(epoch, loader, model, logging=False, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    hausdorff_distance = 0
    BCEL = 0
    n = len(loader.sampler)
    metric_hd = HausdorffDistanceMetric(percentile=95)
    metric_hd.reset()
    metric_dice = DiceMetric(include_background=True, reduction="mean")
    metric_dice.reset()
    metric_BCEL = BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            BCEL += metric_BCEL(y, preds).sum()
            iou_score += metric_largest_box_iou(y.cpu(), preds.cpu())
            metric_dice(y, preds)
            metric_hd(y, preds)

    dice_score = metric_dice.aggregate().item()
    hausdorff_distance = metric_hd.aggregate().item()
    print(
        f"Val-Epoch: {epoch}, Acc: {(num_correct/num_pixels):.2f}, and Dice score: {(dice_score):.2f}, IoU: {(iou_score/n):.2f}, hd: {(hausdorff_distance):.2f}"
    )
    return BCEL


def check_accuracy_test(epoch, loader, model, logging=False, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    hausdorff_distance = 0
    BCEL = 0
    n = len(loader.sampler)
    metric_hd = HausdorffDistanceMetric(percentile=95)
    metric_hd.reset()
    metric_dice = DiceMetric(include_background=True, reduction="mean")
    metric_dice.reset()
    metric_BCEL = BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            BCEL += metric_BCEL(y, preds).sum()
            iou_score += metric_largest_box_iou(y.cpu(), preds.cpu())
            metric_dice(y, preds)
            metric_hd(y, preds)

    dice_score = metric_dice.aggregate().item()
    hausdorff_distance = metric_hd.aggregate().item()

    print(
        f"Epoch: {epoch}, Acc: {(num_correct/num_pixels):.2f}, and Dice score: {(dice_score):.2f}, IoU: {(iou_score/n):.2f}, hd: {(hausdorff_distance):.2f}"
    )
    if logging:
        log_test(epoch,
                 BCEL,
                 num_correct/num_pixels,
                 dice_score,
                 iou_score/n,
                 hausdorff_distance
                 )
    else:
        pass
    model.train()
    del x, y, preds, num_correct, num_pixels, dice_score, iou_score, hausdorff_distance
    torch.cuda.empty_cache() 
    return BCEL

TRAIN_TRANSFORM = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Rotate(limit=30, p=0.8),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ], 
)

TEST_TRANSFORM = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        ToTensorV2(),
    ],
)

transform_padding = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CropAndPad(px = 40, keep_size=True),
        A.Rotate(limit= 15, p = 0.5),
        ToTensorV2(),
    ],
)

transform_test_padding = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CropAndPad(px = 40, keep_size=True),
        ToTensorV2(),
    ],
)

def get_loaders(
    train_dir=TRAIN_IMAGE_DIR,
    train_maskdir=TRAIN_MASKS_DIR,
    test_dir=TEST_IMAGE_DIR,
    test_maskdir=TEST_MASKS_DIR,
    batch_size=BATCH_SIZE,
    train_transform=transform_padding,
    test_transform=transform_test_padding,
    num_workers=0,
    pin_memory=False
):
    train_ds = LungDataset(
        train_dir,
        train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = LungDataset(
        test_dir,
        test_maskdir,
        transform=test_transform,
    )

    test_size =  test_ds.__len__()
    validation_indices = list(range(test_size))[:int(test_size/2)]
    test_indices = list(range(test_size))[int(test_size/2):]

    test_loader = DataLoader(
        Subset(test_ds, test_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    validation_loader = DataLoader(
        Subset(test_ds, validation_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader, validation_loader

def plot_image_mask_pred(model, test_image, test_mask, device="cuda"):
    x = test_image.to(device)
    y = test_mask.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs[0].imshow(test_image[0, 0], cmap='gray')
    axs[0].set_title('X-ray image', fontsize=20)

    axs[1].imshow(test_mask[0, 0], cmap='gray')
    axs[1].set_title('Original mask', fontsize=20)

    axs[2].imshow(preds[0, 0].cpu(), cmap='gray')
    axs[2].set_title('Predicted mask', fontsize=20)

    try:
        fig.suptitle(check_metrics(y[0].unsqueeze(
        0), preds[0].unsqueeze(0)), fontsize=30)
    except:
        pass
    plt.legend()
    plt.show()
    #plt.close(fig)
    return fig

def plot_image_mask_box_pred_box(model, test_image, test_mask, device="cuda"):
    x = test_image.to(device)
    y = test_mask.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    boxes_mask = get_largest_segments_bounding_box(test_mask)[0]
    boxes_pred = get_largest_segments_bounding_box(preds.cpu())[0]

    rect_pred = patches.Rectangle((boxes_pred[0], boxes_pred[1]), boxes_pred[2] - boxes_pred[0],
                                  boxes_pred[3] - boxes_pred[1], linewidth=4, edgecolor='orange', facecolor='none', label='predicted FOV')
    rect_mask = patches.Rectangle((boxes_mask[0], boxes_mask[1]), boxes_mask[2] - boxes_mask[0],
                                  boxes_mask[3] - boxes_mask[1], linewidth=4, edgecolor='dodgerblue', facecolor='none', label='optimal FOV')

    fig = plt.figure(figsize=(10, 10))  # Set the figure size
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 2])  # Make the top row smaller

    # Top image (smaller)
    ax1 = fig.add_subplot(gs[0, :])  # First row, spanning both columns
    ax1.imshow(test_image[0, 0], cmap='gray', vmin = 0, vmax =1)
    ax1.set_title("(a)", size = 16)
    ax1.axis('off')

    # Bottom left image (larger)
    ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
    ax2.imshow(test_mask[0, 0], cmap='gray')
    ax2.set_title("(b)", size = 16)
    ax2.axis('off')

    # Bottom right image (larger)
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax3.imshow(preds[0, 0].cpu(), cmap='gray')
    ax3.set_title("(c)", size = 16)
    ax3.axis('off')

    for ax in [ax1, ax2, ax3]:
        pred_rect_copy = copy(rect_pred)
        mask_rect_copy = copy(rect_mask)
        ax.add_patch(mask_rect_copy)
        ax.add_patch(pred_rect_copy)
    
    ax1.legend(loc='lower left', fontsize = 16)
    #plt.close(fig)
    plt.tight_layout()
    plt.show()
    return fig

def plot_image_mask_box_pred_box_tolerance(model, test_image, test_mask, tolerance, device="cuda"):
    x = test_image.to(device)
    y = test_mask.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    boxes_mask = get_largest_segments_bounding_box(test_mask)[0]
    boxes_pred = get_largest_segments_bounding_box(preds.cpu())[0]
    ################################################
    # Coordinates for the predicted box and ground truth box
    x1_pred, y1_pred, x2_pred, y2_pred = boxes_pred
    x1_gt, y1_gt, x2_gt, y2_gt = boxes_mask

    # Add tolerance
    x1_pred, y1_pred, x2_pred, y2_pred = x1_pred - tolerance, y1_pred - tolerance, x2_pred + tolerance, y2_pred + tolerance

    # Calculate area of predicted box and ground truth box
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # Calculate intersection coordinates
    x1_int = max(x1_pred, x1_gt)
    y1_int = max(y1_pred, y1_gt)
    x2_int = min(x2_pred, x2_gt)
    y2_int = min(y2_pred, y2_gt)
    
    # Check if there is an intersection
    if x1_int < x2_int and y1_int < y2_int:
        # Calculate the area of intersection
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
    else:
        # No intersection
        intersection_area = 0
    
    # Overreach calculation
    overreach = (pred_area - intersection_area) / gt_area if gt_area > 0 else 0
    
    # Underreach calculation
    underreach = (gt_area - intersection_area) / gt_area if gt_area > 0 else 0
    ################################################

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    rect_pred = patches.Rectangle((x1_pred, y1_pred), x2_pred - x1_pred,
                                  y2_pred - y1_pred, linewidth=4, edgecolor='orange', facecolor='none', label='predicted crop with tolerance')
    rect_mask = patches.Rectangle((boxes_mask[0], boxes_mask[1]), boxes_mask[2] - boxes_mask[0],
                                  boxes_mask[3] - boxes_mask[1], linewidth=4, edgecolor='dodgerblue', facecolor='none', label='original crop')

    axs[0].imshow(test_image[0, 0], cmap='gray')
    axs[0].set_title('X-ray image', fontsize=20)

    axs[1].imshow(test_mask[0, 0], cmap='gray')
    axs[1].set_title('Original mask', fontsize=20)

    axs[2].imshow(preds[0, 0].cpu(), cmap='gray')
    axs[2].set_title('Predicted mask', fontsize=20)

    for i in [0, 1, 2]:
        pred_rect_copy = copy(rect_pred)
        mask_rect_copy = copy(rect_mask)
        axs[i].add_patch(mask_rect_copy)
        axs[i].add_patch(pred_rect_copy)
    try:
        fig.suptitle(f"Overreach: {overreach:.3f}, Underreach: {underreach:.3f}", fontsize=30)
    except:
        pass
    plt.legend()
    plt.show()
    #plt.close(fig)
    return fig


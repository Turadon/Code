import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy
from torchvision.ops import (
    masks_to_boxes,
    generalized_box_iou
)


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
    box_1 = get_bounding_box(pred)
    box_2 = get_bounding_box(mask)
    box_iou = generalized_box_iou(box_1, box_2)
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    return box_iou


def get_largest_segments_bounding_box(input):
    '''
    Input: is a torch tensor with shape [B, 1, N, M]
    Output: Bounding box coordinates shape [B, 4] with coordinates (x1, y1, height, width)
    '''
    contour_segments = torch.empty(input.shape)
    for i in range(input.shape[0]):
        seg_mask = np.uint8(input*255)[i, 0]
        _, mask = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)

        # Finden aller zusammenhängenden Komponenten
        num_labels, labels_im = cv2.connectedComponents(mask)

        # Berechnen der Flächen aller Komponenten
        areas = []
        # Starten bei 1, um den Hintergrund auszuschließen
        for label in range(0, num_labels):
            area = np.sum(labels_im == label)
            areas.append(area)

        # Finden der zwei größten Komponenten
        if len(areas) < 2:
            contour_segments[i] = torch.zeros(seg_mask.shape)
        elif len(areas) < 3:
            biggest_labels = sorted(
                range(len(areas)), key=lambda sub: areas[sub])[-3:-1]
            index_mask = (labels_im == biggest_labels[0])
            contour_image = index_mask.astype(int)
            contour_image = torch.Tensor(contour_image).unsqueeze(0)
            contour_segments[i] = contour_image
        else:
            biggest_labels = sorted(
                range(len(areas)), key=lambda sub: areas[sub])[-3:-1]
            index_mask = np.logical_or(
                labels_im == biggest_labels[0], labels_im == biggest_labels[1])
            contour_image = index_mask.astype(int)
            contour_image = torch.Tensor(contour_image).unsqueeze(0)
            contour_segments[i] = contour_image
    boxes = get_bounding_box(contour_segments)
    return boxes

def metric_largest_box_iou(pred, mask):
    '''
    input: pred and mask of model: Tensor[N, 1, X, Y]
    output: sum of box_iou Tensor
    '''
    box_1 = get_largest_segments_bounding_box(pred)
    box_2 = get_largest_segments_bounding_box(mask)
    box_iou = generalized_box_iou(box_1, box_2)
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    return box_iou

def plot_image_mask_lagest_segments(model, test_image, test_mask, device="cuda"):
    x = test_image.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    boxes_mask = get_largest_segments_bounding_box(test_mask)[0]
    boxes_pred = get_largest_segments_bounding_box(preds)[0]

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    rect_pred = patches.Rectangle((boxes_pred[0], boxes_pred[1]), boxes_pred[2] - boxes_pred[0],
                                  boxes_pred[3] - boxes_pred[1], linewidth=4, edgecolor='orange', facecolor='none', label='predicted crop')
    rect_mask = patches.Rectangle((boxes_mask[0], boxes_mask[1]), boxes_mask[2] - boxes_mask[0],
                                  boxes_mask[3] - boxes_mask[1], linewidth=4, edgecolor='green', facecolor='none', label='original crop')

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
    plt.legend()
    plt.close(fig)
    return fig

import torch
import torch.nn as nn
import numpy as np
from train import val_data_loader, criterionGen

def compute_iou(true, pred):
    true_mask = np.asanyarray(true.cpu(), dtype = np.bool)
    pred_mask = np.asanyarray(pred, dtype = np.bool)
    union = np.sum(np.logical_or(true_mask, pred_mask))
    intersection = np.sum(np.logical_and(true_mask, pred_mask))
#    print(union, intersection)
    iou = intersection/union
    return iou

def eval_net(val_data_loader, net_g, device, ):
    ls = 0
    iou = 0
    criterion = nn.BCELoss()
    for batch in val_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)
        prediction = net_g(input)
        ls += criterion(prediction.view(-1), target.view(-1))
        iou += compute_iou(target, prediction)
    return ls/len(val_data_loader), iou/len(val_data_loader)
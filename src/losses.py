#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:33:43 2020

@author: cathy
"""

import torch
import os.path
import torch.nn as nn
import torch.nn.functional as F



def dice_loss_single_index(true, pred, index):
    epsilon = 1
    true = true[:,index]
    pred = pred[:,index] 
    dice_coeff = (2.0 * torch.sum(pred*true) + epsilon) \
        / (torch.sum(true) + torch.sum(pred) + epsilon)
    return (1.0 - dice_coeff)

class LossFunc(nn.Module):
    def __init__(self, weight = None):
        super(LossFunc, self).__init__()
        self.weight = weight#.cuda()        
        self.softmax = nn.Softmax(dim=1)
        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.weightCrossEntropy = nn.CrossEntropyLoss(weight = self.weight)


    def forward(self, pred, label):
        _, label_transform = label.max(1)
        label = label.transpose(1,4).transpose(1,3).transpose(1,2).view(-1,3)
        label_transform = label_transform.unsqueeze(dim = 1)
        pred = pred.transpose(1,4).transpose(1,3).transpose(1,2).view(-1,3)
        label_transform = label_transform.transpose(1,4).transpose(1,3).transpose(1,2).view(-1,1).squeeze()
        pred = self.softmax(pred)
        dice_helix = dice_loss_single_index(label,pred, 1)
        dice_sheet = dice_loss_single_index(label,pred, 2)

        cross_entropy = self.crossEntropyLoss(pred, label_transform)
        weighted_cross_entropy = self.weightCrossEntropy(pred, label_transform)
        total_loss = 0.7 * cross_entropy + 0.3 * weighted_cross_entropy + 0.3 * dice_helix + 0.3 * dice_sheet
        return total_loss
    


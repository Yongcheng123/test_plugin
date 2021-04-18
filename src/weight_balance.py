#import numpy as np
import torch

def getClassDist(labels):

    labels = labels.transpose(1,4).transpose(1,3).transpose(1,2).view(-1,3)
    _, indices = torch.max(labels, 1)
    n_of_element = labels.size(0)
    background  = 0
    helix = 0
    sheet = 0

    for i in range(n_of_element):
        if indices[i] == 0:
            background += 1
        if indices[i] == 1:
            helix += 1
        if indices[i] == 2:
            sheet += 1
    return background, helix, sheet, n_of_element

def inverseOfFrequency(labels, class_dist=None, minimum_frequency=0.0):
    if class_dist == None:
        class_dist = getClassDist(labels)
    backg, helix, sheet, sum_voxels = class_dist
    
    backg_freq = backg / sum_voxels
    helix_freq = helix / sum_voxels
    sheet_freq = sheet / sum_voxels 

    backg_weight = 1.0
    helix_weight = 1.0
    sheet_weight = 1.0

    if backg_freq > minimum_frequency:
        backg_weight = 1.0 / backg_freq

    if helix_freq > minimum_frequency:
        helix_weight = 1.0 / helix_freq

    if sheet_freq > minimum_frequency:
        sheet_weight = 1.0 / sheet_freq

    return torch.tensor([backg_weight, helix_weight, sheet_weight])












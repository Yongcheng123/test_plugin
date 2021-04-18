#import numpy as np
import os
#import sys
from functools import cmp_to_key
from pathlib import Path
import torch
import sys
sys.dont_write_bytecode = True
def removeOutlierChains(batches):
    minSize = 16
    maxSize = 100
    culledBatches = []
    for batch in batches:
        _, _, orig, _= batch
        xLength, yLength, zLength, xMin, yMin, zMin = orig
        if xLength < maxSize+1 and xLength > minSize-1:
            if yLength < maxSize+1 and yLength > minSize-1:
                if zLength < maxSize+1 and zLength > minSize-1:
                    culledBatches.append(batch)
    return culledBatches

def printChainStats(batches, outFileName):
    batches_copy = batches.copy()
    batches_copy.sort(key=cmp_to_key(compareLength))
    with open(outFileName, 'w') as out:
        for batch in batches_copy:
            chain, batch, orig, _, _, is_rotated = batch
            if not is_rotated:
                mrc,pdb,chain = chain
                xLength, yLength, zLength, xMin, yMin, zMin = orig
                line = mrc + " " + pdb + " " + chain + "\t" + str(xLength) + " " \
                    + str(yLength) + " " + str(zLength) + '\n'
                out.write(line)


def compareLength(chain1, chain2):
    coords1 = chain1[2]
    coords2 = chain2[2]
    key1 = coords1[0] * coords1[1] * coords1[2]
    key2 = coords2[0] * coords2[1] * coords2[2]
    if key1 == key2:
        return 0
    if key1 > key2:
        return 1
    if key1 < key2:
        return -1


def readFile(xLength, yLength, zLength, path, xMin, yMin, zMin):
    densities = []
    with open(path) as inf:

        density = torch.zeros([1, 1, xLength, yLength, zLength])
        one_hot_labels = torch.zeros([1, 3, xLength, yLength, zLength])
        length = xLength * yLength * zLength
        densities = torch.zeros([length])
        ind = 0
        for line in inf:
            xCoord, yCoord, zCoord, thresh, label = line.strip().split(",")
            xCoord = int(xCoord)
            yCoord = int(yCoord)
            zCoord = int(zCoord)
            thresh = float(thresh)
            label = int(label)
            try:
                density[0][0][xCoord][yCoord][zCoord]  = thresh
                one_hot_labels[0][label][xCoord][yCoord][zCoord] = 1
            except IndexError as err:
                continue
            else:                
                densities[ind] = thresh
                ind += 1

    for x in range(0, xLength):
        for y in range(0, yLength):
            for z in range(0, zLength):
                if one_hot_labels[0][0][x][y][z] == 0 and \
                        one_hot_labels[0][1][x][y][z] == 0 and \
                        one_hot_labels[0][2][x][y][z] == 0:
                    one_hot_labels[0][0][x][y][z] = 1

    return density, one_hot_labels, densities

def getMRCDimensions(path):
    xMin = 100000000
    xMax = -1
    yMin = 100000000
    yMax = -1
    zMin = 100000000
    zMax = -1
    with open(path) as inf3:
        for line4 in inf3:
            xCoord, yCoord, zCoord, thresh, label = line4.strip().split(",")
            if float(thresh) > 0:
                xCoord = int(xCoord)
                yCoord = int(yCoord)
                zCoord = int(zCoord)
                if xCoord < xMin:
                    xMin = xCoord
                if yCoord < yMin:
                    yMin = yCoord
                if zCoord < zMin:
                    zMin = zCoord
                if xCoord > xMax:
                    xMax = xCoord
                if yCoord > yMax:
                    yMax = yCoord
                if zCoord > zMax:
                    zMax = zCoord
    return xMax, xMin, yMax, yMin, zMax, zMin

def loadBatches(data_list_path, pickle_path, data_path, layers):
    padding = pow(2, layers-1)
    from weight_balance import getClassDist, inverseOfFrequency
    pickle_path = Path(pickle_path)
    batches = []
    if pickle_path.is_file():
        batches = torch.load(pickle_path)
        return batches
    mrcList = []
    proteinList = []
    chainList = []
    densities = []
    with open(data_list_path) as dataList:
        #extract protein name
        for newLine in dataList:
            mrc, pdb, chain = newLine.strip().split()
            mrcList.append(mrc)
            proteinList.append(pdb)
            chainList.append(chain)
    for pair, chain in zip(zip(proteinList, mrcList), chainList):
        protein, mrc = pair
        chainsPath =  data_path + mrc + "_" + protein + "/" + chain  + "_label_stride.txt"
        if os.path.isfile(chainsPath):
            path = chainsPath
            xMax, xMin, yMax, yMin, zMax, zMin = getMRCDimensions(path)
            xLength = xMax
            yLength = yMax
            zLength = zMax
            orig_coords = torch.tensor([xLength, yLength, zLength, xMin, yMin, zMin])
            xLength += padding-xLength%padding
            yLength += padding-yLength%padding
            zLength += padding-zLength%padding
            density, one_hot_labels, densities = readFile(
                xLength, yLength, zLength, path, xMin, yMin, zMin) 
            sys.stdout.flush()            
            class_dist = getClassDist(one_hot_labels)
            background, helix, sheet, n_of_element = class_dist
            print(path, n_of_element)
            sys.stdout.flush()
            if n_of_element == 0:
                continue
            backg_freq = background / n_of_element
            helix_freq = helix / n_of_element
            sheet_freq = sheet / n_of_element
            if helix_freq < 0.0001:
                if sheet_freq < 0.0001:
                    continue
            weights = inverseOfFrequency(
                one_hot_labels, class_dist=class_dist, minimum_frequency=0.0001)
            """class_dist is not tensor"""
            next_batch = normalizeBatch(((mrc, protein, chain),
                            (density, one_hot_labels, weights, class_dist),
                            orig_coords,
                            torch.tensor([xLength, yLength, zLength, xMin, yMin, zMin])),
                             densities)
            sys.stdout.flush()
            batches.append(next_batch)
        else:
            print("ERROR:", chainsPath, "does not exist!")
    batches = removeOutlierChains(batches)
    return batches


def normalizeBatch(batch, densities):

    
    max_density = max(densities)

    densities = torch.tensor([density/max_density for density in densities])

    average_density = densities.mean()

    epsilon = 0.00000007
    std_dev_density = densities.std() + epsilon

    sys.stdout.flush()
    norm_batch = []

    if True:
        chain, batch, orig, coord = batch
        density, one_hot_labels, weights, dist = batch
        xLength, yLength, zLength, _, _, _ = coord
        for x in range(0, xLength):
            for y in range(0, yLength):
                for z in range(0, zLength):
                        d = density[0][0][x][y][z]
                        density[0][0][x][y][z] = (
                            ((d/max_density) - average_density)
                              / std_dev_density)
        norm_batch.append(
            (chain,
            (density,one_hot_labels,weights,dist),
            orig,
            coord))
    return norm_batch[0]


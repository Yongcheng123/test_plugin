import numpy as np
import subprocess
import math
import os
import sys
from pathlib import Path
import logging


def logisticRegression(probs):
    return np.exp(probs) / np.sum(np.exp(probs), axis=0)


def writePrediction(output_path, chain_info, predictedLabels, coordinate_info):
    logging.info("writePridiction start...")
    mrc, pdb, chain = chain_info
    predictedLabels = predictedLabels.cpu().numpy()
    xLength, yLength, zLength, xMin, yMin, zMin = coordinate_info
    numLines = xLength * yLength * zLength
    axisZ = np.zeros((1, numLines), dtype=float)
    axisY = np.zeros((1, numLines), dtype=float)
    axisX = np.zeros((1, numLines), dtype=float)
    for x in range(0, xLength):
        for y in range(0, yLength):
            for z in range(0, zLength):
                curPos = (x) * yLength * zLength + (y) * zLength + (z)
                axisX[0][curPos] = x
                axisY[0][curPos] = y
                axisZ[0][curPos] = z
    path = output_path + mrc + "_" + pdb + "_" + chain + ".txt"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f1 = open(path, "w+")
    for i in range(0, numLines):
        f1.write(
            str(int(axisX[0][i]))
            + " "
            + str(int(axisY[0][i]))
            + " "
            + str(int(axisZ[0][i]))
            + " "
            + str(predictedLabels[i])
            + "\r\n"
        )


def writeVisual(output_path, chain_info, prediction_path, maps_path):
    logging.info("writeVisual start...")
    mrc, pdb, chain = chain_info

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    prediction_path = prediction_path + mrc + "_" + pdb + "_" + chain + ".txt"
    args = (
        mrc
        + " "
        + pdb
        + " "
        + chain
        + " "
        + prediction_path
        + " "
        + output_path
        + " "
        + maps_path
    )
    # args = (
    #     mrc
    #     + " "
    #     + pdb
    #     + " "
    #     + chain
    #     + " "
    #     + prediction_path
    #     + " "
    #     + output_path
    #     + " "
    #     + maps_path
    # )

    subprocess.call("./visualizations/bin/visualization" + " " + args, shell=True)
    try:
        res = subprocess.check_call("./visualizations/bin/visualization" + " " + args, shell=True)
        print("res:", res)
    except subprocess.CalledProcessError as exc:
        print("returncode:", exc.returncode)
        print("cmd:", exc.cmd)
        print("output:", exc.output)

    logging.debug("writeVisual finish...")


def writePredictionsAndVisuals(
    pred_output_path, vis_output_path, prediction, maps_path
):
    logging.info("writePridictionAndVisuals start...")
    chain_info, predicted_labels, coordinate_info = prediction
    writePrediction(pred_output_path, chain_info, predicted_labels, coordinate_info)
    writeVisual(vis_output_path, chain_info, pred_output_path, maps_path)


def writeTrueLabels(output_path, label, maps_path, label_path):
    logging.info("writeTruelabels start...")
    chain_info, _, _ = label
    mrc, pdb, chain = chain_info
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    label_path = label_path + mrc + "_" + pdb + "/" + chain + "_label_stride.txt"
    args = (
        mrc
        + " "
        + pdb
        + " "
        + chain
        + " "
        + label_path
        + " "
        + output_path
        + " "
        + maps_path
        + " "
        + "True"
    )

    subprocess.call(
        os.path.dirname(os.path.abspath(__file__))
        + "./visualizations/bin/visualization"
        + args,
        shell=True,
    )

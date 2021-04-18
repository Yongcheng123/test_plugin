import importlib
import argparse
import random

# from f1_score import f1Scores
import numpy as np
import torch
import time
import sys
import os

# from metrics.metrics import confusion_matrix
from . import predictions
from .losses import LossFunc
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import logging

# =====================================================================================================#
#                                       Continuum Iterator                                            #
# =====================================================================================================#


def load_datasets(args):
    logging.info("Load data path:", sys.path[0])
    test_data = torch.load("./data/testingList.pt")
    return test_data

# def test_model(model, test_data, args):

def test_model(model, test_data, args):

    logging.info("Prediction start...")
    print("Prediction start...")
    sys.stdout.flush
    Model = importlib.import_module(".model.unet", package="chimerax.dl_struct")
    model = Model.Gem_UNet(args)
    if args.cuda:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    logging.info("Checkpoint loading...")
    checkpoint = torch.load("model.chkpt", map_location=map_location)

    model.load_state_dict(checkpoint["model"])

    result = []
    if args.cuda:
        model.to("cuda")
    model.eval()
    with torch.no_grad():
        for i, t_data in enumerate(test_data):
            logging.debug("Run test data %s", i)

            chains, batch, orig, coords = t_data
            xLength, yLength, zLength, _, _, _ = orig
            x_image, labels, pos_weight, class_dist = batch
            mrc, protein, chain = chains
            if args.cuda:
                x_image = x_image.cuda()
                labels = labels.cuda()
            out = model(x_image)
            pred = out[:, :, :xLength, :yLength, :zLength]

            pred = pred.transpose(1, 4).transpose(1, 3).transpose(1, 2).reshape(-1, 3)
        
            _, pred = pred.max(1)

            logging.debug("testing iteration: %s  -- mrc: %s  protein:%s  chain: %s", i, mrc, protein, chain)
            print("testing iteration", i, "\tchain:", mrc, protein, chain)
            sys.stdout.flush()
            predictions.writePredictionsAndVisuals(
                args.test_pred_path,
                args.test_vis_path,
                (chains, pred, orig),
                args.map_path,
            )

            result.append(out)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_memories", type=int, default=0, help="number of memories per task"
    )
    parser.add_argument(
        "--memory_strength",
        default=0.5,
        type=float,
        help="memory strength (meaning depends on memory)",
    )
    # experiment parameters
    parser.add_argument("--cuda", type=str, default="yes", help="Use GPU?")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--test_pred_path", type=str, default="./output/testing_predictions/labels/"
    )
    parser.add_argument(
        "--test_vis_path", type=str, default="./output/testing_predictions/visuals/"
    )
    parser.add_argument("--map_path", type=str, default="./data/EMDBdata_6/")
    parser.add_argument("--n_tasks", type=int, default=3)
    parser.add_argument('--write_pred', default=True)
    
    args = parser.parse_args()

    args.cuda = True if args.cuda == "yes" else False
    if args.cuda == True and not torch.cuda.is_available():
        print("CUDA not available, disabling...")
        args.cuda = False

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    print("loading data...")
    sys.stdout.flush()
    test_data = load_datasets(args)

    Model = importlib.import_module(".model.unet")
    model = Model.Gem_UNet(args)
    
    test_model(model, test_data, args)


if __name__ == "__main__":
    main()

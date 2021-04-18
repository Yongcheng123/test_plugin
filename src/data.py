#from torchvision import transforms
#from PIL import Image
import argparse
#import os.path
#import random
import torch
from batches import *
import sys
from pathlib import Path
sys.dont_write_bytecode = True


parser = argparse.ArgumentParser()

parser.add_argument('--i', default='data/', help='input directory')
parser.add_argument('--o', default='testing_batches.pt', help='output file')


args = parser.parse_args()

DATA_PATH = "./data/"
CHAINS_PATH = DATA_PATH + "EMDBdata_6/"
INPUT_PATH = DATA_PATH + args.i
OUTPUT_PATH = DATA_PATH + args.o

print("load data", OUTPUT_PATH)
sys.stdout.flush()

if Path(OUTPUT_PATH).is_file():
    pass
else:
    batches = loadBatches(INPUT_PATH, OUTPUT_PATH, CHAINS_PATH, layers = 3)
    torch.save(batches, OUTPUT_PATH)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler

from meta import *

EPOCHS = 1000
BATCH_SIZE = 64
DATASET_WHITE_SIZE = 1000000
DATASET_BLACK_SIZE = 1000000
DATASET_SIZE = DATASET_WHITE_SIZE + DATASET_BLACK_SIZE

BATCHES_PER_EPOCH = DATASET_SIZE / BATCH_SIZE

INPUT_DIM = 773
DBN_LAYERS_DIM = [600, 400, 200, 100]
DBN_NUM_LAYERS = len(DBN_LAYERS_DIM)

INIT_LEARNING_RATE = 1e-2
DECAY_FACTOR = 0.99


def StepDecay(epoch):

    print(epoch)
    return float(INIT_LEARNING_RATE * (DECAY_FACTOR ** epoch-1))


lr_sched = LearningRateScheduler(StepDecay)


def main():

    pass

if __name__ == "__main__":
    
    main()
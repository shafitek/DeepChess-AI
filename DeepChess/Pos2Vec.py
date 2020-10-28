import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import math
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD, Adam, schedules
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras import backend as K

sys.path.append("..")
from meta import *

EPOCHS = 50
BATCH_SIZE = 64
DATASET_WHITE_SIZE = 1000000
DATASET_BLACK_SIZE = 1000000
DATASET_SIZE = DATASET_WHITE_SIZE + DATASET_BLACK_SIZE

BATCHES_PER_EPOCH = DATASET_SIZE / BATCH_SIZE

INPUT_DIM = 773
DBN_LAYERS_DIM = [773, 600, 400, 200, 100]
DBN_NUM_LAYERS = len(DBN_LAYERS_DIM)

INIT_LEARNING_RATE = 5e-3
DECAY_FACTOR = 0.98

POS2VEC_WEIGHTS_PATH = os.path.join(MODELS_DIR, "DBN.h5")

FORWARD_PASS_SPLIT = 16000
FORWARD_PASS_IT = int(DATASET_SIZE / FORWARD_PASS_SPLIT)

DBN_WEIGHTS = []


def getNewTrainX(model, trainX, input_shape):

    trainY = np.empty((0, input_shape))
    f_pass_arr = []
    print("Calculating new dataset for", input_shape, "...")

    model.pop()
    for i in range(FORWARD_PASS_IT):
        start = i * FORWARD_PASS_SPLIT
        end = (i+1) * FORWARD_PASS_SPLIT
        f_pass_out = model.predict(trainX[start:end])
        f_pass_arr.append(f_pass_out)

    for stack in f_pass_arr:
        trainY = np.vstack((trainY, stack))

    print("Calculated.", trainY.shape)
    return trainY


class LearningRateDecay(Callback):

    def __init__(self, init_learning_rate, decay_rate):
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        decayed_learning_rate = self.init_learning_rate * \
            math.pow(self.decay_rate, epoch)
        K.set_value(self.model.optimizer.lr, decayed_learning_rate)
        new_lr = float(K.get_value(self.model.optimizer.lr))
        print("Learning Rate at {}: {:.5f}".format(epoch+1, new_lr))


def trainDBN(trainX):
    """ Unsupervised layerwise training """
    for i in range(1, DBN_NUM_LAYERS):
        model = Sequential()
        model.add(Dense(DBN_LAYERS_DIM[i], input_dim=DBN_LAYERS_DIM[i-1], activation='relu'))
        model.add(Dense(DBN_LAYERS_DIM[i-1], activation='sigmoid'))
        
        model.compile(loss=MeanSquaredError(),
                      optimizer=Adam(learning_rate=INIT_LEARNING_RATE),
                      metrics=['accuracy'])
        model.summary()

        model.fit(trainX, trainX,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[LearningRateDecay(INIT_LEARNING_RATE, DECAY_FACTOR)])

        DBN_WEIGHTS.append(model.layers[0].get_weights())

        if i is not DBN_NUM_LAYERS-1:
            trainX = getNewTrainX(model, trainX, DBN_LAYERS_DIM[i])

    model = Sequential()
    model.add(Dense(DBN_LAYERS_DIM[1], input_dim=DBN_LAYERS_DIM[0], activation='relu'))
    for i in range(2, DBN_NUM_LAYERS):
        model.add(Dense(DBN_LAYERS_DIM[i], activation='relu'))

    for i, layer in enumerate(model.layers):
        layer.set_weights(DBN_WEIGHTS[i])

    print("\n-----------")
    print("FINAL MODEL")
    print("-----------\n")

    model.summary()
    print("---\n")

    model.save_weights(POS2VEC_WEIGHTS_PATH)

    with open(META_FILE, 'r') as f:
        meta = json.load(f)
    
    meta['pos2vec_trained'] = "True"
    meta['pos2vec_weights_path'] = POS2VEC_WEIGHTS_PATH

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=4)

    print("Pos2Vec training complete.")
    print("     Path:", POS2VEC_WEIGHTS_PATH)
    print("---")


def main():
    
    print("-------------------------")
    print("| DeepChess: 1. Pos2Vec |")
    print("-------------------------\n")
    print("Base Directory:", BASE_DIR)
    print("Dataset Directory:", DATASET_DIR)
    print("meta.json Directory:", META_FILE)
    print("---\n")

    if not os.path.exists(META_FILE):
        print("No meta.json found. Run setup.py first.")
        print("---")
        return 1

    with open(META_FILE) as f:
        meta = json.load(f)

    if meta['pos2vec_trained'] == "True" and os.path.exists(meta['pos2vec_weights_path']):
        print("Pos2Vec already trained.")
        print("     Path:", meta['pos2vec_weights_path'])
        print("---")
        return 1

    with open(meta['dataset_white_path'], 'rb') as f:
        WHITE_DATASET = np.load(f)

    with open(meta['dataset_black_path'], 'rb') as f:
        BLACK_DATASET = np.load(f)

    print("Preparing the dataset...")

    w_idxs = np.random.choice(
        WHITE_DATASET.shape[0], size=DATASET_WHITE_SIZE, replace=False)
    b_idxs = np.random.choice(
        BLACK_DATASET.shape[0], size=DATASET_BLACK_SIZE, replace=False)

    DATASET = np.vstack((WHITE_DATASET[w_idxs, :], BLACK_DATASET[b_idxs, :]))
    np.random.shuffle(DATASET)

    print("Dataset prepared.")
    print("     SIZE:", DATASET.shape)

    trainDBN(DATASET)

if __name__ == "__main__":

    main()

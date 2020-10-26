import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler

from meta import *

EPOCHS = 200
BATCH_SIZE = 64
DATASET_WHITE_SIZE = 1000000
DATASET_BLACK_SIZE = 1000000
DATASET_SIZE = DATASET_WHITE_SIZE + DATASET_BLACK_SIZE

BATCHES_PER_EPOCH = DATASET_SIZE / BATCH_SIZE

INPUT_DIM = 773
DBN_LAYERS_DIM = [600, 400, 200, 100]
DBN_NUM_LAYERS = len(DBN_LAYERS_DIM)

INIT_LEARNING_RATE = 5e-3
DECAY_FACTOR = 0.98

POS2VEC_WEIGHTS_PATH = os.path.join(MODELS_DIR, "DBN.h5")

def StepDecay(epoch):
    
    print(epoch)
    return float(INIT_LEARNING_RATE * (DECAY_FACTOR ** epoch-1))

lr_sched = LearningRateScheduler(StepDecay)

def baseAutoencoder(trainX, testX):
    
    model = Sequential()
    model.add(Dense(DBN_LAYERS_DIM[0], input_dim=INPUT_DIM, activation='relu'))
    model.add(Dense(INPUT_DIM, activation='linear'))

    loss_fn = losses.CategoricalCrossentropy
    model.compile(loss=loss_fn, optimizer=SGD())
    model.summary()

    # model.fit(trainX, testX, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_sched,])

    # train_eval = model.evaluate(trainX, trainX)
    # test_eval = model.evaluate(testX, testX)

    # print('Base Encoder Metric: train=%.3f, test=%.3f' % (train_eval, test_eval))

    return model

def trainDBN(model, trainX, testX):
    """ Unsupervised layerwise training """
    for i in range(1, DBN_NUM_LAYERS):
        model.pop()
        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(DBN_LAYERS_DIM[i], activation='relu'))
        model.add(Dense(DBN_LAYERS_DIM[i-1], activation='linear'))
        loss_fn = losses.CategoricalCrossentropy
        model.compile(loss=loss_fn, optimizer=SGD())
        model.summary()
        # model.fit(trainX, testX, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_sched,])

        # train_eval = model.evaluate(trainX, trainX)
        # test_eval = model.evaluate(testX, testX)

        # print('Base Encoder Metric: train=%.3f, test=%.3f' % (train_eval, test_eval))
    
    model.pop()
    for layer in model.layers:
        layer.trainable = True
        print(layer.get_weights())
    
    loss_fn = losses.CategoricalCrossentropy
    model.compile(loss=loss_fn, optimizer=SGD())
    model.summary()
    print("---\n")

    model.save_weights(POS2VEC_WEIGHTS_PATH)

    with open(META_FILE, 'r') as f:
        meta = json.load(f)
    
    meta['pos2vec_weights_path'] = POS2VEC_WEIGHTS_PATH

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=4)

    print("Pos2Vec training complete.")
    print("     Path:", POS2VEC_WEIGHTS_PATH)
    print("---")

def main():
    
    if not os.path.exists(META_FILE):
        print("No meta.json found. Run setup.py first.")
        print("---")
        return 1
    
    model = baseAutoencoder([],[])
    trainDBN(model, 0, 0)

if __name__ == "__main__":

    main()

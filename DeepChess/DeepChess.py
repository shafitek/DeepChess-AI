import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import argparse
import random
import math
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU
from tensorflow.keras import losses
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K

sys.path.append("..")
from meta import *

EPOCHS = 100
BATCH_SIZE = 64
INPUT_DIM = 773

TRAIN_SIZE = 1000000
VAL_SIZE = 100000

INIT_LEARNING_RATE = 1e-2
DECAY_FACTOR = 0.99

DEEPCHESS_MODEL_PATH = os.path.join(MODELS_DIR, "DEEPCHESS.h5")


class DataGenerator(Sequence):

    def __init__(self, wx, bx, dataset_size, batch_size=BATCH_SIZE):

        self.wx, self.bx = wx, bx
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        print(wx.shape, bx.shape)
        sm_ds_size = min(wx.shape[0], bx.shape[0])
        if dataset_size > sm_ds_size:
            self.dataset_size = sm_ds_size

    def __len__(self):

        return self.dataset_size//self.batch_size + self.dataset_size % 2

    def __getitem__(self, idx):
        """ Return (W,L) or (L,W) pair batch to the network """
        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        
        range_it = min(self.batch_size, self.dataset_size-(idx*self.batch_size))
        for i in range(range_it):
            if random.randint(0,1):
                X1.append(self.wx[idx*self.batch_size + i])
                X2.append(self.bx[idx*self.batch_size + i])
                Y1.append(1)
                Y2.append(0)
            else:
                X2.append(self.wx[idx*self.batch_size + i])
                X1.append(self.bx[idx*self.batch_size + i])
                Y2.append(1)
                Y1.append(0)

        X1 = np.array(X1)
        X2 = np.array(X2)

        Y12 = np.stack([Y1, Y2], axis=1)
        return [X1, X2], Y12

    def on_epoch_end(self):
        
        np.random.shuffle(self.wx)
        np.random.shuffle(self.bx)


class LearningRateDecay(Callback):

    def __init__(self, init_learning_rate, decay_rate):
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        global EPOCHS_ALREADY_TRAINED
        decayed_learning_rate = self.init_learning_rate * math.pow(self.decay_rate, epoch+EPOCHS_ALREADY_TRAINED)
        K.set_value(self.model.optimizer.lr, decayed_learning_rate)
        new_lr = float(K.get_value(self.model.optimizer.lr))
        print("Learning Rate at {}: {:.5f}".format(EPOCHS_ALREADY_TRAINED+epoch+1, new_lr))


def trainDeepChess(dbn, trainWX, trainBX, testWX, testBX):

    left_input = Input(shape=(INPUT_DIM,))
    right_input = Input(shape=(INPUT_DIM,))

    # Since l_p2v and r_p2v use the same model instance, they share weights
    l_p2v = dbn(left_input)
    r_p2v = dbn(right_input)

    head = Concatenate()([l_p2v, r_p2v])
    h1 = Dense(400, activation=LeakyReLU(alpha=0.3))(head)
    h2 = Dense(200, activation=LeakyReLU(alpha=0.3))(h1)
    h3 = Dense(100, activation=LeakyReLU(alpha=0.3))(h2)
    prediction = Dense(2, activation='softmax')(h3)

    # Siamese Network
    model = Model(inputs=[left_input, right_input], outputs=prediction)
    
    model.compile(loss=losses.CategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=INIT_LEARNING_RATE),
                  metrics=['accuracy'])

    training_generator = DataGenerator(trainWX, trainBX, TRAIN_SIZE)
    validation_generator = DataGenerator(testWX, testBX, VAL_SIZE, batch_size=32)

    filepath = os.path.join(
        MODELS_CHECKPOINT_DIR, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_accuracy',
                                 mode='max',)

    global CONTINUE_TRAINING
    if CONTINUE_TRAINING:
        with open(META_FILE, 'rb') as f:
            meta = json.load(f)
        model.load_weights(meta['deepchess_model_path'])
        
    model.summary()

    model.fit(
        epochs=EPOCHS,
        x=training_generator,
        validation_data=validation_generator,
        callbacks=[checkpoint, LearningRateDecay(INIT_LEARNING_RATE, DECAY_FACTOR)],
        use_multiprocessing=True,
        workers=8
    )

    print("---")
    print("\nSaving model... ", end='')
    model.save(DEEPCHESS_MODEL_PATH)
    print("SAVED.")

    with open(META_FILE, 'r') as f:
        meta = json.load(f)

    meta['deepchess_trained'] = "True"
    meta['deepchess_model_path'] = DEEPCHESS_MODEL_PATH
    meta['deepchess_epochs_trained'] += EPOCHS

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=4)

    print("DeepChess training complete.")
    print("     Path:", DEEPCHESS_MODEL_PATH)
    print("---\n")


def Pos2Vec(weights_path):
    
    model = Sequential()
    model.add(Dense(600, input_dim=INPUT_DIM, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(400, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(200, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(100, activation=LeakyReLU(alpha=0.3)))
    
    model.load_weights(weights_path)

    print("\n-------------")
    print("POS2VEC MODEL")
    print("-------------\n")

    model.summary()

    print("---\n")

    return model


def main():

    print("---------------------------")
    print("| DeepChess: 2. DeepChess |")
    print("---------------------------\n")
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

    global CONTINUE_TRAINING
    if meta['deepchess_trained'] == "True" and not CONTINUE_TRAINING:
        model = load_model(meta['deepchess_model_path'])
        model.summary()
        print('---\n')
        print("DeepChess model already trained.")
        print("     Path:", meta['deepchess_model_path'])
        print("---")
        return 1

    if CONTINUE_TRAINING:
        global EPOCHS_ALREADY_TRAINED
        EPOCHS_ALREADY_TRAINED = meta['deepchess_epochs_trained']
        print('---')
        print('Epochs trained already:', EPOCHS_ALREADY_TRAINED)
        print("---")

    with open(meta['dataset_white_path'], 'rb') as f:
        WHITE_DATASET_F = np.load(f)

    with open(meta['dataset_black_path'], 'rb') as f:
        BLACK_DATASET_F = np.load(f)

    print("Preparing the dataset...")

    w_ds_train_size = meta['num_white_positions_train'] - \
        meta['num_white_positions_test']
    b_ds_train_size = meta['num_black_positions_train'] - \
        meta['num_black_positions_test']

    np.random.shuffle(WHITE_DATASET_F)
    np.random.shuffle(BLACK_DATASET_F)

    W_DS_TRAIN = WHITE_DATASET_F[:w_ds_train_size]
    B_DS_TRAIN = BLACK_DATASET_F[:b_ds_train_size]

    W_DS_VAL = WHITE_DATASET_F[w_ds_train_size:]
    B_DS_VAL = BLACK_DATASET_F[b_ds_train_size:]

    print("\nDatasets prepared.")
    print("     WHITE_TRAIN Shape:", W_DS_TRAIN.shape)
    print("     BLACK_TRAIN Shape:", B_DS_TRAIN.shape)

    print("     WHITE_VAL Shape:", W_DS_VAL.shape)
    print("     BLACK_VAL Shape:", B_DS_VAL.shape)
    print("---\n")

    dbn = Pos2Vec(meta['pos2vec_weights_path'])

    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    
    print('\n---')
    print("Training started at {}.".format(start_time))
    print('---\n')

    trainDeepChess(dbn, W_DS_TRAIN, B_DS_TRAIN, W_DS_VAL, B_DS_VAL)

    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print('\n---')
    print("Training started at {}.".format(start_time))
    print("Training ended at {}.".format(end_time))
    print('---\n')

    print("ALL DONE!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--continue', action="store_true", dest='continue_train',
                        help='Resumes training the model if a model already exists.')

    args = parser.parse_args()
    global CONTINUE_TRAINING
    CONTINUE_TRAINING = args.continue_train
    main()

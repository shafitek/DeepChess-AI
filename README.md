# DeepChess-AI
## DETAILS

The AI - implemented using the minimax algorithm with alpha-beta pruning uses a neural network based score function.

The score function is based on the following academic paper:

[DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess](https://arxiv.org/abs/1711.09667)

### Differences from the paper

1. I used LeakyReLU instead of ReLU as mentioned in the paper. With ReLU, the model did not converge at all, probably due to the gradients vanishing. LeakyReLU fixed that problem.
2. My dataset consisted of 2,100,000 White won positions, 1,600,000 Black won positions, and 100,000 from each color was used for validation.
3. I only trained the DeepChess model 500 epochs, acheiving a validation of ~90%. I barely trained Pos2Vec - only for 20 epochs for each layer.

## OVERVIEW

### TO TRAIN:
1. Run `python3 setup.py`. This will create a `meta.json` file that will be used throughout in the training process. `meta.json` contains some meta data that will be used so the scripts know things like where the datasets are stored, how many epochs was trained, etc.
2. Then run `python3 scripts/GenerateDataset.py`. This will download, uncompress, and generate datasets for each color. After uncompressing, the dataset file offsets for white wins and black wins are calculated and stored in parallel. Then using the offsets, 8 processes - 4 for each color, reads 10 random moves from each game. Finally two final datasets are created, one for white win and the other for black win.
3. Run `python3 scripts/Pos2Vec.py` to train the Deep Belief Network. After training is complete, its weights are stored in `models/DBN.h5` file.
4. Run `python3 scripts/DeepChess.py` to train the DeepChess model. After training is complete, the whole model is stored in `models/DEEPCHESS.h5 file`.

### TO PLAY:

Open `PlayChess.ipynb` to use the model to play games. The game is played against Stockfish. The DeepChess engine can be found in `engine/deepchess.py`. 

#### Libraries and API Versions:
TensorFlow `2.3.0`\
Keras `2.4.3`

### Directory

```
.
├── dataset
|   - Directory where the datasets are stored.
|   - The processed dataset will also be stored here.
│   └── .gitkeep
├── scripts
│   ├── __init__.py
│   ├── GenerateDataset.py
│       - After running `setup.py`, run this. This will download, uncompress,
│         and generates the datasets. It uses about multiprocessing to generate
│         in 3+ million datasets in a short time but it is very memory heavy.
│   ├── Pos2Vec.py
│       - This is the first network that needs to be trained. This will generate
│         weights that will be used when training the main network.
│   └── DeepChess.py
│       - This is the main network. The last two scripts must be run before
│         running this. This will begin trianing the network. If you want to
│         train more epochs, simply append '--continie' when running the script.
├── engine
│   ├── __init__.py
│   ├── engines_info.txt
│   ├── deepchess.py
│       - DeepChess engine.
│   └── stockfish_x64
│       - Stockfish engine binary file.
├── .gitignore
├── meta.json
│   - This will be generated after you run setup.py.
├── meta.py
├── models
│   ├── checkpoints
│   │   ├── .gitkeep
│   ├── DBN.h5
│       - WEIGHTS ONLY of my trained DBN model. Trained for only a few epochs.
│   └── DEEPCHESS.h5
│       - FULL MODEL of my trained DeepChess model. Validation accuray ~90%.
│         Trained for 500 epochs.
├── PlayChess.ipynb
│   - DeepChess vs Stockfish. 
├── README.md
└── setup.py
    - File to run first. Creates a file called `meta.json`
```

## SOURCES

https://www.tensorflow.org/guide/keras/ \
https://keras.io/api/ \
https://blog.chesshouse.com/how-to-read-and-write-algebraic-chess-notation/ \
https://python-chess.readthedocs.io/en/latest/core.html \
https://python-chess.readthedocs.io/en/latest/pgn.html \
https://machinelearningmastery.com/greedy-layer-wise-pretraining-tutorial/ \
https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/ \
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

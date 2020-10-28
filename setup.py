import os
import json

from meta import *

def main():

    print("-------------------")
    print("| DeepChess Setup |")
    print("-------------------\n")

    print("Base Directory: {}\n".format(BASE_DIR))

    if os.path.exists(META_FILE):
        print("Setup already exectued.")
        print("---")
        return 1
    
    meta_def = {
            "dataset_url": "http://ccrl.chessdom.com/ccrl/4040/CCRL-4040.[1183411].pgn.7z",
            "dataset_downloaded": "False",
            "dataset_decompressed": "False",
            "dataset_offsets_calculated": "False",
            "dataset_generated": "False",
            "time_to_calc_dataset_offset": -1,
            "time_to_generate_dataset": -1,
            "compressed_dataset_path": "",
            "dataset_white_path": "",
            "dataset_white_offset_path": "",
            "dataset_black_path": "",
            "dataset_black_offset_path": "",
            "pos2vec_trained": "False",
            "pos2vec_weights_path": "",
            "deepchess_trained": "False",
            "deepchess_model_path": "",
            "max_num_white_positions_train": 2100000,
            "max_num_black_positions_train": 1600000,
            "num_white_positions_train": 0,
            "num_black_positions_train": 0,
            "num_white_positions_test": 100000,
            "num_black_positions_test": 100000,
            "default_engine_path": "engine/stockfish_x64"
        }
    
    with open(META_FILE, 'w') as f:
        json.dump(meta_def, f, indent=4)

    print("meta.json created at:", META_FILE)
    print("Setup complete.")
    print("---")
    

if __name__ == "__main__":
    main()

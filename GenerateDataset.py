import os
import sys
import time
import requests
import json
import glob
import argparse
import threading
from multiprocessing import Process
from pyunpack import Archive

import chess
import chess.pgn
import random
import numpy as np
import pandas as pd

from meta import *

ACCEPTED_COMPRESSED_EXTENSION = ['.7z', '.rar', '.zip', '.tar.gz']
ACCEPTED_DECOMPRESSED_EXTENSION = ['.pgn']

# https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor
def spinning_cursor():

    while True:
        for cursor in '|/-\\':
            yield cursor


def downloadDataset(meta):

    if meta['dataset_downloaded'] == 'False':
        DATASET_URL = meta['dataset_url']
        _, DATASET_FILE_EXTENSION = os.path.splitext(DATASET_URL)
        if DATASET_FILE_EXTENSION not in ACCEPTED_COMPRESSED_EXTENSION:
            print("Invalid dataset URL.")
            return 1

        DATASET_COMPRESSED = os.path.join(
            DATASET_DIR, "DATASET"+DATASET_FILE_EXTENSION)

        r = requests.head(DATASET_URL)
        r_content_size = int(r.headers['Content-Length'])
        if r.status_code != 200:
            print("Dataset not found. Please enter valid dataset link in meta.json")
            return 1

        print('Dataset found!\n     Size: {:,.0f}'.format(
            r_content_size/float(1 << 20))+" MB\n")

        print("Downloading dataset...Please wait")

        # Progress bar - https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
        with open(DATASET_COMPRESSED, 'wb') as f:
            r = requests.get(DATASET_URL, stream=True)
            downloaded = 0
            for data in r.iter_content(chunk_size=max(int(r_content_size/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/r_content_size)
                sys.stdout.write(
                    '\r[{}{}] - [{:,.0f} MB/{:,.0f} MB]'.format('█' * done, '.' * (
                        50-done), downloaded/float(1 << 20), r_content_size/float(1 << 20))
                )
                sys.stdout.flush()
            sys.stdout.write('\n')

        print("Download complete.\n")
        meta['dataset_downloaded'] = 'True'
        meta['compressed_dataset_path'] = DATASET_COMPRESSED

        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=4)

    print("Dataset Uncompressed:", meta['compressed_dataset_path'])
    print("     Size: {:,.0f}MB".format(os.path.getsize(
        meta['compressed_dataset_path'])/float(1 << 20)))
    print("---\n")


def decompressDataset(meta):

    if meta['dataset_decompressed'] == 'False':
        print("Unpacking dataset...Please wait ", end='')

        global decompressing
        decompressing = True

        def decompress():
            Archive(meta['compressed_dataset_path']).extractall(DATASET_DIR)
            global decompressing
            decompressing = False

        tr = threading.Thread(target=decompress)
        tr.start()

        spinner = spinning_cursor()
        while(decompressing):
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            time.sleep(0.3)
            sys.stdout.write('\b')
        sys.stdout.write('\b')
        print('')

        meta['dataset_decompressed'] = 'True'

        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=4)

    print("Dataset unpacked.")
    print("---\n")


def getBitBoard(board):

    white_pawns = board.pieces(chess.PAWN, chess.WHITE).tolist()
    white_rooks = board.pieces(chess.ROOK, chess.WHITE).tolist()
    white_knight = board.pieces(chess.KNIGHT, chess.WHITE).tolist()
    white_bishop = board.pieces(chess.BISHOP, chess.WHITE).tolist()
    white_queen = board.pieces(chess.QUEEN, chess.WHITE).tolist()
    white_king = board.pieces(chess.KING, chess.WHITE).tolist()
    black_pawns = board.pieces(chess.PAWN, chess.BLACK).tolist()
    black_rooks = board.pieces(chess.ROOK, chess.BLACK).tolist()
    black_knight = board.pieces(chess.KNIGHT, chess.BLACK).tolist()
    black_bishop = board.pieces(chess.BISHOP, chess.BLACK).tolist()
    black_queen = board.pieces(chess.QUEEN, chess.BLACK).tolist()
    black_king = board.pieces(chess.KING, chess.BLACK).tolist()

    additional_5 = [board.turn,
                    board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)]

    vector = white_pawns + white_rooks + white_knight + white_bishop + white_queen + white_king + \
        black_pawns + black_rooks + black_knight + \
        black_bishop + black_queen + black_king + additional_5

    bitboard = np.asarray(vector)

    return bitboard

def makeDatasetByColor(pid, dataset_inst, color, max_to_gen, s_idx, end_idx, dataset_name):

    global OFFSETS_WHITE
    global OFFSETS_BLACK

    OFFSET = OFFSETS_WHITE if color == 1 else OFFSETS_BLACK

    DATASET_NPY = os.path.join(DATASET_DIR, dataset_name)
    games_played = 0
    positions_computed = 0
    
    temp_data = np.empty((0, 773), dtype=np.bool)

    while(positions_computed < max_to_gen and (s_idx+games_played) < end_idx):
        if games_played % 5000 == 0:
            with open(DATASET_NPY, 'wb') as f:
                np.save(f, temp_data, allow_pickle=False, fix_imports=False)
            print(dataset_name, "{}: [{} / {}]".format("WHITE" if color == 1 else "BLACK",
                                         positions_computed, max_to_gen))

        dataset_inst.seek(OFFSET[s_idx+games_played])
        current_game = chess.pgn.read_game(dataset_inst)
        ply_cnt = int(current_game.headers['PlyCount'])
        moves_mask = np.asarray([0]*(ply_cnt-15)+[1]*15, dtype=np.bool)
        np.random.shuffle(moves_mask[5:])

        board = current_game.board()
        idx = 0
        rec = 0

        for move in current_game.mainline_moves():
            if rec >= 10:
                break
            if moves_mask[idx] == 0 or board.is_capture(move):
                board.push(move)
                idx += 1
                continue

            bitboard = getBitBoard(board)
            temp_data = np.vstack((temp_data, bitboard))
            positions_computed += 1

            board.push(move)
            rec += 1
            idx += 1
        
        games_played += 1
        
    print(dataset_name, "{}: [{} / {}]".format("WHITE" if color == 1 else "BLACK",
                                 positions_computed, max_to_gen))

    if (temp_data.shape[0] > 0):
        with open(DATASET_NPY, 'wb') as f:
            np.save(f, temp_data, allow_pickle=False, fix_imports=False)


def makeDataset(meta):

    global OFFSETS_WHITE
    global OFFSETS_BLACK

    if meta['dataset_generated'] == "True":
        print("Dataset already generated.")
        print("     WHITE:", meta['dataset_white_path'])
        print("         SHAPE:", meta['num_white_positions_train'])
        print("     BLACK:", meta['dataset_black_path'])
        print("         SHAPE:", meta['num_black_positions_train'])
        print("---\n")
        return

    print("Target number of positions to generate:")
    print("     WHITE:", NUM_WHITE_POSITIONS)
    print("     BLACK:", NUM_BLACK_POSITIONS, "\n")

    dataset_files = glob.glob(os.path.join(
        DATASET_DIR, '*'+ACCEPTED_DECOMPRESSED_EXTENSION[0]))
    dataset_file = dataset_files[0]

    print("Generating datasets...\n")

    TOTAL_NUM_PROCESS_PER_COLOR = 4
    DATA_TO_GEN_P_PROCESS_WHITE = int(NUM_WHITE_POSITIONS / TOTAL_NUM_PROCESS_PER_COLOR)
    DATA_TO_GEN_P_PROCESS_BLACK = int(NUM_BLACK_POSITIONS / TOTAL_NUM_PROCESS_PER_COLOR)
    GAMES_TO_SPLIT_W = int(OFFSETS_WHITE.shape[0] / TOTAL_NUM_PROCESS_PER_COLOR)
    GAMES_TO_SPLIT_B = int(OFFSETS_BLACK.shape[0] / TOTAL_NUM_PROCESS_PER_COLOR)

    start_time = time.time()

    white_processes = []
    black_processes = []

    datasets_suffix_w = "-DATASET-WHITE.npy"
    datasets_suffix_b = "-DATASET-BLACK.npy"

    for pid in range(TOTAL_NUM_PROCESS_PER_COLOR):
        w_start = pid * GAMES_TO_SPLIT_W
        w_end = (pid + 1) * GAMES_TO_SPLIT_W
        b_start = pid * GAMES_TO_SPLIT_B
        b_end = (pid + 1) * GAMES_TO_SPLIT_B

        dataset_inst_white = open(dataset_file, mode='r', encoding="utf-8")
        dataset_inst_black = open(dataset_file, mode='r', encoding="utf-8")

        white_processes.append(
            Process(target=makeDatasetByColor, args=(
                pid, dataset_inst_white, 1, DATA_TO_GEN_P_PROCESS_WHITE, w_start, w_end, str(pid+1)+datasets_suffix_w, ))
        )
        black_processes.append(
            Process(target=makeDatasetByColor, args=(
                pid, dataset_inst_black, 0, DATA_TO_GEN_P_PROCESS_BLACK, b_start, b_end, str(pid+1)+datasets_suffix_b, ))
        )

    for proc in white_processes:
        proc.start()

    for proc in black_processes:
        proc.start()

    for proc in black_processes:
        proc.join()

    for proc in white_processes:
        proc.join()

    time_to_gen = time.time() - start_time

    # CHECK FOR ALL 8 FILES and MERGE THEM TOGETHER
    dataset_gen_files_w = glob.glob(os.path.join(
        DATASET_DIR, '*'+datasets_suffix_w))
    dataset_gen_files_b = glob.glob(os.path.join(
        DATASET_DIR, '*'+datasets_suffix_b))

    print('\n---')
    print('Combining the datasets...')

    temp_data_w = np.empty((0, 773), dtype=np.bool)
    temp_data_b = np.empty((0, 773), dtype=np.bool)

    for file in dataset_gen_files_w:
        with open(file, 'rb') as f:
            np_arr = np.load(f)
        
        temp_data_w = np.vstack((temp_data_w, np_arr))

    W_DATASET_NPY = os.path.join(DATASET_DIR, "WHITE-DATASET.npy")
    with open(W_DATASET_NPY, 'wb') as f:
        np.save(f, temp_data_w, allow_pickle=False, fix_imports=False)

    for file in dataset_gen_files_b:
        with open(file, 'rb') as f:
            np_arr = np.load(f)

        temp_data_b = np.vstack((temp_data_b, np_arr))

    B_DATASET_NPY = os.path.join(DATASET_DIR, "BLACK-DATASET.npy")
    with open(B_DATASET_NPY, 'wb') as f:
        np.save(f, temp_data_b, allow_pickle=False, fix_imports=False)

    delete_f_glob_name = "*-DATASET-*.npy"
    dataset_files = glob.glob(os.path.join(
        DATASET_DIR, delete_f_glob_name))

    for file in dataset_files:
        os.remove(file)

    print('Combined.')
    print("---")

    meta['dataset_white_path'] = W_DATASET_NPY
    meta['dataset_black_path'] = B_DATASET_NPY
    meta['num_white_positions_train'] = temp_data_w.shape[0]
    meta['num_black_positions_train'] = temp_data_b.shape[0]
    meta['dataset_generated'] = "True"
    meta['time_to_generate_dataset'] = round(time_to_gen, 3)

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=4)

    print("\nDatasets generated.")
    print("     WHITE:", W_DATASET_NPY)
    print("         SHAPES:", temp_data_w.shape[0])
    print("     BLACK:", B_DATASET_NPY)
    print("         SHAPES:", temp_data_b.shape[0])
    print("     Time to generate:", meta['time_to_generate_dataset'], "seconds")
    print("---\n")


def makeDatasetOffsets(dataset_inst, COLOR, offset_file_path):

    offsets = []
    while True:
        offset = dataset_inst.tell()
        current_game_headers = chess.pgn.read_headers(dataset_inst)
        if current_game_headers is None:
            break
        
        ply_cnt = int(current_game_headers['PlyCount'])

        if ply_cnt < 20:
            continue

        # 1 if white wins, 0 if black wins, 2 if its a draw
        result = 1 if "1-0" in current_game_headers['Result'] else 0 if "0-1" in current_game_headers['Result'] else 2

        if result == COLOR:
            offsets.append(offset)

    np_offsets = np.asarray(offsets)

    with open(offset_file_path, 'wb') as f:
        np.save(f, np_offsets, allow_pickle=False, fix_imports=False)

    dataset_inst.seek(0)


def calculateDatasetOffsets(meta):

    global OFFSETS_WHITE
    global OFFSETS_BLACK

    if meta['dataset_offsets_calculated'] == "True":
        with open(meta['dataset_white_offset_path'], 'rb') as f:
            OFFSETS_WHITE = np.load(f)

        with open(meta['dataset_black_offset_path'], 'rb') as f:
            OFFSETS_BLACK = np.load(f)
            
        print("Dataset offsets calculated.")
        print("     WHITE:", meta['dataset_white_offset_path'])
        print("         WON:", OFFSETS_WHITE.shape[0], "games")
        print("     BLACK:", meta['dataset_black_offset_path'])
        print("         WON:", OFFSETS_BLACK.shape[0], "games")
        print("---\n")
        return

    dataset_files = glob.glob(os.path.join(
        DATASET_DIR, '*'+ACCEPTED_DECOMPRESSED_EXTENSION[0]))
    dataset_file = dataset_files[0]

    DATASET_OW_PATH = os.path.join(DATASET_DIR, "OFFSETS-White.npy")
    DATASET_OB_PATH = os.path.join(DATASET_DIR, "OFFSETS-Black.npy")

    dataset_inst_white = open(dataset_file, mode='r', encoding="utf-8")
    dataset_inst_black = open(dataset_file, mode='r', encoding="utf-8")

    print("Calculating dataset offsets for white and black...\n")

    white_offsets_p = Process(
        target=makeDatasetOffsets, args=(dataset_inst_white, 1, DATASET_OW_PATH,))
    black_offsets_p = Process(
        target=makeDatasetOffsets, args=(dataset_inst_black, 0, DATASET_OB_PATH,))

    start_time = time.time()

    white_offsets_p.start()
    black_offsets_p.start()

    white_offsets_p.join()
    black_offsets_p.join()

    time_to_calc_offsets = time.time() - start_time

    with open(DATASET_OW_PATH, 'rb') as f:
        OFFSETS_WHITE = np.load(f)

    with open(DATASET_OB_PATH, 'rb') as f:
        OFFSETS_BLACK = np.load(f)

    meta['dataset_white_offset_path'] = DATASET_OW_PATH
    meta['dataset_black_offset_path'] = DATASET_OB_PATH
    meta['dataset_offsets_calculated'] = "True"
    meta['time_to_calc_dataset_offset'] = round(time_to_calc_offsets, 3)

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=4)


    print("\nDatasets offsets calculated.")
    print("     WHITE:", DATASET_OW_PATH)
    print("         WON:", OFFSETS_WHITE.shape[0], "games")
    print("     BLACK:", DATASET_OB_PATH)
    print("         WON:", OFFSETS_BLACK.shape[0], "games")
    print("     Time to calculate:",
          meta['time_to_calc_dataset_offset'], "seconds")
    print("---\n")

def main(args):

    print("-------------------------------")
    print("| DeepChess Dataset Generator |")
    print("-------------------------------\n")
    print("Base Directory:", BASE_DIR)
    print("Dataset Directory:", DATASET_DIR)
    print("meta.json Directory:", META_FILE)
    print("---\n")

    if not os.path.exists(META_FILE):
        print("meta.json file missing. Run `python3 setup.py` first.")
        return 1

    if args.clean:
        files = glob.glob(os.path.join(DATASET_DIR, '*'))
        print('Cleaning up dataset directory...')
        for file in files:
            os.remove(file)

        try:
            os.remove(META_FILE)
            print('Deleted meta.json file.')
        except OSError:
            pass

        print('You can the run setup again.')
        print('\nComplete.')
        return 0

    if args.regenerate:
        with open(META_FILE, 'r') as f:
            meta = json.load(f)

        files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
        print('Cleaning up dataset directory...')
        for file in files:
            os.remove(file)

        meta['dataset_white_path'] = []
        meta['dataset_black_path'] = []
        meta['dataset_generated'] = False
        meta['time_to_generate_dataset'] = -1

        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=4)

        print('Regenerating...\n')

    with open(META_FILE, 'r') as f:
        meta = json.load(f)

    global NUM_WHITE_POSITIONS
    global NUM_BLACK_POSITIONS

    NUM_WHITE_POSITIONS = meta['max_num_white_positions_train']
    NUM_BLACK_POSITIONS = meta['max_num_black_positions_train']

    downloadDataset(meta)
    decompressDataset(meta)
    calculateDatasetOffsets(meta)
    makeDataset(meta)

    print("Complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clean', action="store_true", dest='clean',
                        help='Resets the project. Completely removes everything in the dataset directory and deletes meta.json file.')
    parser.add_argument('--regenerate', action="store_true", dest='regenerate',
                        help='Removes the generated CSV files and regenerates data. This keeps the .pgn file.')

    args = parser.parse_args()

    main(args)

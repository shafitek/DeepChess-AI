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


def makeDatasetByColor(DATASET_NPY, COLOR, dataset_inst):

    positions_count = NUM_WHITE_POSITIONS if COLOR == 1 else NUM_BLACK_POSITIONS
    games_count = positions_count/10
    temp_data = np.empty((0, 773), dtype=np.bool)
    positions_computed = 0

    while(games_count > 0):
        if games_count % 5000 == 0:
            with open(DATASET_NPY, 'wb') as f:
                np.save(f, temp_data, allow_pickle=False, fix_imports=False)
            print("{}: [{} / {}]".format("WHITE" if COLOR == 1 else "BLACK", 
                                         positions_computed, positions_count))

        offset = dataset_inst.tell()
        current_game_headers = chess.pgn.read_headers(dataset_inst)
        ply_cnt = int(current_game_headers['PlyCount'])

        # 1 if white wins, 0 if black wins, 2 if its a draw
        result = 1 if "1-0" in current_game_headers['Result'] else 0 if "0-1" in current_game_headers['Result'] else 2

        if ply_cnt < 30 or result == 2 or result != COLOR:
            continue

        games_count -= 1

        dataset_inst.seek(offset)
        current_game = chess.pgn.read_game(dataset_inst)
        moves_mask = np.random.randint(2, size=ply_cnt)

        board = current_game.board()
        idx = 0
        rec = 0

        for move in current_game.mainline_moves():
            if idx < 5:
                board.push(move)
            else:
                if rec >= 10:
                    break

                if board.is_capture(move) or moves_mask[idx]:
                    board.push(move)
                    idx += 1
                    continue

                rec += 1
                bitboard = getBitBoard(board)
                temp_data = np.vstack((temp_data, bitboard))

                board.push(move)

            idx += 1
        
        positions_computed += 10

    print("{}: [{} / {}]".format("WHITE" if COLOR == 1 else "BLACK",
                                 positions_computed, positions_count))

    if (temp_data.shape[0] > 0):
        with open(DATASET_NPY, 'wb') as f:
            np.save(f, temp_data, allow_pickle=False, fix_imports=False)


def makeDataset(meta):

    print("Number of positions to generate:")
    print("     WHITE:", NUM_WHITE_POSITIONS)
    print("     BLACK:", NUM_BLACK_POSITIONS, "\n")

    DATASET_WHITE_NPY = os.path.join(DATASET_DIR, "DATASET_WHITE.npy")
    DATASET_BLACK_NPY = os.path.join(DATASET_DIR, "DATASET_BLACK.npy")

    dataset_files = glob.glob(os.path.join(
        DATASET_DIR, '*'+ACCEPTED_DECOMPRESSED_EXTENSION[0]))
    dataset_file = dataset_files[0]

    dataset_inst_white = open(dataset_file, mode='r', encoding="utf-8")
    dataset_inst_black = open(dataset_file, mode='r', encoding="utf-8")

    white_p = Process(target=makeDatasetByColor, args=(
        DATASET_WHITE_NPY, 1, dataset_inst_white, ))
    black_p = Process(target=makeDatasetByColor, args=(
        DATASET_BLACK_NPY, 0, dataset_inst_black, ))

    print("Generating datasets...\n")

    start_time = time.time()

    white_p.start()
    black_p.start()

    white_p.join()
    black_p.join()

    time_to_gen = time.time() - start_time

    meta['dataset_white_path'] = DATASET_WHITE_NPY
    meta['dataset_black_path'] = DATASET_BLACK_NPY
    meta['dataset_generated'] = "True"
    meta['time_to_generate_dataset'] = round(time_to_gen, 3)

    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=4)

    with open(DATASET_WHITE_NPY, 'rb') as f:
        white_data_f = np.load(f)

    with open(DATASET_BLACK_NPY, 'rb') as f:
        black_data_f = np.load(f)

    print("\nDatasets generated.")
    print("     WHITE:", DATASET_WHITE_NPY)
    print("         SHAPE:", white_data_f.shape)
    print("     BLACK:", DATASET_BLACK_NPY)
    print("         SHAPE:", black_data_f.shape)
    print("     Time to generate:", meta['time_to_generate_dataset'], "seconds")
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

        meta['dataset_white_path'] = ''
        meta['dataset_black_path'] = ''
        meta['dataset_generated'] = False
        meta['time_to_generate_dataset'] = -1

        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=4)

        print('Regenerating...\n')

    with open(META_FILE, 'r') as f:
        meta = json.load(f)

    if meta['dataset_generated'] == "True":
        print("Dataset already generated.")
        print("---\n")
        return 1

    global NUM_WHITE_POSITIONS
    global NUM_BLACK_POSITIONS

    NUM_WHITE_POSITIONS = meta['num_white_positions_train']
    NUM_BLACK_POSITIONS = meta['num_black_positions_train']

    downloadDataset(meta)
    decompressDataset(meta)
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

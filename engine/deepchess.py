import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import math
import random
import numpy as np

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

import chess
import chess.pgn

sys.path.append("..")
from meta import *

INPUT_DIM = 773
BATCH_SIZE = 64
INIT_LEARNING_RATE = 1e-2

MAX_INF = sys.maxsize
MIN_INF = -sys.maxsize-1

# Force TF to use CPU instead of GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DeepChessEngine:

    def __init__(self, depth, color=True):
        self.__depth = depth
        self.__color = color
        self.move = None

        self.__model = load_model(
            "models/DEEPCHESS.h5", custom_objects={"LeakyReLU": LeakyReLU})


    def __getBitBoard(self, board):

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

        bitboard = np.array(vector, dtype=np.bool).reshape(-1, 773)
        return bitboard
    
    # https://github.com/oripress/DeepChess/blob/master/game.py
    # I used it as a reference from the original implementation
    def __minimaxAlphaBeta(self, board, color, depth, alpha, beta):
        if depth >= self.__depth:
            return board
        # white's turn
        if color:
            v = MIN_INF
            for move in board.legal_moves:
                n_board = board.copy()
                n_board.push(move)

                if v == MIN_INF:
                    v = self.__minimaxAlphaBeta(n_board, False, depth+1, alpha, beta)
                    v_bitboard = self.__getBitBoard(v)
                else:
                    v_bitboard = self.__getBitBoard(v)
                    n_v = self.__minimaxAlphaBeta(n_board, False, depth+1, alpha, beta)
                    n_v_bitboard = self.__getBitBoard(n_v)
                    best_v = self.__model.predict([v_bitboard, n_v_bitboard])[0]
                    if best_v[1] > best_v[0]:
                        v = n_v
                        v_bitboard = n_v_bitboard
                
                if alpha == MIN_INF:
                    alpha = v

                alpha_bitboard = self.__getBitBoard(alpha)
                scores_alpha = self.__model.predict([v_bitboard, alpha_bitboard], batch_size=1)[0]

                if scores_alpha[0] > scores_alpha[1]:
                    alpha = v

                if beta != MAX_INF:
                    beta_bitboard = self.__getBitBoard(beta)
                    scores_beta = self.__model.predict([v_bitboard, beta_bitboard], batch_size=1)[0]
                    if scores_beta[0] > scores_beta[1]:
                        break
            return v
        # black's turn
        else:
            v = MAX_INF
            for move in board.legal_moves:
                n_board = board.copy()
                n_board.push(move)

                if v == MAX_INF:
                    v = self.__minimaxAlphaBeta(n_board, True, depth+1, alpha, beta)
                    v_bitboard = self.__getBitBoard(v)
                else:
                    v_bitboard = self.__getBitBoard(v)
                    n_v = self.__minimaxAlphaBeta(n_board, True, depth+1, alpha, beta)
                    n_v_bitboard = self.__getBitBoard(n_v)
                    best_v = self.__model.predict([v_bitboard, n_v_bitboard], batch_size=1)[0]
                    if best_v[0] > best_v[1]:
                        v = n_v
                        v_bitboard = n_v_bitboard


                if beta == MAX_INF:
                    beta = v

                v_bitboard = self.__getBitBoard(v)
                beta_bitboard = self.__getBitBoard(beta)

                scores_beta = self.__model.predict([beta_bitboard, v_bitboard], batch_size=1)[0]

                if scores_beta[0] > scores_beta[1]:
                    beta = v

                if alpha != MIN_INF:
                    alpha_bitboard = self.__getBitBoard(alpha)
                    scores_alpha = self.__model.predict([alpha_bitboard, v_bitboard])[0]
                    if scores_alpha[0] > scores_alpha[1]:
                        break

            return v


    def play(self, board, dummy=None):

        v = MIN_INF
        alpha = MIN_INF
        beta = MAX_INF

        for move in board.legal_moves:
            n_board = board.copy()
            n_board.push(move)

            if v == MIN_INF:
                v = n_board
                self.move = move
                if alpha == MIN_INF:
                    alpha = v
                continue
            
            v_bitboard = self.__getBitBoard(v)

            n_v = self.__minimaxAlphaBeta(n_board, False, 1, alpha, beta)
            n_v_bitboard = self.__getBitBoard(n_v)

            best_v = self.__model.predict([v_bitboard, n_v_bitboard],batch_size=1)[0]
            if best_v[1] > best_v[0]:
                v = n_v
                v_bitboard = n_v_bitboard

            alpha_bitboard = self.__getBitBoard(alpha)
            scores_alpha = self.__model.predict([v_bitboard, alpha_bitboard], batch_size=1)[0]

            if scores_alpha[0] > scores_alpha[1]:
                alpha = v
                self.move = move

        return self

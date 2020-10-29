import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import math
import random
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

import chess
import chess.pgn

sys.path.append("..")
from meta import *

INPUT_DIM = 773
BATCH_SIZE = 64

class DeepChessEngine:

    def __init__(self, depth, color=True):
        self.__depth = depth
        self.__color = color

        self.move = None

        basic_board = chess.Board()
        basic_board.push(chess.Move.from_uci("h2h3"))
        self.__alpha = basic_board.copy()
        basic_board.push(chess.Move.from_uci("a7a6"))
        self.__beta = basic_board.copy()
        
        del basic_board

        self.__model = load_model(
            "models/BEST/DEEPCHESS.h5", custom_objects={"LeakyReLU": LeakyReLU})


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

        bitboard = np.array(vector, dtype=np.bool)
        tiled_by_batch_size = np.tile(bitboard, (BATCH_SIZE, 1))
        return tiled_by_batch_size
    

    def __minimaxAlphaBeta(self, board, color, depth, alpha, beta):
        if depth == self.__depth:
            return board.copy()
        # white's turn
        if color:
            v = self.__alpha
            for move in board.legal_moves:
                board.push(move)
                v = self.__minimaxAlphaBeta(board.copy(), False, depth+1, alpha, beta)
                board.pop()

                v_bitboard = self.__getBitBoard(v)
                alpha_bitboard = self.__getBitBoard(alpha)
                beta_bitboard = self.__getBitBoard(beta)

                scores_alpha = self.__model.predict([v_bitboard, alpha_bitboard])[0]
                
                if depth == 0:
                    print(scores_alpha)

                if scores_alpha[0] > scores_alpha[1]:
                    alpha = v.copy()
                    if(depth == 0):
                        self.move = move

                scores_beta = self.__model.predict([v_bitboard, beta_bitboard])[0]

                if scores_beta[0] > scores_beta[1]:
                    break
            return v
        # black's turn
        else:
            v = self.__beta
            for move in board.legal_moves:
                board.push(move)
                v = self.__minimaxAlphaBeta(board.copy(), True, depth+1, alpha, beta)
                board.pop()

                v_bitboard = self.__getBitBoard(v)
                alpha_bitboard = self.__getBitBoard(alpha)
                beta_bitboard = self.__getBitBoard(beta)

                scores_beta = self.__model.predict([beta_bitboard, v_bitboard])[0]

                if scores_beta[0] > scores_beta[1]:
                    beta = v.copy()

                scores_alpha = self.__model.predict([alpha_bitboard, v_bitboard])[0]

                if scores_alpha[0] > scores_alpha[1]:
                    break

            return v


    def play(self, board, dummy=None):
        c_board = board.copy()
        self.__minimaxAlphaBeta(c_board, self.__color,
                                0, self.__alpha.copy(), self.__beta.copy())
        if self.move == None:
            print("MOVE IS NONE!!")
        return self

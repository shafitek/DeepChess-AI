import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import random
import numpy as np
from tensorflow.keras.models import Model
import chess

sys.path.append("..")
from meta import *


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

def AlphaBeta(alpha, beta):
    pass

def getNextMove(move):
    # Return a move
    pass

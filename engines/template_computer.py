from os import stat
import chess, random, math, time, chess.pgn
import io
from api.utils.decorators import timer


class Computer:
    def __init__(self, side):
        if side == "b":
            self.white_player = False
        else:
            self.white_player = True

        self.piece_score = {
            1: 10,  # Pawn
            2: 29,  # Knight
            3: 30,  # Bishop
            4: 50,  # Rook
            5: 90,  # Queen
            6: 10000,  # King
        }
        pass

    @staticmethod
    def pgn_2_fenlist(pgn_input):

        # assumption that the pgn is valid
        # for now
        act_pgn = io.StringIO(pgn_input)
        game = chess.pgn.read_game(act_pgn)

        move_list = list()
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            move_list.append(board.fen())
        return move_list

    @staticmethod
    def fenlist_2_pgn(fenlist):

        # not finished
        # issues with result
        game = chess.pgn.Game()
        game.headers["Event"] = "What"
        game.headers["White"] = "Baboon"
        game.headers["Black"] = "Bagoon"

        # dont know if going to use it
        # so suck it future me
        m = Computer.__find_move(chess.Board().fen(), fenlist[0])
        node = game.add_variation(m)
        for i in range(1, len(fenlist)):
            m = Computer.__find_move(fenlist[i - 1], fenlist[i])
            node = node.add_variation(m)

        # game.headers["Result"] = game.board().result()
        # print(game)
        return game

    @staticmethod
    def __find_move(fen1, fen2):

        board = chess.Board(fen1)
        available_moves = board.legal_moves

        for move in available_moves:
            board.push(move)

            if board.fen() == fen2:
                return move
            else:
                board.pop()

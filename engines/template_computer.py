from os import stat
import chess, random, math, time, chess.pgn
import io
# from api.utils.decorators import timer


class Computer:
    piece_score = {
        1: 100,  # Pawn
        2: 300,  # Knight
        3: 300,  # Bishop
        4: 500,  # Rook
        5: 900,  # Queen
        6: 100000,  # King
    }

    piece_score_text = {
        "p": piece_score[1],
        "n": piece_score[2], 
        "b": piece_score[3],
        "r": piece_score[4],
        "q": piece_score[5],
        "k": piece_score[6],
    }
    def __init__(self, side):
        if side == "b":
            self.white_player = False
        else:
            self.white_player = True

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

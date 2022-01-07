from api.engines.random_engine.computer import Computer
import chess, math, time
from api.engines.template_computer import Computer
import api.utils.decorators as d
from pprint import pprint
from api.utils.logger import MyLogger
import cProfile
from collections import Counter

logger = MyLogger(__name__)

class MiniMaxComputer(Computer):
    """
    simple minimax engine, with alpha-beta pruning and transposition table
    """

    def __init__(self, side="b", depth=None):
        if depth == None:
            self.depth = 5
        else:
            self.depth = depth

        self.transposition_table = {}
        super().__init__(side)

    @d.timer_log
    def think(self, fen: str) -> chess.Move:
        self.transposition_table = {}
        board = chess.Board(fen)
        move, eval = self.minimax(
            board, self.depth, -math.inf, math.inf, self.white_player
        )
        # pprint(self.transposition_table)
        return move

    def evaluate_position(self, board):
        score = 0
        for key in self.piece_score:
            score += (
                len(board.pieces(key, True)) - len(board.pieces(key, False))
            ) * self.piece_score[key]

        score *= 1 if board.turn else -1
        t = board.fen().split(" ")
        t.pop()
        self.transposition_table[" ".join(t)] = score
        return score

    def minimax(self, board, depth, alpha, beta, white_player):
        if depth == 0 or board.is_game_over():

            # handler for when the game is over
            b = board.outcome()
            if b is not None:
                if b.winner == True:
                    # white won
                    return None, self.piece_score[6]
                elif b.winner == False:
                    # black won
                    return None, -self.piece_score[6]
                else:
                    # draw
                    return None, 0

            # the depth limit has been reached
            else:
                return None, self.evaluate_position(board)

        best_move = None
        if white_player:
            max_eval = -math.inf

            # create legal moves for a given position
            legal_moves = list(board.legal_moves)
            for move in legal_moves:

                # try out every move
                board.push(move)
                t = board.fen().split(" ")
                t.pop()

                # look up if already evaluated, if not recurrency
                if " ".join(t) in self.transposition_table:
                    eval = self.transposition_table[" ".join(t)]
                else:
                    ret_move, eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                # alpha-beta pruning, and picking the best move
                if eval >= max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta < eval:
                    break
            return best_move, max_eval

        else:
            min_eval = math.inf

            # create legal moves for a given position
            legal_moves = list(board.legal_moves)
            for move in legal_moves:

                # try out every move
                board.push(move)
                t = board.fen().split(" ")
                t.pop()

                # look up if already evaluated, if not recurrency
                if " ".join(t) in self.transposition_table:
                    eval = self.transposition_table[" ".join(t)]
                else:
                    ret_move, eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                # alpha-beta pruning, and picking the best move
                if eval <= min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta < eval:
                    break
            return best_move, min_eval

    @staticmethod
    def piece_2_number(letter):
        temp = {
            "p": 1,
            "r": 2,
            "n": 3,
            "b": 4,
            "q": 5,
            "k": 6,
            "P": 7,
            "R": 8,
            "N": 9,
            "B": 10,
            "Q": 11,
            "K": 12
        }
        return temp[letter]

    def init_zobrist(self):
        pass

    def zorbist_hash(self, board):
        pass

if __name__ == "__main__":

    pc = MiniMaxComputer()
    fen = "r1bqk1nr/ppp2ppp/2p5/2b1p3/4P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 1 5"
    cProfile.run('pc.think("r1bqk1nr/ppp2ppp/2p5/2b1p3/4P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 1 5")', sort="tottime")

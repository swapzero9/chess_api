from audioop import reverse
import random
from api.engines.random_engine.computer import Computer
import chess, math, time
from api.engines.template_computer import Computer
import api.utils.decorators as d
from pprint import pprint
from api.utils.logger import MyLogger
import chess.syzygy

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
        self.s_path = "api/engines/minimax_engine/syzygy_tables"
        self.transposition_table = {}
        
        super().__init__(side)
        with open("api/engines/minimax_engine/opening_database.json") as f:
            from json import load
            self.opening_db = load(f)

    @d.timer_log
    def think(self, fen: str) -> chess.Move:
        self.transposition_table = dict()
        board = chess.Board(fen)

        num_pieces = 0
        for piece in range(1, 7):
            num_pieces += len(board.pieces(piece, True)) + len(board.pieces(piece, False))
        
        if board.fullmove_number <= 10:
            move = self.opening_lookup(board)
        elif num_pieces > 6:
            move, _ = self.minimax(board, self.depth, -math.inf, math.inf)
        else:
            move, _ = self.syzygy_lookup(board)

        return move

    def syzygy_lookup(self, node):
        with chess.syzygy.open_tablebase(self.s_path) as tablebase:
            start = tablebase.probe_dtz(node)
            best_move = None
            goal = -math.inf if start > 0 else 0
            if start > 0:
                for move in node.legal_moves:
                    node.push(move)
                    ev = tablebase.probe_dtz(node)
                    if ev > goal and ev < 0:
                        goal = tablebase.probe_dtz(node)
                        best_move = move
                    node.pop()
            elif start < 0:
                for move in node.legal_moves:
                    node.push(move)
                    ev = tablebase.probe_dtz(node)
                    if ev > goal:
                        goal = tablebase.probe_dtz(node)
                        best_move = move
                    node.pop()
            else: # start = 0
                for move in node.legal_moves:
                    node.push(move)
                    ev = tablebase.probe_dtz(node)
                    node.pop()
                    if ev == 0:
                        return move
            return best_move

    def opening_lookup(self, node:chess.Board):
        nfen = node.fen().replace("/", "_").replace(" ", "")
        if nfen in self.opening_db:
            visit_sum = 0
            for _, value in self.opening_db[nfen].items():
                visit_sum += value
            
            r = random.randint(1, visit_sum) - 1
            cumsum = 0
            for move, value in self.opening_db[nfen].items():
                cumsum += value
                if cumsum > r:
                    return chess.Move.from_uci(move)
        else:
            move, _ = self.minimax(node, self.depth, -math.inf, math.inf)
            return move

    def evaluate_position(self, board):
        score = 0

        # number of pieces
        for key in self.piece_score:
            score += (
                len(board.pieces(key, True)) - len(board.pieces(key, False))
            ) * self.piece_score[key]

        # pawn advancement
        if board.turn:
            score += sum([int(i/8) + 1 for i in board.pieces(1, True)]) / 4
        else:
            score += sum([(8 - int(i/8)) for i in board.pieces(1, False)]) / 4

        # piece activity
        score += sum([len(board.attackers(board.turn, sq)) for sq in range(64)]) / 6

        score *= 1 if board.turn else -1
        return score

    def minimax(self, node, depth, alpha, beta):
        if node.is_game_over():
            # game ended give scores accordingly
            b = node.outcome()
            if b.winner:
                return None,  100000
            elif not b.winner:
                return None, -100000
            else:
                return None, 0
        elif depth == 0:
            return None, self.evaluate_position(node)
        
        best_move:chess.Move = None
        if node.turn: # whites turn looking for max
            goal = -math.inf
            legal = list(node.legal_moves)
            legal.sort(key=lambda x: node.is_capture(x), reverse=True)
            legal.sort(key=lambda x: self.test_check(node, x), reverse=True)
            legal.sort(key=lambda x: self.test_checkmate(node, x), reverse=True)
            for move in legal:

                node.push(move)
                z_hash = node.zobrist_hash()
                if z_hash in self.transposition_table:
                    eval = self.transposition_table[z_hash]
                else:
                    _, eval = self.minimax(node, depth-1, alpha, beta)
                    self.transposition_table[z_hash] = eval
                node.pop()

                if eval > goal:
                    goal = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                
            return best_move, goal
        else:
            goal = math.inf
            legal = list(node.legal_moves)
            legal.sort(key=lambda x: node.is_capture(x), reverse=True)
            legal.sort(key=lambda x: self.test_check(node, x), reverse=True)
            legal.sort(key=lambda x: self.test_checkmate(node, x), reverse=True)
            for move in legal:

                node.push(move)
                z_hash = node.zobrist_hash()
                if z_hash in self.transposition_table:
                    eval = self.transposition_table[z_hash]
                else:
                    _, eval = self.minimax(node, depth-1, alpha, beta)
                    self.transposition_table[z_hash] = eval
                node.pop()

                if eval < goal:
                    goal = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, goal

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

    @staticmethod
    def test_check(node, move):
        node.push(move)
        r = node.is_check()
        node.pop()
        return r

    @staticmethod
    def test_checkmate(node, move):
        node.push(move)
        r = node.is_checkmate()
        node.pop()
        return r

if __name__ == "__main__":

    pc = MiniMaxComputer()
    print(pc.think("8/5ppp/R7/4p1k1/1r1p4/1n2PPP1/2r1NK1P/3R4 w - - 0 33"))

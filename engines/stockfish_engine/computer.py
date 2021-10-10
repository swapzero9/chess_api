from api.engines.template_computer import Computer
from api.utils.logger import MyLogger
from stockfish import Stockfish
import api.utils.decorators as d
import chess

module_logger = MyLogger(__name__)

class StockfishComputer(Computer):

    def __init__(self, side, elo, timeout=100):
        super().__init__(side)
        self.model = Stockfish("./api/engines/stockfish_engine/stockfish.exe")
        self.timeout = timeout
        self.starting_elo = elo
        self.current_elo = elo
        self.model.set_elo_rating(elo)
        self.model.set_depth(5)

        module_logger().info(self.model.get_parameters())

    def think(self, fen):
        board = chess.Board(fen)
        self.model.set_fen_position(fen)
        move = chess.Move.from_uci(self.model.get_best_move_time(self.timeout)) # uci move

        if move in board.legal_moves:
            return move
        else:
            return list(board.legal_moves)[0]

    def set_timeout(self, timeout):
        self.timeout = timeout

    def improve(self, amount):
        self.current_elo += amount
        self.model.set_elo_rating(self.current_elo)

    def reset_elo(self):
        self.model.set_elo_rating(self.starting_elo)

if __name__ == "__main__":

    # example
    stock = StockfishComputer("b")
    a = stock.think("rn1qkbnr/ppp3pp/5p2/3p4/3Pp1Q1/4P3/PPP2PPP/RNB1KB1R b KQkq - 0 6")
    print(a)
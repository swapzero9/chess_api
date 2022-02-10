from api.engines.template_computer import Computer
from api.utils.logger import MyLogger
import chess.engine
import chess
import api.utils.decorators as d

module_logger = MyLogger(__name__)

class StockfishComputer(Computer):

    def __init__(self, side, elo, timeout=100):
        super().__init__(side)
        self.engine_path = "./api/engines/stockfish_engine/stockfish.exe"
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.timeout = timeout
        self.starting_elo = elo
        self.current_elo = elo

    def __del__(self):
        print("deleted")
        self.engine.quit()

    def think(self, fen):
        board = chess.Board(fen)
        info = self.engine.analyse(board, chess.engine.Limit(depth=10))
        return info["pv"][0]

    def set_timeout(self, timeout):
        self.timeout = timeout

    def improve(self, amount):
        self.current_elo += amount
        self.model.set_elo_rating(self.current_elo)

    def reset_elo(self):
        self.model.set_elo_rating(self.starting_elo)

if __name__ == "__main__":

    # example
    stock = StockfishComputer("b", 200)
    a = stock.think("rn1qkbnr/ppp3pp/5p2/3p4/3Pp1Q1/4P3/PPP2PPP/RNB1KB1R b KQkq - 0 6")
    print(a)
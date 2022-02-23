from matplotlib import path
from api.engines.stockfish_engine.computer import StockfishComputer
from api.utils.logger import MyLogger
from datetime import datetime
import api.utils.decorators as d
import chess, chess.pgn
import os, os.path, sys


module_logger = MyLogger(__name__)

class Validation:
    """
    play n matches in batch of 10 and validate how good the engine really is
    """

    def __init__(self, player1, *, player2=None, batches=500, path=None):
        self.player1 = player1
        if player2 is None:
            self.player2 = StockfishComputer("b", 200)
        else:
            self.player2 = player2

        self.total_batches = batches
        self.path = path

    @d.timer_log
    def __call__(self):
        for _ in range(self.total_batches):            
            self.single_valid_game(self.player1, self.player2)
            self.single_valid_game(self.player2, self.player1)


    @d.timer_log
    def single_valid_game(self, player_white, player_black):
        # start a game
        game = chess.Board()
        pgn = chess.pgn.Game()
        print("started the game")
        pgn.headers["White"] = player_white.__class__.__name__
        pgn.headers["Black"] = player_black.__class__.__name__
        pgn.setup(game)
        node = None
        is_first_move = True
        while not game.is_game_over():
            move = None
            # move if white
            if game.turn:
                move = player_white.think(game.fen())
                assert move in game.legal_moves
                game.push(move)

            # move if black
            else:
                move = player_black.think(game.fen())
                assert move in game.legal_moves
                game.push(move)

            if is_first_move:
                node = pgn.add_variation(move)
                is_first_move = False
            else:
                node = node.add_variation(move)

        pgn.headers["Result"] = game.result()
        print(f"-----------------------------------------------------------\nGAME FINISHED\n{pgn}\n")
        with open(f"{self.path}/{datetime.now().strftime('%d%m%Y_%H%M%S')}.pgn", mode="w") as f:
            f.write(str(pgn))


from api.engines.ai_engine.computer import AiComputer
from api.engines.ai_engine.models.final.net import Net as Net1

from api.engines.ai_engine_new.computer import AiComputer2
from api.engines.ai_engine_new.models.final.net import Net as Net2

from api.engines.minimax_engine.computer import MiniMaxComputer

def ai_vs_ai():
    p1 = AiComputer(load_model=True, model_name="final.pt", net=Net1)
    p2 = AiComputer2(load_model=True, model_name="final.pt", net=Net2)
    v = Validation(
        player1=p1,
        player2=p2,
        batches=100,
        path="./validation_games/ai1_vs_ai2"
    )
    v()

def ai1_vs_stockfish():
    p1 = AiComputer(load_model=True, model_name="final.pt", net=Net1)
    v = Validation(
        player1=p1,
        batches=100,
        path="./validation_games/ai1_vs_stockfish"
    )
    v()

def ai1_vs_minmax():
    p1 = AiComputer(load_model=True, model_name="final.pt", net=Net1)
    p2 = MiniMaxComputer()
    v = Validation(
        player1=p1,
        player2=p2,
        batches=100,
        path="./validation_games/ai1_vs_minmax"
    )
    v()

def ai2_vs_stockfish():
    p1 = AiComputer2(load_model=True, model_name="final.pt", net=Net2)
    v = Validation(
        player1=p1,
        batches=100,
        path="./validation_games/ai2_vs_stockfish"
    )
    v()

def ai2_vs_minmax():
    p1 = AiComputer2(load_model=True, model_name="final.pt", net=Net2)
    p2 = MiniMaxComputer()
    v = Validation(
        player1=p1,
        player2=p2,
        batches=100,
        path="./validation_games/ai2_vs_minmax"
    )
    v()

if __name__ == "__main__":
    # p = sys.argv[1]

    # if os.path.exists(p) and os.path.isdir(p):
    #     from api.engines.montecarlo_engine.computer import MonteCarloComputer
    #     from api.engines.minimax_engine.computer import MiniMaxComputer
    #     m = MonteCarloComputer(10)
    #     b = MiniMaxComputer(depth=5)
        
    #     v = Validation(b, player2=m, batches=100, path=p)
    #     v()
    import multiprocessing as mp

    processes = [
        # ai1_vs_stockfish,
        # ai1_vs_minmax, 
        ai_vs_ai, 
        ai_vs_ai, 
        ai_vs_ai, 
        ai_vs_ai, 
        # ai2_vs_stockfish,
        # ai2_vs_minmax
    ]

    pipi = list()
    for p in processes:
        proc = mp.Process(target=p)
        proc.start()
        pipi.append(proc)

    for p in pipi:
        p.join()

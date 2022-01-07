from api.utils.logger import MyLogger
import api.utils.decorators as d
import chess, chess.pgn
import itertools
import multiprocessing as mp
from api.engines.ai_engine_new.computer import AiComputer2
from api.engines.ai_engine_new.models.architecture1.net import Net


module_logger = MyLogger(__name__, MyLogger.DEBUG)

class TrainingSession:
    def __init__(self, name:str, eng, net, amount):
        self.eng = eng
        self.net = net
        self.amount_of_iterations = amount
        self.games_in_iteration = 10

    d.timer_log
    def train(self):

        rang = range(self.amount_of_iterations) if self.amount_of_iterations is not None else itertools.count()
        self.manager = mp.Manager()
        for i in rang:
            queue = self.manager.list()

            for j in range(self.games_in_iteration):
                queue.append([self.eng, self.net, j*2, ""])
                queue.append([self.eng, self.net, j*2+1, ""])

            with mp.Pool(processes=4) as pool:
                pool.map(TrainingSession.single_game, queue)


    @staticmethod
    @d.timer
    def single_game(args):

        player_white = args[0](net=args[1], seed=args[2])
        player_black = args[0](net=args[1], seed=args[2])
        ind = args[2]

        print(f"game {ind} started")

        # start a game
        game = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["White"] = player_white.__class__.__name__
        pgn.headers["Black"] = player_black.__class__.__name__
        pgn.setup(game)
        node = None
        whites_turn = True
        is_first_move = True
        while not game.is_game_over():
            is_legal = False
            legal_moves = game.legal_moves
            move = None
            # move if white
            if whites_turn:
                while not is_legal:
                    move = player_white.think(game.fen())
                    if move in legal_moves:
                        game.push(move)
                        whites_turn = False
                        is_legal = True

            # move if black
            else:
                while not is_legal:
                    move = player_black.think(game.fen())
                    if move in legal_moves:
                        game.push(move)
                        whites_turn = True
                        is_legal = True
            if is_first_move:
                node = pgn.add_variation(move)
                is_first_move = False
            else:
                node = node.add_variation(move)

        wc = player_white.__class__.__name__ if game.result() == "1-0" else (player_black.__class__.__name__ if game.result() == "0-1" else "none")

        pgn.headers["Result"] = game.result()
        print(f"game {ind} finished")
        print(pgn)

if __name__ == "__main__":
    t = TrainingSession("penis", AiComputer2, Net, 6)
    t.train()

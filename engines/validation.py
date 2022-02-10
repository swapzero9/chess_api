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

        # # create validation node related to training node
        # self.validation_session_node = Node(
        #     "ValidationSession",
        #     player1=self.player1.__class__.__name__,
        #     player2=self.player2.__class__.__name__,
        #     date_start=datetime.today(),
        #     games_in_batch=self.games_in_batch,
        #     total_batches=self.total_batches,
        # )
        # validation_rel = Relationship(self.training_node, "ValidatingTraining", self.validation_session_node)

        for i in range(self.total_batches):

            # validation_iter = Node(
            #     "ValidationSessionIteration",
            #     iteration=i,
            #     timestamp=datetime.today(),
            # )
            # validation_rel = Relationship(self.validation_session_node, "Iteration", validation_iter)

            # self.create_db_elements([
            #     validation_iter,
            #     validation_rel
            # ])
            
            self.single_valid_game(self.player1, self.player2)
            self.single_valid_game(self.player2, self.player1)


    @d.timer_log
    def single_valid_game(self, player_white, player_black):
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
            print("yes")
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

        pgn.headers["Result"] = game.result()
        print(f"-----------------------------------------------------------\nGAME FINISHED\n{pgn}\n")
        with open(f"{self.path}/{datetime.now().strftime('%d%m%Y_%H%M%S')}.pgn", mode="w") as f:
            f.write(str(pgn))
        # game_node = Node(
        #     "Game",
        #     timestamp=datetime.today(),
        #     number_of_moves=math.ceil(len(list(pgn.mainline_moves())) / 2),
        #     is_checkmate=game.is_checkmate(),
        #     is_stalemate=game.is_stalemate(),
        #     is_insufficient=game.is_insufficient_material(),
        #     winner=game.result(),
        #     winner_c=wc,
        #     game_pgn=str(pgn),
        # )
        # played_relationship = Relationship(iter_node, "Played", game_node)
        # self.create_db_elements([
        #     game_node,
        #     played_relationship
        # ])

    # def create_db_elements(self, ar):
    #     try:
    #         tx = self.db.begin()
    #         for el in ar:
    #             tx.create(el)
    #         self.db.commit(tx)
    #     except Exception as ex:
    #         module_logger().exception(ex)


if __name__ == "__main__":
    p = sys.argv[1]
    

    if os.path.exists(p) and os.path.isdir(p):
        from api.engines.montecarlo_engine.computer import MonteCarloComputer
        from api.engines.minimax_engine.computer import MiniMaxComputer
        m = MonteCarloComputer(2000)
        b = MiniMaxComputer()
        
        v = Validation(b, player2=m, batches=100, path=p)
        v()

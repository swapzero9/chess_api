from api.engines.stockfish_engine.computer import StockfishComputer
from api.utils.garbage_functions import method_exists
from api.utils.logger import MyLogger
from datetime import datetime
from py2neo import Graph, Node, Relationship
import api.utils.decorators as d
import chess, chess.pgn, math


module_logger = MyLogger(__name__)

class Validation:
    """
    play n matches in batch of 10 and validate how good the engine really is
    """

    def __init__(self, training_node, player1, *, player2=None, batches=2):

        self.db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))
        self.player1 = player1
        if player2 is None:
            self.player2 = StockfishComputer("b", 200)
        else:
            self.player2 = player2

        self.games_in_batch = 4
        self.total_batches = batches
        self.training_node = training_node

    @d.timer_log
    def __call__(self):

        # create validation node related to training node
        self.validation_session_node = Node(
            "ValidationSession",
            player1=self.player1.__class__.__name__,
            player2=self.player2.__class__.__name__,
            date_start=datetime.today(),
            games_in_batch=self.games_in_batch,
            total_batches=self.total_batches,
        )
        validation_rel = Relationship(self.training_node, "ValidatingTraining", self.validation_session_node)

        self.create_db_elements([
            self.validation_session_node,
            validation_rel,
        ])

        for i in range(self.total_batches):

            validation_iter = Node(
                "ValidationSessionIteration",
                iteration=i,
                timestamp=datetime.today(),
            )
            validation_rel = Relationship(self.validation_session_node, "Iteration", validation_iter)

            self.create_db_elements([
                validation_iter,
                validation_rel
            ])
            
            invert = False
            for game in range(self.games_in_batch):
                # handle game and create node with relation to iteration node
                
                if invert:
                    self.single_valid_game(validation_iter, self.player1, self.player2)
                else:
                    self.single_valid_game(validation_iter, self.player2, self.player1)
                invert = invert == False

            # improve engine if possible
            if method_exists(self.player1, "improve"):
                self.player1.improve(200)

            if method_exists(self.player2, "improve"):
                self.player2.improve(200)

        # reset elo of engines
        if method_exists(self.player1, "reset_elo"):
            self.player1.reset_elo()

        if method_exists(self.player2, "reset_elo"):
            self.player2.reset_elo()


    @d.timer_log
    def single_valid_game(self, iter_node, player_white, player_black):
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
        game_node = Node(
            "Game",
            timestamp=datetime.today(),
            number_of_moves=math.ceil(len(list(pgn.mainline_moves())) / 2),
            is_checkmate=game.is_checkmate(),
            is_stalemate=game.is_stalemate(),
            is_insufficient=game.is_insufficient_material(),
            winner=game.result(),
            winner_c=wc,
            game_pgn=str(pgn),
        )
        played_relationship = Relationship(iter_node, "Played", game_node)
        self.create_db_elements([
            game_node,
            played_relationship
        ])

    def create_db_elements(self, ar):
        try:
            tx = self.db.begin()
            for el in ar:
                tx.create(el)
            self.db.commit(tx)
        except Exception as ex:
            module_logger().exception(ex)
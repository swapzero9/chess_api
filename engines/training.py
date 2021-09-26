import chess, chess.pgn, math
from datetime import datetime
import api.utils.decorators
from api.utils.garbage_functions import method_exists

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

class TrainingSession:
    def __init__(self, name:str, engine_white, engine_black, amount: int = 10):
        self.engine_white = engine_white
        self.engine_black = engine_black
        self.amount_of_iterations = amount
        self.games_in_iteration = 10

        # self.training_session_node = NeoNode("TrainingNode", d)
        self.training_session_node = Node(
            "TrainingNode",
            name=name,
            date_start=datetime.today(),
            amount=self.amount_of_iterations * self.games_in_iteration,
        )
        self.db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))

        tx = self.db.begin()
        tx.create(self.training_session_node)
        self.db.commit(tx)

    @api.utils.decorators.timer
    def train(self):
        for i in range(self.amount_of_iterations):

            # iteration node related to training_session_node
            iteration_node = Node(
                "Training_iteration",
                iteration=i,
                timestamp=datetime.today()
            )
            iter_relationship = Relationship(self.training_session_node, "Iteration", iteration_node)
            tx = self.db.begin()
            tx.create(iteration_node)
            tx.create(iter_relationship)
            self.db.commit(tx)

            for j in range(self.games_in_iteration):
                # multiThreading?
                self.single_game(iteration_node)

            # get games from the current iteration
            rel_matcher = RelationshipMatcher(self.db)
            q = rel_matcher.match((iteration_node, None), "Played").all()

            # training and stuff
            if method_exists(self.engine_white, "learn"):
                self.engine_white.learn("w", q)

            if method_exists(self.engine_white, "save_model"):
                self.engine_white.save_model()

            if method_exists(self.engine_black, "learn"):
                self.engine_white.learn("b", q)

            if method_exists(self.engine_black, "save_model"):
                self.engine_white.save_model()

            # swap engines around
            # for the future boi
            # temp = self.engine_white
            # self.engine_white = self.engine_black
            # self.engine_black = temp


    @api.utils.decorators.timer
    def single_game(self, iter_node):
        # start a game
        game = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["White"] = self.engine_white.__name__
        pgn.headers["Black"] = self.engine_black.__name__
        pgn.setup(game)
        node = None
        whites_turn = True
        is_first_move = True
        while not game.is_game_over():
            # print("---i--------------------------------")
            # print(game)
            is_legal = False
            legal_moves = game.legal_moves
            move = None
            # move if white
            if whites_turn:
                while not is_legal:
                    move = self.engine_white.think(game.fen())
                    if move in legal_moves:
                        game.push(move)
                        whites_turn = False
                        is_legal = True

            # move if black
            else:
                while not is_legal:
                    move = self.engine_black.think(game.fen())
                    if move in legal_moves:
                        game.push(move)
                        whites_turn = True
                        is_legal = True
            if is_first_move:
                node = pgn.add_variation(move)
                is_first_move = False
            else:
                node = node.add_variation(move)

        wc = self.engine_white.__name__ if game.result() == "1-0" else (self.engine_black.__name__ if game.result() == "0-1" else "none")

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
        tx = self.db.begin()
        tx.create(game_node)
        tx.create(played_relationship)
        self.db.commit(tx)

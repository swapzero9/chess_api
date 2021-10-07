import itertools
import chess, chess.pgn, math, torch
from datetime import datetime
import api.utils.decorators
from api.utils.garbage_functions import method_exists

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

class TrainingSession:
    def __init__(self, name:str, p1, p2=None, amount=None):
        self.player1 = p1
        if p2 is None:
            self.save_twice = False
            self.player2 = p1
        else:
            self.save_twice = True
            self.player2 = p2
        self.amount_of_iterations = amount
        self.games_in_iteration = 3

        # self.training_session_node = NeoNode("TrainingNode", d)
        self.training_session_node = Node(
            "TrainingNode",
            name=name,
            date_start=datetime.today(),
            amount=("Infinity" if self.amount_of_iterations is None else self.amount_of_iterations * self.games_in_iteration),
        )
        self.db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))

        tx = self.db.begin()
        tx.create(self.training_session_node)
        self.db.commit(tx)

    @api.utils.decorators.timer
    def train(self):

        def count(start=0, step=1):
            # count(10) --> 10 11 12 13 14 ...
            # count(2.5, 0.5) -> 2.5 3.0 3.5 ...
            n = start
            while True:
                yield n
                n += step

        rang = range(self.amount_of_iterations) if self.amount_of_iterations is not None else itertools.count()
        invert = True
        for i in rang:

            # iteration node related to training_session_node
            iteration_node = Node(
                "TrainingIteration",
                iteration=i,
                timestamp=datetime.today()
            )
            i += 1
            iter_relationship = Relationship(self.training_session_node, "Iteration", iteration_node)
            tx = self.db.begin()
            tx.create(iteration_node)
            tx.create(iter_relationship)
            self.db.commit(tx)

            # single game
            if invert:
                self.single_game(iteration_node, self.player1, self.player2)
            else:
                self.single_game(iteration_node, self.player2, self.player1)

            # get games from the current iteration
            rel_matcher = RelationshipMatcher(self.db)
            q = rel_matcher.match((iteration_node, None), "Played").all()

            # training and stuff
            if method_exists(self.player1, "learn"):
                pass
                self.player1.learn("w", q)
                

            if method_exists(self.player2, "learn"):
                self.player2.learn("b", q)

            if i % 1000 == 0:
                if method_exists(self.player1, "save_model"):
                    self.player1.save_model()

                if self.save_twice and method_exists(self.player2, "save_model"):
                    self.player2.save_model()


    @api.utils.decorators.timer
    def single_game(self, iter_node, player_white, player_black):
        # start a game
        game = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["White"] = player_white.__name__
        pgn.headers["Black"] = player_black.__name__
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

        wc = player_white.__name__ if game.result() == "1-0" else (player_black.__name__ if game.result() == "0-1" else "none")

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

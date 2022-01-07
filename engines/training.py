from api.engines.validation import Validation
from api.utils.garbage_functions import method_exists
from api.utils.logger import MyLogger
from datetime import datetime
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
import api.utils.decorators as d
import chess, chess.pgn, math
import itertools

module_logger = MyLogger(__name__, MyLogger.DEBUG)

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
        self.games_in_iteration = 4


        # self.training_session_node = NeoNode("TrainingNode", d)
        self.training_session_node = Node(
            "TrainingNode",
            name=name,
            date_start=datetime.today(),
            amount=("Infinity" if self.amount_of_iterations is None else self.amount_of_iterations * self.games_in_iteration),
        )
        self.db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))

        # check if training session already exists with the given name
        # if yes raise an exception
        matcher = NodeMatcher(self.db)
        res = matcher.match("TrainingNode", name=name).all()
        # assert len(res) == 0 # node already exists

        self.validation = Validation(self.training_session_node, self.player1)
        self.create_db_elements([self.training_session_node])

    @d.timer_log
    def train(self):

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
            iter_relationship = Relationship(self.training_session_node, "GameIteration", iteration_node)
            self.create_db_elements([
                iteration_node,
                iter_relationship
            ])

            invert = invert == False
            if invert:
                self.single_game(iteration_node, self.player1, self.player2)
            else:
                self.single_game(iteration_node, self.player2, self.player1)

            # get games from the current iteration
            rel_matcher = RelationshipMatcher(self.db)
            q = rel_matcher.match((iteration_node, None), "Played").all()

            # training and stuff
            if method_exists(self.player1, "learn"):
                self.player1.learn("w", q)

            if method_exists(self.player2, "learn"):
                self.player2.learn("b", q)

            # save model every n-th game
            if i % 1000 == 0:
                if method_exists(self.player1, "save_model"):
                    self.player1.save_model()

                if self.save_twice and method_exists(self.player2, "save_model"):
                    self.player2.save_model()

            # validate against real engine
            if i % 100 == 0:
                self.validation()


    @d.timer_log
    def single_game(self, iter_node, player_white, player_black):

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
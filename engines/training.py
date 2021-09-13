import chess, chess.pgn, math
from datetime import datetime
import api.utils.decorators

# from api.database.neo4j_driver import Neo4j_Driver, NeoNode, NeoRelationship
from py2neo import Graph, Node, Relationship

class TrainingSession:
    def __init__(self, engine_white, engine_black, amount: int = 10):
        self.engine_white = engine_white
        self.engine_black = engine_black
        self.amount_of_games = amount

        # self.training_session_node = NeoNode("TrainingNode", d)
        self.training_session_node = Node(
            "TrainingNode",
            name="random Test",
            date_start=datetime.today(),
            amount=self.amount_of_games,
        )
        self.db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))

        tx = self.db.begin()
        tx.create(self.training_session_node)
        self.db.commit(tx)

    def train(self):
        for i in range(self.amount_of_games):
            # start a game
            game = chess.Board()
            pgn = chess.pgn.Game()
            pgn.headers["White"] = "rufus"
            pgn.headers["Black"] = "doofus"
            pgn.setup(game)
            node = None
            whites_turn = True
            is_first_move = True
            while not game.is_game_over():
                # print("------------------------------------")
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

            pgn.headers["Result"] = game.result()
            print(type(str(pgn)))
            game_node = Node(
                "Game",
                timestamp=datetime.today(),
                number_of_moves=math.ceil(len(list(pgn.mainline_moves())) / 2),
                is_checkmate=game.is_checkmate(),
                is_stalemate=game.is_stalemate(),
                is_insufficient=game.is_insufficient_material(),
                winner=game.result(),
                game_pgn=str(pgn),
            )
            played_relationship = Relationship(self.training_session_node, "Played", game_node)
            tx = self.db.begin()
            tx.create(game_node)
            tx.create(played_relationship)
            self.db.commit(tx)
            

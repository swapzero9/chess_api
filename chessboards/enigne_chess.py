from logging import error
from py2neo import Graph, Node
from py2neo.matching import NE
from api.utils.logger import MyLogger
import api.utils.decorators as d
from fastapi import APIRouter
from api.engines.random_engine.computer import RandomComputer
from api.engines.minimax_engine.computer import MiniMaxComputer
from api.engines.montecarlo_engine.computer import MonteCarloComputer
from api.engines.ai_engine.computer import AiComputer
from api.engines.ai_engine.models.final.net import Net
from api.engines.stockfish_engine.computer import StockfishComputer
from api.engines.ai_engine_new.computer import AiComputer2
from api.engines.ai_engine_new.models.final.net import Net as Net2

import chess, os, chess.pgn, datetime, math
from py2neo import Graph, Node
from api.classes.chess_classes import EngineSparingInput, EngineSparingGame, ErrorDatabase


router = APIRouter()
module_logger = MyLogger(__name__)

@router.post("/sparing")
async def sparing(details: EngineSparingInput):

    p1 = select_player(details.player_white)
    p2 = select_player(details.player_black)

    if p1 != None and p2 != None:
        results = single_game(p1, p2)
        return EngineSparingGame(pgn=results)
    else: 
        return ErrorDatabase(error="i dunno")


def select_player(name): 
    try:
        if name == "minimax":
            return MiniMaxComputer()
        elif name == "random":
            return RandomComputer()
        elif name == "montecarlo":
            return MonteCarloComputer(100)
        elif name == "stockfish":
            return StockfishComputer("b", 200)
        elif name == "ai":
            try:
                return AiComputer(load_model=True, net=Net, model_name="final.pt")
            except Exception as ex:
                module_logger().exception(ex)
                return AiComputer(net=Net)
        elif name == "ai2":
            try:
                return AiComputer2(load_model=True, net=Net2, model_name="final.pt")
            except Exception as ex:
                module_logger().exception(ex)
                return AiComputer2(net=Net2)
        else:
            return None
    except Exception as ex:
        module_logger().exception(ex)

def single_game(p1, p2):
    game = chess.Board()
    pgn = chess.pgn.Game()
    pgn.headers["White"] = p1.__class__.__name__
    pgn.headers["Black"] = p2.__class__.__name__
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
                move = p1.think(game.fen())
                if move in legal_moves:
                    game.push(move)
                    whites_turn = False
                    is_legal = True

        # move if black
        else:
            while not is_legal:
                move = p2.think(game.fen())
                if move in legal_moves:
                    game.push(move)
                    whites_turn = True
                    is_legal = True
        if is_first_move:
            node = pgn.add_variation(move)
            is_first_move = False
        else:
            node = node.add_variation(move)

    wc = p1.__class__.__name__ if game.result() == "1-0" else (p2.__class__.__name__ if game.result() == "0-1" else "none")

    pgn.headers["Result"] = game.result()
    game_node = Node(
        "SparingGame",
        timestamp=datetime.datetime.today(),
        number_of_moves=math.ceil(len(list(pgn.mainline_moves())) / 2),
        is_checkmate=game.is_checkmate(),
        is_stalemate=game.is_stalemate(),
        is_insufficient=game.is_insufficient_material(),
        winner=game.result(),
        winner_c=wc,
        game_pgn=str(pgn),
    )

    db = Graph(os.environ["DB_URL"], auth=(os.environ["DB_ADMIN"], os.environ["DB_PASS"]))
    try:
        tx = db.begin()
        tx.create(game_node)
        db.commit(tx)
    except Exception as ex:
        module_logger().exception(ex)

    return str(pgn)
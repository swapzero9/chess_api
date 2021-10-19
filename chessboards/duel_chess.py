from fastapi import APIRouter
from api.classes.chess_classes import InputFen, OutputFen
from api.engines.random_engine.computer import RandomComputer
from api.engines.minimax_engine.computer import MiniMaxComputer
from api.engines.ai_engine.computer import AiComputer
from api.engines.ai_engine.models.architecture3.net import Net
from api.engines.stockfish_engine.computer import StockfishComputer
import api.utils.decorators as d

import chess

router = APIRouter()
engs = {}

# on app start init engines
@router.on_event("startup")
async def startup():
    print("init engines")
    engs["minimax"] = MiniMaxComputer()
    engs["random"] = RandomComputer()
    engs["stockfish"] = StockfishComputer("b", 200)
    engs["ai"] = AiComputer(net=Net)
    print("inited")

@d.timer_log
@router.post("/position")
async def position(details: InputFen):

    eng = engs[details.targetComputer]
    print(eng.__class__.__name__)
    # create board and make a move
    board = chess.Board(details.fen)
    move = eng.think(details.fen)
    board.push(move)

    # parse it for the client
    f = move.uci()[0:2]
    t = move.uci()[2:4]

    # parse promotion
    p = None
    if move.uci()[-1] in ["q", "n", "b", "r"]:
        p = move.uci()[-1]

    # create a return element
    ret = OutputFen(fen=board.fen(), moveFrom=f, moveTo=t, promotion=p)

    return ret

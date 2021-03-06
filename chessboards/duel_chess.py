from fastapi import APIRouter
from api.classes.chess_classes import InputFen, OutputFen, DuelChessGame
from api.engines.random_engine.computer import RandomComputer
from api.engines.minimax_engine.computer import MiniMaxComputer
from api.engines.ai_engine.computer import AiComputer
from api.engines.ai_engine.models.final.net import Net
from api.engines.stockfish_engine.computer import StockfishComputer
from api.engines.ai_engine_new.computer import AiComputer2
from api.engines.ai_engine_new.models.final.net import Net as Net2
from api.engines.montecarlo_engine.computer import MonteCarloComputer
import api.utils.decorators as d
from api.utils.logger import MyLogger
from py2neo import Graph, Node
import chess, chess.pgn, os, io

router = APIRouter()
engs = {}

module_logger = MyLogger(__name__)


# on app start init engines
@router.on_event("startup")
async def startup():
    print("init engines")
    engs["minimax"] = MiniMaxComputer()
    engs["montecarlo"] = MonteCarloComputer(100)
    engs["random"] = RandomComputer()
    engs["stockfish"] = StockfishComputer("b", 200)
    engs["ai"] = AiComputer(net=Net, load_model=True, model_name="final.pt")
    engs["ai2"] = AiComputer2(net=Net2, load_model=True, model_name="final.pt")
    print("inited")

@d.timer_log
@router.post("/position")
async def position(details: InputFen):

    eng = engs[details.targetComputer]
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

@d.timer_log
@router.post("/save_game")
async def save_game(details: DuelChessGame):
    
    p = io.StringIO(details.pgn)
    game = chess.pgn.read_game(p)
    
    game.headers["White"] = "Player"
    game.headers["Black"] = details.opponent

    board = chess.Board()
    for move in game.mainline_moves():
        board.push(move)
    
    game.headers["Result"] = board.result()

    n = Node("DuelGame", 
        timestamp=details.timestamp,
        opponent=details.opponent,
        result=board.result(),
        pgn=str(game)
    )

    db = Graph(os.environ["DB_URL"], auth=(os.environ["DB_ADMIN"], os.environ["DB_PASS"]))

    try:
        tx = db.begin()
        tx.create(n)
        db.commit(tx)
    except Exception as ex:
        module_logger().exception(ex)
    
    return {
        "hurray": "yaaay"
    }


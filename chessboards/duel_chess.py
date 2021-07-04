from fastapi import APIRouter
from api.classes.duel_chess_classes import InputFen
from api.classes.duel_chess_classes import OutputFen
from api.engines.random_engine.computer import RandomComputer
from api.engines.minimax_engine.computer import MiniMaxComputer

import chess

router = APIRouter()


@router.post("/position")
async def position(details: InputFen):

	eng = None
	if details.targetComputer == "minimax_engine":
		eng = MiniMaxComputer("b", 7)
		print("chosen minimax")
	else:
		eng = RandomComputer("b")
		print("chosen random")
	
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

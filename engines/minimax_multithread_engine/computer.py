from api.engines.template_computer import Computer
import chess, threading

class MiniMultiComputer(Computer):
	def __init__(self, side):
		super().__init__(side)


	def think(fen: str) -> chess.Move:
		pass
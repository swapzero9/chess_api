import chess, random, math, time


class Computer:

	def __init__(self, side):
		if side == 'b':
			self.white_player = False
		else:
			self.white_player = True

		self.piece_score = {
			1: 10,		# Pawn
			2: 29,		# Knight
			3: 30,		# Bishop
			4: 50,		# Rook
			5: 90,		# Queen
			6: 10000,	# King
		}
		pass

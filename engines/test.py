from api.engines.training import TrainingSession
from api.engines.random_engine.computer import RandomComputer
from api.engines.minimax_engine.computer import MiniMaxComputer
import chess, pprint
import time
from operator import itemgetter, attrgetter

if __name__ == "__main__":
	w = MiniMaxComputer('w', 6)
	b = MiniMaxComputer('w', 6)
	# t = TrainingSession(w,b, 1)
	# t.train()
	# print(w.white_player)
	t = time.time()	
	a = w.think('5k2/1Q4pp/p1pp4/3Np3/P3P3/3P2qb/1P3r2/R6K w - - 2 24')
	elapsed = time.time() - t
	print(elapsed)
	print(a)

	# board = chess.Board('8/3k4/8/8/8/8/5PPP/3r2K1 b - - 0 1')
	# l = list(board.legal_moves)
	# for i in l:
	# 	print(i.uci())

	# board = chess.Board('8/4k3/8/7q/8/8/3r1PPP/6K1 b - - 0 1')
	# legal_moves = list(board.legal_moves)
	# new = []
	# for move in legal_moves:
	# 	board.push(move)
	# 	temp = board.copy()
	# 	board.pop()
	# 	if board.is_capture(move) or temp.is_check():
	# 		new.append((move, True))
	# 	else:
	# 		new.append((move, False))
	
	# asd = sorted(new, key=itemgetter(1))
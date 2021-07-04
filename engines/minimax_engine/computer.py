from api.engines.random_engine.computer import Computer
import chess, math, time
from api.engines.template_computer import Computer
from operator import itemgetter
from numba import jit, cuda

class MiniMaxComputer(Computer):
	"""
	simple minimax engine, with alpha-beta pruning and transposition table
	"""
	def __init__(self, side, depth=None):
		if depth == None:
			self.depth = 5
		else:
			self.depth = depth
		
		self.transposition_table = {}
		super().__init__(side)

	def think(self, fen: str) -> chess.Move:
		self.transposition_table = {}
		board = chess.Board(fen)
		move, eval = self.minimax(board, self.depth, -math.inf, math.inf, self.white_player)
		return move

	def evaluate_position(self, board):		
		temp = chess.BaseBoard(board_fen=board.board_fen())
		score = 0
		for key in self.piece_score:
			score += (len(temp.pieces(key, True)) - len(temp.pieces(key, False))) * self.piece_score[key]
		
		score *= (board.turn if 1 else -1)
		self.transposition_table[board.fen()] = score
		return score 

	def minimax(self, board, depth, alpha, beta, white_player):
		if depth == 0 or board.is_game_over():

			# handler for when the game is over
			b = board.outcome()
			if b is not None:
				if b.winner == True:	
					# white won
					return None, self.piece_score[6]
				elif b.winner == False:
					# black won
					return None, -self.piece_score[6]
				else:
					# draw
					return None, 0

			# the depth limit has been reached
			else:
				return None, self.evaluate_position(board)
		
		best_move = None
		if white_player:
			max_eval = -math.inf
			
			# create legal moves for a given position
			legal_moves = list(board.legal_moves)
			for move in legal_moves:

				# try out every move 
				board.push(move)
				temp = board.copy()
				board.pop()

				# look up if already evaluated, if not recurrency
				if temp.fen() in self.transposition_table:
					eval = self.transposition_table[temp.fen()]
				else:
					ret_move, eval = self.minimax(temp, depth-1, alpha, beta, False)
				
				# alpha-beta pruning, and picking the best move
				if eval > max_eval:
					max_eval = eval
					best_move = move
				alpha = max(alpha, eval)
				if beta < eval:
					break
			return best_move, max_eval

		else:
			min_eval = math.inf

			# create legal moves for a given position
			legal_moves = list(board.legal_moves)
			for move in legal_moves:

				# try out every move 
				board.push(move)
				temp = board.copy()
				board.pop()

				# look up if already evaluated, if not recurrency
				if temp.fen() in self.transposition_table:
					eval = self.transposition_table[temp.fen()]
				else:
					ret_move, eval = self.minimax(temp, depth-1, alpha, beta, True)

				# alpha-beta pruning, and picking the best move
				if eval < min_eval:
					min_eval = eval
					best_move = move
				beta = min(beta, eval)
				if beta < eval:
					break
			return best_move, min_eval


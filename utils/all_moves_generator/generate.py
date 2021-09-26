import chess


def generate_file():

	# final output file with number
	output_file = open("all_moves.csv", "w+")
	m = 0
	output_file.write("move\n")

	# Queen moves
	base_fen_list = ["8", "8", "8", "8", "8", "8", "8", "8"]
	garbage = " w - - 0 1"
	for rank in range(8):
		for file in range(8):

			# create fen
			fen = base_fen_list.copy()
			first = "" if file == 0 else str(file)
			last = "" if file == 7 else str(7 - file)

			fen[rank] = f"{first}Q{last}"
			f = "/".join(fen) + garbage

			board = chess.Board(f)
			for move in list(board.legal_moves):
				output_file.write(f"{move.uci()}\n")
				m += 1

	# Knight moves
	base_fen_list = ["8", "8", "8", "8", "8", "8", "8", "8"]
	garbage = " w - - 0 1"
	for rank in range(8):
		for file in range(8):

			# create fen
			fen = base_fen_list.copy()
			first = "" if file == 0 else str(file)
			last = "" if file == 7 else str(7 - file)

			fen[rank] = f"{first}N{last}"
			f = "/".join(fen) + garbage

			board = chess.Board(f)
			for move in list(board.legal_moves):
				output_file.write(f"{move.uci()}\n")
				m += 1

	# promotions from 2-1 and from 7-8
	# 3 moves from each central square, 
	# 2 moves from the sides,
	# 4 promotion options
	temp = "8/PPPPPPPP/8/8/8/8/pppppppp/8 w - - 0 1"
	prom = chess.Board(temp).legal_moves

	# pawn pushes
	for move in list(prom):
		output_file.write(f"{move.uci()}\n")

	temp = "nnnnnnnn/PPPPPPPP/8/8/8/8/pppppppp/8 w - - 0 1"
	prom = chess.Board(temp).legal_moves

	# pawn takes
	for move in list(prom):
		output_file.write(f"{move.uci()}\n")

	# repeat for black side
	temp = "8/PPPPPPPP/8/8/8/8/pppppppp/8 b - - 0 1"
	prom = chess.Board(temp).legal_moves

	# pawn pushes
	for move in list(prom):
		output_file.write(f"{move.uci()}\n")

	temp = "8/PPPPPPPP/8/8/8/8/pppppppp/NNNNNNNN b - - 0 1"
	prom = chess.Board(temp).legal_moves

	# pawn takes
	for move in list(prom):
		output_file.write(f"{move.uci()}\n")

def test_fen(fen):

	all_moves = list()
	with open("all_moves.csv", "r") as f:
		all_moves_text = f.readlines()
		for m in all_moves_text:
			all_moves.append(m.strip())

	board = chess.Board(fen)
	legal_moves = list(board.legal_moves)
	i = 1
	for m in legal_moves:
		print(f"{i}: {m.uci()}, is in csv: {m.uci() in all_moves}")
		i += 1

if __name__ == "__main__":

	# generate all the possible moves in any position
	# final output size from ai nn
	generate_file()

	test_fen("8/1P6/8/8/8/8/8/8 w - - 0 1")
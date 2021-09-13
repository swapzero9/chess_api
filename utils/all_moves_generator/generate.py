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

	test_fen("rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
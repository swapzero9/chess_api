from dataclasses import dataclass
import chess, chess.syzygy, chess.pgn
import json, os


with chess.syzygy.open_tablebase("syzygy_tables") as tablebase:
    board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
    print(tablebase.probe_wdl(board))

    for move in board.legal_moves:
        board.push(move)
        print(f"{board.fen()}, {tablebase.probe_dtz(board)}")
        board.pop()

pgn = open("lichess_db.pgn")


@dataclass
class Node:
    fen: str
    amount: int
    prob: float

def test():
    base = "opening_db"
    db = dict()
    db["rnbqkbnr_pppppppp_8_8_8_8_PPPPPPPP_RNBQKBNRwKQkq-01"] = dict()

    i = 0
    while True:
        i += 1
        game = chess.pgn.read_game(pgn)
        print(i / 1000) if i % 1000 == 0 else print("", end="")

        if game is None:
            break
        
        if i > 8000000:
            break

        board = chess.Board()
        for move in game.mainline_moves():
            if board.fullmove_number >= 8:
                break

            fen = board.fen()
            normalised_fen = fen.replace("/", "_").replace(" ", "")
            board.push(move)

            if normalised_fen in db:
                if move.uci() in db[normalised_fen]:
                    db[normalised_fen][move.uci()] += 1
                else:
                    db[normalised_fen][move.uci()] = 1
            else:
                db[normalised_fen] = dict()
                db[normalised_fen][move.uci()] = 1

            # if f"{normalised_fen}.json" in os.listdir(f"./{base}"):
            #     with open(f"./{base}/{normalised_fen}.json", mode="r") as j:
            #         data = json.load(j)

            #     if move.uci() in data["children"]:
            #         data["children"][move.uci()] += 1
            #     else:
            #         data["children"][move.uci()] = 1

            #     with open(f"./{base}/{normalised_fen}.json", mode="w") as j:
            #         json.dump(data, j)
            # else:
            #     temp = dict()
            #     temp["children"] = dict()
            #     temp["children"][move.uci()] = 1
            #     with open(f"./{base}/{normalised_fen}.json", mode="w") as j:
            #         json.dump(temp, j)
    with open("test.json", mode="w") as j:
        json.dump(db, j) 

if __name__ == "__main__":
    test()

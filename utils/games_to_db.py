import sys, os, os.path
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from datetime import datetime
from typing import List
import chess, chess.pgn

def insert_to_db(db, elements):
    try:
        tx = db.begin()
        for el in elements:
            tx.create(el)
        db.commit(tx)
    except Exception as ex:
        print(f"!!!!!!!!!!!!!!!!\n{ex}")
    pass

def create_games(db, parent_node, path):
    files = os.listdir(path)
    game_nodes = list()
    for index, file in enumerate(files):
        with open(f"{path}/{file}", "r") as f:
            pgn = chess.pgn.read_game(f)

        if pgn is None:
            continue
        if len(sys.argv) == 4:
            pgn.headers["White"] = str(sys.argv[3])
            pgn.headers["Black"] = str(sys.argv[3])
        
        timestamp = file.split(".")[0]
        t = datetime.strptime(timestamp, "%d%m%Y_%H%M%S")

        node = Node("GameNode",
            game_number=(index + 1),
            game_pgn=str(pgn),
            timestamp=t,
            winner=(pgn.headers["Result"]),
            winner_c=pgn.headers["White"] if pgn.headers["Result"] == "1-0" else (pgn.headers["Black"] if pgn.headers["Result"] == "0-1" else None),
            p1=pgn.headers["White"],
            p2=pgn.headers["Black"]
        )
        rel = Relationship(parent_node, "Played", node)
        game_nodes.append(node)
        game_nodes.append(rel)

    insert_to_db(db, game_nodes)

if __name__ == "__main__":
    
    # argv[1] => path
    # argv[2] => {t/v}_name
    # argv[3]

    # establish db connection
    db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))

    if os.path.exists(sys.argv[1]):
        # num of games
        l = len(os.listdir(sys.argv[1]))
        args = sys.argv[2].split("-")
        if args[0] == "t":
            # create training session games
            training_node = Node(
                "TrainingNode",
                name=args[1],
                timestamp=datetime.today(),
                num_games=l
            )
            insert_to_db(db, [training_node])
            create_games(db, training_node, sys.argv[1])


        elif args[0] == "v":
            # create validation session games
            validation_node = Node(
                "ValidationNode",
                name=args[1],
                timestamp=datetime.today(),
                num_games=l
            )
            create_games(db, validation_node, sys.argv[1])

    else:
        print("GAMES PATH DOES NOT EXISTS!!!!!!!!!!")
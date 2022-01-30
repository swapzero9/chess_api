from fastapi import APIRouter
from api.classes.chess_classes import ChessGame, ErrorDatabase, TrainingNodeList, SelectTrainingNode
from py2neo import Graph, NodeMatcher, RelationshipMatcher, Node
from api.utils.logger import MyLogger
import os

module_logger = MyLogger(__name__)

router = APIRouter()

@router.post("/get_game")
def last_game(details: SelectTrainingNode):

    try: 
        db = Graph(os.environ["DB_URL"], auth=(
            os.environ["DB_ADMIN"],
            os.environ["DB_PASS"]
        ))

        # game = db.run(f"MATCH (n:TrainingNode {{name: \"{details.node_name}\"}})-->(t:TrainingIteration)-->(g:Game {{}}) RETURN g.game_pgn ORDER BY t.timestamp DESC LIMIT 1").to_series().to_list()
        
        if details.type == "training":
            game = db.run(f'MATCH (n:TrainingNode {{name: "{details.node_name}"}})-[p:Played]->(g:GameNode {{game_number: {details.game_number}}}) return g.game_pgn LIMIT 1').to_series().to_list()
            if len(game) == 0:
                return ErrorDatabase(error="no games found")
            else:
                return ChessGame(
                    pgn=game[0],
                    iteration=1,
                    engine_name=details.node_name
                )
        elif details.type == "validation":
            game = db.run(f'MATCH (n:ValidationNode {{name: "{details.node_name}"}})-[p:Played]->(g:GameNode {{game_number: {details.game_number}}}) return g.game_pgn LIMIT 1').to_series().to_list()
            if len(game) == 0:
                return ErrorDatabase(error="no games found")
            else:
                return ChessGame(
                    pgn=game[0],
                    iteration=1,
                    engine_name=details.node_name
                )

    except Exception as ex:
        module_logger().exception(ex)
        return ErrorDatabase(error="someerror")


@router.get("/last_validation_game")
def last_validation_game():
    
    try: 
        db = Graph(os.environ["DB_URL"], auth=(
            os.environ["DB_ADMIN"],
            os.environ["DB_PASS"]
        ))

        rel_match = RelationshipMatcher(db)
        node_match = NodeMatcher(db)
        validation_sess = node_match.match("ValidationSession").order_by("_.date_start desc").first()
        
        last_iter = rel_match.match((validation_sess, None), "Iteration").first().nodes[-1]
        last_game = rel_match.match((last_iter, None), "Played").first().nodes[-1]
        pgn = last_game["game_pgn"]

        training_session = rel_match.match((None, validation_sess), "ValidatingTraining").first().nodes[0]
        name = training_session["name"]
        return ChessGame(
            pgn=pgn,
            iteration=1,
            engine_name=name
        )
    except Exception as ex:
        module_logger().exception(ex)
        return ErrorDatabase(error="someerror")

@router.get("/training_nodes")
def distinct_node_names():

    try:
        db = Graph(os.environ["DB_URL"], auth=(
            os.environ["DB_ADMIN"],
            os.environ["DB_PASS"]
        ))

        query = db.run("MATCH (n:TrainingNode) return n.name as name, n.num_games as games").to_data_frame()
        ret = list()
        for i in range(len(query)):
            rec = dict(query.iloc[i])
            ret.append({
                "name": rec["name"],
                "type": "training",
                "game_number": int(rec["games"])
            })
        query = db.run("MATCH (n:ValidationNode) return n.name as name, n.num_games as games").to_data_frame()
        for i in range(len(query)):
            rec = dict(query.iloc[i])
            ret.append({
                "name": rec["name"],
                "type": "validation",
                "game_number": int(rec["games"])
            })
        return TrainingNodeList(node_list=ret)

    except Exception as ex:
        module_logger().exception(ex)
        return ErrorDatabase(error="someerror")

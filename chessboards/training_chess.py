from fastapi import APIRouter
from api.classes.chess_classes import ChessGame, ErrorDatabase
from py2neo import Graph, NodeMatcher, RelationshipMatcher, Node
import os


router = APIRouter()

@router.get("/last_training_game")
def last_game():

    db = Graph(os.environ["DB_URL"], auth=(
        os.environ["DB_ADMIN"],
        os.environ["DB_PASS"]
    ))

    rel_match = RelationshipMatcher(db)
    node_match = NodeMatcher(db)
    last_iteration_node = node_match.match("TrainingIteration").order_by("_.timestamp desc").first()
    
    last_game = rel_match.match((last_iteration_node, None), "Played").first().nodes[-1]
    pgn = last_game["game_pgn"]
    training_session = rel_match.match((None, last_iteration_node), "GameIteration").first().nodes[0]
    name = training_session["name"]
    return ChessGame(
        pgn=pgn,
        iteration=1,
        engine_name=name
    )


@router.get("/last_validation_game")
def last_validation_game():
    
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

@router.get("/training_node")
def distinct_node_names():

    pass
from py2neo import Graph, NodeMatcher, RelationshipMatcher, Node
from api.utils.logger import MyLogger
import os
from fastapi import APIRouter
from api.classes.chess_classes import ChessGamesStatistics

module_logger = MyLogger(__name__)

router = APIRouter()

@router.get("/stat_data")
def stat_data():

    # duel data
    db = Graph(os.environ["DB_URL"], auth=(os.environ["DB_ADMIN"], os.environ["DB_PASS"]))
    q = db.run("Match(n:DuelGame) return n.result as result, n.opponent as opponent").to_data_frame()
    duel_ret = dict()
    for i in range(len(q)):
        rec = dict(q.iloc[i])
        if rec["opponent"] not in duel_ret:
            duel_ret[rec["opponent"]] = [
                1 if rec["result"] == "0-1" else 0,
                1 if rec["result"] == "1-0" else 0,
                1 if rec["result"] == "1/2-1/2" else 0,
            ]
        else:
            ind = 0 if rec["result"] == "0-1" else (1 if rec["result"] == "1-0" else 2)
            duel_ret[rec["opponent"]][ind] += 1

    # training data
    q = db.run("Match(n:TrainingNode)-[p:Played]->(g:GameNode) return n.name as tsession, g.winner as result").to_data_frame()
    training_ret = dict()
    for i in range(len(q)):
        rec = dict(q.iloc[i])
        if rec["tsession"] not in training_ret:
            training_ret[rec["tsession"]] = [
                1 if rec["result"] == "1-0" else 0,
                1 if rec["result"] == "0-1" else 0,
                1 if rec["result"] == "1/2-1/2" else 0,
            ]
        else:
            ind = 0 if rec["result"] == "1-0" else (1 if rec["result"] == "0-1" else 2)
            training_ret[rec["tsession"]][ind] += 1

    ret = ChessGamesStatistics(
        duel_statistics=duel_ret,
        training_statistics=training_ret,
        validation_statistics={},
    )
    return ret

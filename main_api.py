from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os 
from api.chessboards import puzzles_chess
from api.chessboards import duel_chess
from api.chessboards import training_chess
from api.chessboards import enigne_chess
from api.chessboards import statistics
from py2neo import Graph
from api.classes.chess_classes import DBSummary

app = FastAPI()
if "PAGE_URL" not in os.environ:
    origins = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
else:
    origins = [
        os.environ["PAGE_URL"]
    ]

@app.on_event("startup")
def setup_environment():
    if "DB_URL" not in os.environ:
        os.environ["DB_URL"] = "bolt://localhost:7687"
    if "DB_ADMIN" not in os.environ:
        os.environ["DB_ADMIN"] = "neo4j"
    if "DB_PASS" not in os.environ:
        os.environ["DB_PASS"] = "s3cr3t"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    puzzles_chess.router,
    prefix="/puzzles",
    tags=["puzzles"],
)

app.include_router(
    training_chess.router,
    prefix="/training",
    tags=["training"],
)

app.include_router(
    duel_chess.router,
    prefix="/duel",
    tags=["duel"],
)

app.include_router(
    enigne_chess.router,
    prefix="/engine",
    tags=["engine"],
)

app.include_router(
    statistics.router,
    prefix="/statistics",
    tags=["statistics"]
)


@app.get("/")
async def read_root():
    return {"yes": "boi"}

@app.get("/summary")
async def summary():
    # q1 MATCH (n:ValidationNode)-[p:Played]->(g:GameNode) RETURN COUNT(g)
    # q2 MATCH (n:TrainingNode)-[p:Played]->(g:GameNode) RETURN COUNT(g)
    # q3 MATCH (n:DuelGame) RETURN COUNT(n)

    db = Graph(os.environ["DB_URL"], auth=(os.environ["DB_ADMIN"], os.environ["DB_PASS"]))

    r1 = db.run("MATCH (n:ValidationNode)-[p:Played]->(g:GameNode) RETURN COUNT(g)").to_series()[0]
    r2 = db.run("MATCH (n:TrainingNode)-[p:Played]->(g:GameNode) RETURN COUNT(g)").to_series()[0]
    r3 = db.run("MATCH (n:DuelGame) RETURN COUNT(n)").to_series()[0]

    return DBSummary(
        validation_games=r1,
        training_games=r2,
        duel_games=r3
    )

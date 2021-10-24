from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os 
from api.chessboards import puzzles_chess
from api.chessboards import duel_chess
from api.chessboards import training_chess
from api.chessboards import enigne_chess

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://192.168.0.129:8080",
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



@app.get("/")
async def read_root():
    return {"yes": "boi"}

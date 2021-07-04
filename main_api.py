from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .chessboards import puzzles_chess
from .chessboards import duel_chess

app = FastAPI()

origins = [
	"http://localhost:8080",
	"http://192.168.0.129:8080",
]

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
	duel_chess.router,
	prefix="/duel",
	tags=["duel"],
)

@app.get("/")
async def read_root():
	return {"yes": "boi"}

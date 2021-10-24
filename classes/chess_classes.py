from pydantic import BaseModel
from typing import Optional
import time


class InputFen(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    fen: str
    moveFrom: str
    moveTo: str
    targetComputer: str


class OutputFen(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    fen: str
    moveFrom: str
    moveTo: str
    promotion: Optional[str] = None


class ChessGame(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    pgn: str
    iteration: int
    engine_name: str

class ErrorDatabase(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    error: str

class DuelChessGame(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    pgn: str
    opponent: str

class EngineSparingInput(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    player_white: str
    player_black: str

class EngineSparingGame(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    pgn: str

class TrainingNodeList(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    node_list: list

class SelectTrainingNode(BaseModel):
    timestamp: Optional[int] = round(time.time() * 1000)
    node_name: str
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
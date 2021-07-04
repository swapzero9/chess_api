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

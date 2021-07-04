from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

import time

class InputFen(BaseModel):
	gameId: str
	timestamp: Optional[int] = round(time.time() * 1000)
	fen: str
	moveFrom: str
	moveTo: str

router = APIRouter()

@router.post("/")
async def test(whatever: InputFen):
	return whatever

@router.post("/position")
async def legalmove(details: InputFen):
	return details
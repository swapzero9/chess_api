import chess, random, math, time
from api.engines.template_computer import Computer


class RandomComputer(Computer):
    def __init__(self, side):
        """
        takes an engine argument as a variable which is pytorch nn class
        side argument is for verification which side is Computer playing
        timeout just for not calculating deep enough
        """
        self.timeout = 0.5
        super().__init__(side)
        pass

    def think(self, fen: str) -> chess.Move:
        """
        takes a fen element, uses minimax and alpha beta search to predict the best movement
        calls pytorch nn class to predict evaluation of the position
        """

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        for i in legal_moves:
            temp = board.root()
            temp.push(i)
            if temp.is_checkmate():
                time.sleep(self.timeout)
                return i
        r = math.floor(random.random() * len(legal_moves))

        move = legal_moves[r]

        # time.sleep(self.timeout)
        return move

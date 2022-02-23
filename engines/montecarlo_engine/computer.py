
from api.engines.template_computer import Computer
from dataclasses import dataclass
import chess, random
import numpy as np
from time import perf_counter
from api.utils.decorators import timer
from pprint import pprint

class MonteCarloComputer(Computer):

    def __init__(self, iterations):
        super().__init__("whatever")

        self.iterations = iterations
        self.mcts = MonteCarloComputer.MonteCarloSearch()

    def think(self, fen:str) -> chess.Move:
        b = chess.Board(fen)
        try:
            move = self.mcts(fen, self.iterations)
            r = chess.Move.from_uci(move)
            if r in b.legal_moves:
                return r
            else:
                return random.choice(list(b.legal_moves))
        except Exception:
            print('crit')
            return random.choice(list(b.legal_moves))
    class MonteCarloSearch:
        @dataclass
        class Node:
            p: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # position of 
            t: bool = True
            v: float = 0 # winning score of current node
            n: int = 0 # number of times parent node has been visited
            parent = None
            children = None
            def eval (self):
                return (self.v / (self.n + 1)) + np.sqrt(2) * np.sqrt(np.log(self.parent.n + 1) / (self.n + 1))
            #eval = lambda x,y,z: x /10 +  np.sqrt(np.log(y+1) / (z+1))
        def __init__(self) -> None:
            pass

        def __call__(self, start_fen, iterations = 2000):
            curr_node = self.Node(start_fen)
            curr_node.children = list()
            start_board = chess.Board(start_fen)
            curr_node.t = start_board.turn
            for move in start_board.legal_moves:
                start_board.push(move)
                child = self.Node(p=start_board.fen())
                child.t = start_board.turn
                child.parent = curr_node
                curr_node.children.append(child)
                start_board.pop()

            selected = None
            while iterations > 0:
                # print(f"Iteration : {iterations} \r",end="" )
                curr_node.children.sort(key=lambda x: x.eval(), reverse=start_board.turn) 
                selected = curr_node.children[0]

                ex_child = self.expand(selected)
                rol_node = chess.Board(selected.p)
                leaf = self.rollout(rol_node, ex_child)
                reward = self.get_normalised_outcome(leaf)
                curr_node = self.backpropagation(leaf, reward)
                iterations -= 1

            curr_node.children.sort(key=lambda x: x.eval(), reverse=start_board.turn) 
            best = curr_node.children[0].p

            for move in start_board.legal_moves:
                start_board.push(move)
                if start_board.fen() == best:
                    return move.uci()
                start_board.pop()

        def expand(self, curr_node):
            if curr_node.children is None:
                curr_node.children = list()

            if len(curr_node.children) == 0:
                return curr_node
            else:
                curr_node.children.sort(key=lambda x: x.eval(), reverse=curr_node.t) 
                selected = curr_node.children[0]
                return self.expand(selected)

        def rollout(self, rol_node, curr_node):
            if rol_node.is_game_over():
                return curr_node
            else:
                if curr_node.children is None:
                    curr_node.children = list()

                moves = list(rol_node.legal_moves)
                final_move = random.choice(moves)
                for move in moves:
                    rol_node.push(move)
                    child = self.Node(p=rol_node.fen())
                    child.t = rol_node.turn
                    child.parent = curr_node
                    curr_node.children.append(child)
                    rol_node.pop()

                rol_node.push(final_move)
                curr_node = curr_node.children[moves.index(final_move)]
                return self.rollout(rol_node, curr_node)

        def backpropagation(self, curr_node, reward):
            curr_node.n += 1
            curr_node.v += reward
            while curr_node.parent is not None:
                curr_node.n += 1
                curr_node.v += reward
                curr_node = curr_node.parent
            return curr_node

        def get_normalised_outcome(self, node):
            b = chess.Board(node.p)
            if not b.is_game_over():
                return 0
            o = b.outcome().winner
            if o is None:
                return 0
            elif o:
                return 1
            else: 
                return -1

if __name__ == "__main__":
    m = MonteCarloComputer(50)
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))

    t = perf_counter()
    print(m.think(board.fen()))
    print(f"Elapsed: {perf_counter() - t}")
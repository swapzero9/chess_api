from aaa.monte_carlo_algorithm import MonteCarloSearch
from api.engines.template_computer import Computer
from dataclasses import dataclass
import chess, random
import numpy as np

class MonteCarloComputer(Computer):


    def __init__(self, iterations):
        super().__init__("whatever")

        self.iterations = iterations
        self.mcts = MonteCarloSearch()

    def think(self, fen:str) -> chess.Move:

        move = self.mcts(fen, self.iterations)
        r = chess.Move.from_uci(move)
        return r

    class MonteCarloSearch:

        @dataclass
        class Node:
            p: chess.Board = chess.Board() # position of 
            v: float = 0 # winning score of current node
            n: int = 0 # number of times parent node has been visited
            c: int = 0 # number of times child node has been visited
            parent = None
            children = None
            def eval (self):
                return self.v + 2 * np.sqrt(np.log(self.n + 1 + np.finfo(np.float32).eps) / (self.c + np.finfo(np.float32).eps))

        def __init__(self) -> None:
            pass

        def __call__(self, start, iterations = 2000):
            
            curr_node = start if isinstance(start, self.Node) else self.Node(chess.Board(start))
            curr_node.children = list()
            for move in curr_node.p.legal_moves:
                child = self.Node(chess.Board(curr_node.p.fen()))
                child.p.push(move)
                child.parent = curr_node
                curr_node.children.append(child)

            selected = None
            while iterations > 0:
                if curr_node.p.turn: # white player, looking for max
                    curr_node.children.sort(key=lambda x: x.eval(), reverse=True) 
                    selected = curr_node.children[0]

                    ex_child = self.expand(selected)
                    self.rollout(ex_child)
                    curr_node = self.backpropagation(self.leaf_node, self.reward)

                else: #black player looking for min
                    curr_node.children.sort(key=lambda x: x.eval(), reverse=False) 
                    selected = curr_node.children[0]

                    ex_child = self.expand(selected)
                    self.rollout(ex_child)
                    curr_node = self.backpropagation(self.leaf_node, self.reward)
                iterations -= 1

            best = None
            if curr_node.p.turn:
                curr_node.children.sort(key=lambda x: x.eval(), reverse=True) 
                best = curr_node.children[0].p.fen()
                
            else:
                curr_node.children.sort(key=lambda x: x.eval(), reverse=False) 
                best = curr_node.children[0].p.fen()

            for move in curr_node.p.legal_moves:
                curr_node.p.push(move)
                if curr_node.p.fen() == best:
                    return move.uci()
                curr_node.p.pop()

        def expand(self, curr_node):
            if curr_node.children is None:
                curr_node.children = list()

            if len(curr_node.children) == 0:
                return curr_node
            else:
                if curr_node.p.turn:
                    curr_node.children.sort(key=lambda x: x.eval()) 
                    selected = curr_node.children[0]
                else: #black player looking for min
                    curr_node.children.sort(key=lambda x: x.eval(), reverse=True) 
                    selected = curr_node.children[0]

                self.expand(selected)

        def rollout(self, curr_node):
            if curr_node is None:
                return
            if curr_node.p.is_game_over():
                o = curr_node.p.outcome().winner
                self.leaf_node = curr_node
                if o:
                    self.reward = 1
                elif not o:
                    self.reward = -1
                else: 
                    self.reward = 0
                return
            else:
                if curr_node.children is None:
                    curr_node.children = list()

                for move in curr_node.p.legal_moves:
                    child = self.Node(chess.Board(curr_node.p.fen()))
                    child.p.push(move)
                    child.parent = curr_node
                    curr_node.children.append(child)

                r = random.choice(curr_node.children)
                self.rollout(r)

        def backpropagation(self, curr_node, reward):
            curr_node.c += 1
            curr_node.v += reward
            while curr_node.parent is not None:
                curr_node.n += 1
                curr_node = curr_node.parent
            return curr_node

if __name__ == "__main__":
    m = MonteCarloComputer(2000)
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    from time import perf_counter

    t = perf_counter()
    print(m.think(board.fen()))
    print(f"Elapsed: {perf_counter() - t}")
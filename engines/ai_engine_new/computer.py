# from api.engines.template_computer import Computer
import torch
import chess, chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import os
from dataclasses import dataclass
import random
from torch.utils.data import Dataset, DataLoader
from typing import List
from pprint import pprint

@dataclass
class Node:
    p: chess.Board = chess.Board() # position of eval node
    prior: float = 0
    v: float = 0 # winning score of current node
    n: int = 0 # number of times parent node has been visited
    c: int = 0 # number of times child node has been visited
    parent = None
    children = None
    def eval (self):
        return self.v + 2 * self.prior * np.sqrt(np.log(self.n + 1 + np.finfo(np.float32).eps) / (self.c + np.finfo(np.float32).eps))

class AiComputer2:
    """
    Different from previous
    Instead of outputing matrix of size (moves x 1968)
    Outputs (moves x 1)
    where number represents how good the position is
    And maybe second net to traverse the position deeper
    """

    CUDA_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, *, load_model=False, model_name="model.pt", net:nn.Module, seed:int = random.randint(0, 100)):
        self.model = net()
        self.model.to(self.CUDA_DEVICE)
        # self.model_summary()

        # self.model_path = "./api/engines/ai_engine_new/models"
        self.model_path = "./models"
        self.model_name = model_name
        self.seed = seed
        random.seed(self.seed)
        self.__thinker = AiComputer2.MonteCarloSearch()

        if load_model:
            m = os.listdir(self.model_path)
            if self.model_name in m:
                self.model.load_state_dict(torch.load(f"{self.model_path}/{self.model_name}"))

        self.transform = AiComputer2.TransformToTensor()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_criterion = nn.MSELoss()

    def save_model(self, name=None):
        if self.model is not None:
            if name is not None:
                torch.save(self.model.state_dict(), f"{self.model_path}/{name}")
            else:
                torch.save(self.model.state_dict(), f"{self.model_path}/{self.model_name}")

    def model_summary(self):
        model_stats = summary(self.model, [(1,8,8), (1,1,21)])
        if model_stats is not None:
            print(model_stats)

    def think(self, fen):
        move = self.__thinker(self, fen, iterations=(50 + self.seed))
        temp = chess.Board(fen)
        m = chess.Move.from_uci(move)
        if m in temp.legal_moves:
            return m
        else:
            return list(temp.legal_moves)[0]

    def learn(self):
        pass

    def create_moveset(self, arg):
        return self.ChessMoveset(arg)

    def predict_probabilities(self, nodelist:List[Node]):
        d = self.create_moveset(nodelist)
        dl = DataLoader(d, batch_size=len(d))
        positions = next(iter(dl))
        with torch.no_grad():
            ret:torch.Tensor = self.model(positions)
        for index, node in enumerate(nodelist):
            node.prior = ret[index].item()
        return nodelist


    class TransformToTensor(object):
        """
        takes a fen input or a list and creates tensorlist
        """

        def __init__(self):
            pass
        
        def __call__(self, inp):
            assert isinstance(inp, list) or isinstance(inp, str)

            if isinstance(inp, list):
                t = list()
                for i in range(len(inp)):
                    t.append(AiComputer2.TransformToTensor.handle_single_fen(inp[i]))
                return t
            else:
                t = AiComputer2.TransformToTensor.handle_single_fen(inp)
                return t
        
        @staticmethod
        def handle_single_fen(fen_raw: str):

            temp = fen_raw.split(" ") 
            fen = temp.pop(0)
            temp.pop() # pop last element off of list (move number)
            castle = AiComputer2.castle_move(fen_raw)
            en = AiComputer2.en_passant(fen_raw)
            desc = torch.cat((castle, en), dim=1)
            desc = desc.to(AiComputer2.CUDA_DEVICE)
            
            ranks = fen.split("/")
            i = 0 
            rank_tensor = torch.zeros(
                8, 8, dtype=torch.float32, 
                device=AiComputer2.CUDA_DEVICE
            )
            for rank in ranks:
                j = 0
                for letter in rank:
                    if letter.isnumeric():
                        j += int(letter)
                    else:
                        piece = AiComputer2.piece_2_number(letter)
                        rank_tensor[i, j] = piece
                        j += 1
                i += 1
            rank_tensor.unsqueeze_(0)
            return rank_tensor, desc, fen_raw

    class MonteCarloSearch:

        def __init__(self) -> None:
            pass

        def __call__(self, model, start, iterations = 2000):
            self.model:AiComputer2 = model
            curr_node:Node = start if isinstance(start, Node) else Node(chess.Board(start))
            curr_node.children = list()
            for move in curr_node.p.legal_moves:
                child = Node(chess.Board(curr_node.p.fen()))
                child.p.push(move)
                child.parent = curr_node
                curr_node.children.append(child)
            curr_node.children = self.model.predict_probabilities(curr_node.children)

            selected = None
            while iterations > 0:
                if curr_node.p.turn: # white player, looking for max
                    curr_node.children.sort(key=lambda x: x.eval(), reverse=True) 
                    selected = curr_node.children[0]

                    self.expand(selected)
                    max_val = self.predict_prior(self.expanded_node)
                    curr_node = self.backpropagation(self.expanded_node, max_val)

                else: #black player looking for min
                    curr_node.children.sort(key=lambda x: x.eval(), reverse=False) 
                    selected = curr_node.children[0]

                    self.expand(selected)
                    max_val = self.predict_prior(self.expanded_node)
                    curr_node = self.backpropagation(self.expanded_node, max_val)
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
                self.expanded_node = curr_node
            else:
                if curr_node.p.turn:
                    curr_node.children.sort(key=lambda x: x.eval()) 
                    selected = curr_node.children[0]
                else: #black player looking for min
                    curr_node.children.sort(key=lambda x: x.eval(), reverse=True) 
                    selected = curr_node.children[0]
                self.expand(selected)

        def predict_prior(self, node:Node):
            # curr_node.children = self.model.predict_probabilities(curr_node.children)
            if node.p.is_game_over():
                print("dupa")
                return 100 if node.parent.p.turn else (-100 if not node.parent.p.turn else 0)
            
            if node.children is None:
                node.children = list()

            if len(node.children) == 0:
                for move in node.p.legal_moves:
                    child = Node(chess.Board(node.p.fen()))
                    child.p.push(move)
                    child.parent = node
                    node.children.append(child)
            node.children = self.model.predict_probabilities(node.children)
            node.children.sort(key=lambda x: x.eval(), reverse=True)
            return node.children[0].eval()

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
                    child = Node(chess.Board(curr_node.p.fen()))
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

    class ChessMoveset(Dataset):
        def __init__(self, nodelist: List[Node]) -> None:
            self.tsfm = AiComputer2.TransformToTensor()
            self.dataset = list()
            for node in nodelist:
                self.dataset.append(self.tsfm(node.p.fen()))

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return self.dataset[index]


    """
    #######################################
        TRANSFORMATIONS AND STUFF
    #######################################
    """
    PIECES_NUMBERS = {
        "p": 1,
        "r": 2,
        "n": 3,
        "b": 4,
        "q": 5,
        "k": 6,
        "P": 7,
        "R": 8,
        "N": 9,
        "B": 10,
        "Q": 11,
        "K": 12
    }

    PIECES_SYMBOLS = {
        1: "p",
        2: "r",
        3: "n",
        4: "b",
        5: "q",
        6: "k",
        7: "P",
        8: "R",
        9: "N",
        10: "B",
        11: "Q",
        12: "K",
    }

    @staticmethod
    def piece_2_number(letter):
        return AiComputer2.PIECES_NUMBERS[letter]

    @staticmethod
    def number_2_piece(number):
        if number not in AiComputer2.PIECES_SYMBOLS:
            return None
        else:
            return AiComputer2.PIECES_SYMBOLS[number]

    @staticmethod
    def fen_2_tensor(fen):
        if len(fen.split(" ")) > 1:
            fen = fen.split(" ")[0]
        ranks = fen.split("/")
        i = 0
        rank_tensor = torch.zeros(
            8, 8, dtype=torch.float32, device=AiComputer2.CUDA_DEVICE
        )
        for rank in ranks:
            j = 0
            for letter in rank:
                if letter.isnumeric():
                    j += int(letter)
                else:
                    piece = AiComputer2.piece_2_number(letter)
                    rank_tensor[i, j] = piece
                    j += 1
            i += 1
        return rank_tensor

    @staticmethod
    def castle_move(fen):
        castles = fen.split(" ")[2]
        ret = torch.zeros((1, 5))
        if "K" in castles:
            ret[0][0] = 1.0
        if "Q" in castles:
            ret[0][1] = 1.0
        if "k" in castles:
            ret[0][2] = 1.0
        if "q" in castles:
            ret[0][3] = 1.0
        
        # move number
        move_number = fen.split(" ")[5]
        ret[0][4] = int(move_number) / 500 # normalization for nn
        return ret

    @staticmethod
    def en_passant(fen):
        ret = torch.zeros((1, 16))
        en = fen.split(" ")[3]
        if en == "-":
            return ret
        lol = "abcdefgh"
        i = 0
        for letter in lol:
            if f"{letter}3" == en:
                ret[0][i] = 1.0
                break
            i += 1
            if f"{letter}6" == en:
                ret[0][i] = 1.0
                break
            i += 1
        return ret

if __name__ == "__main__":

    # fen = "r1bqkb1r/pp1ppppp/2n2n2/2p5/5P2/4PN2/PPPPB1PP/RNBQK2R b KQkq - 4 4"

    # board, description = eng.transform(fen)
    # pprint(board.shape)
    # pprint(description.shape)
    # # eng.model(ten, desc)

    print("here")
    ########################
    # monte carlo testing
    
    from api.engines.ai_engine_new.models.architecture1.net import Net

    eng = AiComputer2(net=Net)

    board = chess.Board("5rk1/pp4p1/n5N1/5q1Q/2pP1P2/6P1/1P3P1P/6K1 w - - 9 36")
    eng.think(board.fen())

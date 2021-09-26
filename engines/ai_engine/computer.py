from numpy.core.fromnumeric import argmax
from pprint import pprint
from torch._C import device
import torch.nn as nn
import torch
import torch.nn.functional as F
from api.engines.template_computer import Computer
from torch.utils.data import Dataset, DataLoader
import chess, chess.pgn
import pandas as pd
import os
from api.utils.castling_move_number.castle import castle_move
from api.utils.en_passant_generator.generate import en_passant

import io
from py2neo import Graph, NodeMatcher, RelationshipMatcher


class AiComputer(Computer):

    # global cuda_devie
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.__name__ = "AiComputer"
        self.model = AiComputer.Net()
        self.model.to(self.cuda_device)
        self.tsfm = AiComputer.TransformToTensor()
        self.moves = pd.read_csv("./api/utils/all_moves_generator/all_moves.csv")
        self.dataset = None
        self.model_path = "./api/engines/ai_engine/models"

        m = os.listdir(self.model_path)
        if "model.pt" in m:
            self.model.load_state_dict(torch.load(f"{self.model_path}/model.pt"))

    def save_model(self):
        if self.model is not None:
            torch.save(self.model.state_dict(), f"{self.model_path}/model.pt")

    def think(self, fen: str) -> chess.Move:

        temp = self.predict_single(fen)
        m = chess.Move.from_uci(temp)
        legal = chess.Board(fen).legal_moves
        if m in legal:
            return m

        return list(legal)[0]

    def predict_single(self, fen):

        with torch.no_grad():
            ten, garb = self.tsfm(fen)
            ten.unsqueeze_(0)
            ret = self.model(ten, garb)

        cp = self.moves.copy()
        cp["prediction"] = ret[0,:].tolist()

        legal = chess.Board(fen).legal_moves
        uci = list()
        for move in legal:
            uci.append(move.uci())

        cp["legal"] = cp["move"]
        cp["legal"] = cp["legal"].isin(uci)
        cp = cp.sort_values(by=[f"legal", f"prediction"], ascending=[False, False])
        # cp = cp.sort_values(by=f"prediction", ascending=False)
        return cp.iloc[0]["move"]

    def predict_full(self, inp):

        # multiple moves out of transform
        fens = inp[0]
        tens = inp[1][0]
        desc = inp[1][1]
        desc.squeeze_(0)
        with torch.no_grad():
            ret = self.model(tens, desc)
            batch_size = list(ret.size())[0]
            cp = self.moves.copy()
            for b in range(batch_size):
                a = ret[b, :]
                
                cp[f"prediction_{b}"] = a.tolist()
                # cp = cp.sort_values(by="prediction", ascending=True)
                
                # legal_moves
                legal = chess.Board(fens[b]).legal_moves
                uci = list()
                for move in legal:
                    uci.append(move.uci())

                cp[f"legal_{b}"] = cp["move"]
                cp[f"legal_{b}"] = cp[f"legal_{b}"].isin(uci)
                # cp = cp.sort_values(by=[f"legal_{b}", f"prediction_{b}"], ascending=[False, False])

            return cp

    def learn(self, side, query):
        
        print(query[0].nodes[-1]["game_pgn"])

        d = AiComputer.create_dataset(query, side)
        loader = DataLoader(d, batch_size=1)
        
        for a, b in enumerate(loader):
            # print(b[0][0]) # fen
            # print(b[1][0]) # ten
            # print(b[1][1]) # additional
            # print(b[1]) # tensors and shit
            print(a)
            c = self.predict_full(b)
            print(c)
            break

    @staticmethod
    def create_dataset(query, side):

        # get all the nodes from graph element with the given nodename
        dataset = AiComputer.ChessMovesDataset(
            query_nodes=query,
            side=side,
            transform=AiComputer.TransformToTensor()
        )

        return dataset

    class Net(nn.Module):
        def __init__(self):
            super(AiComputer.Net, self).__init__()

            self.conv1 = nn.Conv2d(1, 50, 2)
            self.conv2 = nn.Conv2d(50, 4000, 1)
            self.pool = nn.MaxPool2d(2,2)
            self.fc1 = nn.Linear(400 + 21, 3600)
            self.fc2 = nn.Linear(3600, 1968)

        def forward(self, x1, x2):
            x1 = self.pool(F.mish(self.conv1(x1)))
            x1 = self.pool(F.mish(self.conv2(x1)))
            x1 = torch.flatten(x1, 1)
            x = torch.cat((x1, x2), dim=1)
            x = F.mish(self.fc1(x))
            x = F.softmax(self.fc2(x))
            return x

    class ChessMovesDataset(Dataset):
        # class for future building of chess moves dataset
        # multiple games into one 

        def __init__(self, query_nodes, side, transform=None):
            """
            query must have pgn in order to be added to games dataset
            and the last element must be the Game Node
            otherwise error
            """

            self.tf1 = AiComputer.TranformToFenlist(side)
            self.tf2 = transform
            
            # assert transform2 is not None and transform1 is None
            self.games_moves = list()
            for game in query_nodes:
                assert "Game" in game.nodes[-1].labels
                
                temp = dict(game.nodes[-1])
                assert temp["game_pgn"] is not None

                temp_moves = self.tf1(temp["game_pgn"])
                for move in temp_moves:
                    if move not in self.games_moves:
                        self.games_moves.append(move)

        def __len__(self):
            return len(self.games_moves)

        def __getitem__(self, idx):

            item = self.games_moves[idx]
            ten = ""
            if self.tf2 is not None:
                ten = self.tf2(item)
            return item, ten


    class TranformToFenlist(object):
        """
        not sure if correct approach
        """

        def __init__(self, side):
            self.side = True if "w" else False
            pass

        def __call__(self, pgn_string):
            act_pgn = io.StringIO(pgn_string)
            game = chess.pgn.read_game(act_pgn)

            move_list = list()
            board = game.board()
            for move in game.mainline_moves():
                t = self.side == board.turn
                board.push(move)
                if t:
                    move_list.append(board.fen())
            return move_list

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
                    t.append(AiComputer.TransformToTensor.handle_single_fen(inp[i]))
                return t
            else:
                t = AiComputer.TransformToTensor.handle_single_fen(inp)
                return t
        
        @staticmethod
        def handle_single_fen(fen_raw: str):

            temp = fen_raw.split(" ") 
            fen = temp.pop(0)
            temp.pop() # pop last element off of list (move number)
            castle = castle_move(fen_raw)
            en = en_passant(fen_raw)
            desc = torch.cat((castle, en), dim=1)
            desc = desc.to(AiComputer.cuda_device)
            
            ranks = fen.split("/")
            i = 0 
            rank_tensor = torch.zeros(
                8, 8, dtype=torch.float32, 
                device=AiComputer.cuda_device
            )
            for rank in ranks:
                j = 0
                for letter in rank:
                    if letter.isnumeric():
                        j += int(letter)
                    else:
                        piece = AiComputer.piece_2_number(letter)
                        rank_tensor[i, j] = piece
                        j += 1
                i += 1
            rank_tensor.unsqueeze_(0)
            return rank_tensor, desc

    @staticmethod
    def piece_2_number(letter):
        if letter == "p":
            return 1
        elif letter == "r":
            return 2
        elif letter == "n":
            return 3
        elif letter == "b":
            return 4
        elif letter == "q":
            return 5
        elif letter == "k":
            return 6
        elif letter == "P":
            return 7
        elif letter == "R":
            return 8
        elif letter == "N":
            return 9
        elif letter == "B":
            return 10
        elif letter == "Q":
            return 11
        elif letter == "K":
            return 12

    @staticmethod
    def number_2_piece(number):
        if number == 1:
            return "p"
        elif number == 2:
            return "r"
        elif number == 3:
            return "n"
        elif number == 4:
            return "b"
        elif number == 5:
            return "q"
        elif number == 6:
            return "k"
        elif number == 7:
            return "P"
        elif number == 8:
            return "R"
        elif number == 9:
            return "N"
        elif number == 10:
            return "B"
        elif number == 11:
            return "Q"
        elif number == 12:
            return "K"
        else:
            return None

    @staticmethod
    def fen_2_tensor(fen):
        if len(fen.split(" ")) > 1:
            fen = fen.split(" ")[0]
        ranks = fen.split("/")
        i = 0
        rank_tensor = torch.zeros(
            8, 8, dtype=torch.float32, device=AiComputer.cuda_device
        )
        for rank in ranks:
            j = 0
            for letter in rank:
                if letter.isnumeric():
                    j += int(letter)
                else:
                    piece = AiComputer.piece_2_number(letter)
                    rank_tensor[i, j] = piece
                    j += 1
            i += 1
        return rank_tensor

    @staticmethod
    def tensor_2_fen(ten):
        rows_ar = list()
        s1 = ten.size(0)
        s2 = ten.size(1)
        for i in range(s1):
            row = ten[i]
            row_str = ""
            empty = 0
            for j in range(s2):
                el = row[j]
                piece = AiComputer.number_2_piece(el.item())
                if piece is not None:
                    if empty != 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += piece
                else:
                    empty += 1
                    if j == s2 - 1:
                        row_str += str(empty)
            rows_ar.append(row_str)
        return "/".join(rows_ar)


if __name__ == "__main__":

    ai = AiComputer()

    db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))
    matcher = NodeMatcher(db)

    games = matcher.match("Game").all()
    sessions = matcher.match("TrainingNode").first()

    rel_matcher = RelationshipMatcher(db)
    res = rel_matcher.match((sessions, None), "Played").all()
    
    # ai.create_dataset(res)

    # dt = DataLoader(ai.dataset, batch_size=4, shuffle=True)

    # for i in enumerate(dt):

    #     pprint(i[1])
    #     t = ai.predict_full(i[1])
    #     print(t)
    #     break

    f = "r2qkb1r/pppn1ppp/8/3Pp3/4Q3/8/PP1P1PPP/R1B1KBNR w KQkq e6 0 8"
    print(ai.predict_single(f))
from numpy.core.fromnumeric import argmax
from pprint import pprint
import torch.nn as nn
import torch
import torch.nn.functional as F
from api.engines.template_computer import Computer
from torch.utils.data import Dataset, DataLoader
import chess, chess.pgn
import pandas as pd

import io
from py2neo import Graph, NodeMatcher, RelationshipMatcher


class AiComputer(Computer):

    # global cuda_devie
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):

        self.model = AiComputer.Net()
        self.model.to(self.cuda_device)
        self.tsfm = AiComputer.TransformToTensor()
        self.moves = pd.read_csv("./api/utils/all_moves_generator/all_moves.csv")
        self.dataset = None
        # fen_raw = "rn2kbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        # fen = fen_raw.split(" ")[0]
        # a = self.tsfm(fen)
        # print(a)
        # ten.unsqueeze_(0)
        # ten.unsqueeze_(0)
        # a = self.model(ten)
        # print(a)

    def predict(self, fen):
        with torch.no_grad():
            ten = self.tsfm(fen)
            ret = self.model(ten)
            return self.moves.iloc[torch.argmax(ret).item()]

    def predict_full(self, fen):
        with torch.no_grad():
            print(fen)
            ten = self.tsfm(fen)
            ret = self.model(ten)

            batch_size = list(ret.size())[0]
            cp = self.moves.copy()
            for b in range(batch_size):
                a = ret[b, :]
                
                cp[f"prediction_{b}"] = a.tolist()
                # cp = cp.sort_values(by="prediction", ascending=True)
                
                # legal_moves
                legal = chess.Board(fen).legal_moves
                uci = list()
                for move in legal:
                    uci.append(move.uci())

                cp[f"legal_{b}"] = cp["move"]
                cp[f"legal_{b}"] = cp[f"legal_{b}"].isin(uci)
                cp = cp.sort_values(by=[f"legal_{b}", f"prediction_{b}"], ascending=[False, False])

            return cp

    def create_dataset(self, query):

        # get all the nodes from graph element with the given nodename
        self.dataset = AiComputer.ChessMovesDataset(query_nodes=query)

    class Net(nn.Module):
        def __init__(self):
            super(AiComputer.Net, self).__init__()

            # define architecture
            self.conv1 = nn.Conv2d(1, 50, 2)
            self.conv2 = nn.Conv2d(50, 4000, 1)
            self.pool = nn.MaxPool2d(2,2)
            self.fc1 = nn.Linear(400 * 1 * 1, 3600)
            self.fc2 = nn.Linear(3600, 1792)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class ChessMovesDataset(Dataset):
        # class for future building of chess moves dataset
        # multiple games into one 

        def __init__(self, query_nodes, transform=None):
            """
            query must have pgn in order to be added to games dataset
            and the last element must be the Game Node
            otherwise error
            """

            self.tf1 = AiComputer.TranformToFenlist()
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
            if self.tf2 is not None:
                item = self.tf2(item)
            return item


    class TranformToFenlist(object):
        """
        not sure if correct approach
        """

        def __init__(self):
            pass

        def __call__(self, pgn_string):
            act_pgn = io.StringIO(pgn_string)
            game = chess.pgn.read_game(act_pgn)

            move_list = list()
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
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
        def handle_single_fen(fen: str):
            if len(fen.split(" ")) > 1:
                fen = fen.split(" ")[0]
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
            return rank_tensor

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

    fen_raw = "rn2kbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    # print(ai.predict_full(fen_raw))

    db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))
    matcher = NodeMatcher(db)

    games = matcher.match("Game").all()
    sessions = matcher.match("TrainingNode").first()

    rel_matcher = RelationshipMatcher(db)
    res = rel_matcher.match((sessions, None), "Played").all()
    
    ai.create_dataset(res)

    dt = DataLoader(ai.dataset, batch_size=4, shuffle=True)

    for i in enumerate(dt):
        t = ai.predict_full(i[1])
        print(t)
        break
from api.utils.decorators import timer
from pprint import pprint
import torch.nn as nn
import torch
import torch.nn.functional as F
from api.engines.template_computer import Computer
from torch.utils.data import Dataset, DataLoader
import chess, chess.pgn, time
import pandas as pd
import os
from api.utils.castling_move_number.castle import castle_move
from api.utils.en_passant_generator.generate import en_passant

import io
from py2neo import Graph, NodeMatcher, RelationshipMatcher


class AiComputer(Computer):

    # global cuda_devie
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    __name__ = "AiComputer"

    def __init__(self, load_model=False):
        self.model = AiComputer.Net()
        self.model.to(self.cuda_device)
        self.tsfm = AiComputer.TransformToTensor()
        self.moves = pd.read_csv("./api/utils/all_moves_generator/all_moves.csv")
        self.dataset = None
        self.model_path = "./api/engines/ai_engine/models"

        if load_model:
            m = os.listdir(self.model_path)
            if "model.pt" in m:
                self.model.load_state_dict(torch.load(f"{self.model_path}/model.pt"))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.50)
        self.loss_criterion = nn.MSELoss()

        self.legal_move_reward = 1
        self.decay = 0.99999
        self.min_legal_move_reward = 0.3

    def save_model(self, name="model.pt"):
        if self.model is not None:
            torch.save(self.model.state_dict(), f"{self.model_path}/{name}")

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

    def predict_full(self, inp, reward):

        # multiple moves out of transform
        fens = inp[0]["fen"]
        played_move = inp[0]["played_move"]
        tens = inp[1][0]
        desc = inp[1][1]
        ret = self.model(tens, desc)
        batch_size = list(ret.size())[0]
        
        target = torch.zeros(ret.shape, device=ret.device)
        for b in range(batch_size):
            fen = fens[b]
            legales = list(chess.Board(fen).legal_moves)
            uci_legales = [m.uci() for m in legales]
            for index, row in self.moves.iterrows():
                move = row["move"]
                if move in uci_legales:
                    target[b, index] = self.min_legal_move_reward
                    if move == played_move[b]:
                        target[b, index] = reward

        return ret, target

    @timer
    def learn(self, side, query):

        # if won highest reward 
        # if lost standard reward for making legal move
        game = query[0].nodes[-1]
        if (game["winner"] == "1-0" and side == "w") or (game["winner"] == "0-1" and side == "b"):
            reward = 10
        else:
            reward = self.min_legal_move_reward

        d = AiComputer.create_dataset(query, side)
        loader = DataLoader(d, batch_size=10, shuffle=True)
        print(f"len of dataset: {len(d)}")
        t = time.time()

        for a, b in enumerate(loader):
            net_ret, target = self.predict_full(b, reward)
            self.optimizer.zero_grad()
            loss = self.loss_criterion(net_ret, target)
            # print(f"Loss equal to: {loss.item()}")
            loss.backward()
            self.optimizer.step()

        if self.legal_move_reward > self.min_legal_move_reward:
            self.legal_move_reward = self.legal_move_reward * self.decay

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
            self.fc1 = nn.Linear(4000 + 21, 3600)
            self.fc2 = nn.Linear(3600, 1968)

        def forward(self, x1, x2):
            x1 = self.pool(F.mish(self.conv1(x1)))
            x1 = self.pool(F.mish(self.conv2(x1)))
            x1 = torch.flatten(x1, 1)
            x2 = torch.flatten(x2, 1)
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
            self.games_positions = list()
            i = 0 
            for game in query_nodes:
                assert "Game" in game.nodes[-1].labels
                
                temp = dict(game.nodes[-1])
                assert temp["game_pgn"] is not None

                temp_fens, temp_moves = self.tf1(temp["game_pgn"])
                if len(temp_moves) != len(temp_fens):
                    if len(temp_fens) > len(temp_moves):
                        temp_fens.pop()
                    else:
                        temp_moves.pop()
                assert len(temp_fens) == len(temp_moves)

                for j in range(len(temp_fens)):
                    fenn = temp_fens[j]
                    move = temp_moves[j]
                    if fenn not in self.games_positions:
                        item = dict()
                        item["fen"] = fenn
                        item["played_move"] = move
                        self.games_positions.append(item)
                i += 1

        def __len__(self):
            return len(self.games_positions)

        def __getitem__(self, idx):

            item = self.games_positions[idx]
            ten = ""
            if self.tf2 is not None:
                ten = self.tf2(item["fen"])
            return item, ten


    class TranformToFenlist(object):
        """
        not sure if correct approach
        """

        def __init__(self, side):
            self.side = True if side == "w" else False
            pass

        def __call__(self, pgn_string):
            act_pgn = io.StringIO(pgn_string)
            game = chess.pgn.read_game(act_pgn)

            positions_to_play = list()
            move_list = list()
            board = game.board()
            next = False
            if self.side:
                positions_to_play.append(board.fen())
                next = True

            for move in game.mainline_moves():
                if next:
                    move_list.append(move.uci())
                    next = False
                board.push(move)
                t = self.side == board.turn
                if t:
                    positions_to_play.append(board.fen())
                    next = True
                

            return positions_to_play, move_list

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
        temp = {
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
        return temp[letter]

    @staticmethod
    def number_2_piece(number):
        temp = {
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
        if number not in temp:
            return None
        else:
            return temp[number]

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

    f = "r2qkb1r/pppn1ppp/8/3Pp3/4Q3/8/PP1P1PPP/R1B1KBNR w KQkq e6 0 8"
    print(ai.predict_single(f))
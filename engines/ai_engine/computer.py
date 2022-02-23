from api.engines.template_computer import Computer
from api.utils.logger import MyLogger
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
import api.utils.decorators as d
import chess, chess.pgn, time
import io, random
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime

# use this you dumb fuck
module_logger = MyLogger(__name__)

class AiComputer(Computer):

    # global cuda_devie
    CUDA_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, *, load_model=False, model_name="model.pt", net:nn.Module):
        MOVES:List[chess.Move] = AiComputer.generate_moves_list()
        self.model = net()
        self.model.to(self.CUDA_DEVICE)
        # self.model_summary()

        self.tsfm = AiComputer.TransformToTensor()
        self.moves = pd.read_csv("./api/utils/all_moves_generator/all_moves.csv")
        self.dataset = None
        self.model_path = "./api/engines/ai_engine/models"
        self.model_name = model_name
        self.games_path = "./api/engines/ai_engine/games"

        if load_model:
            m = os.listdir(self.model_path)
            if self.model_name in m:
                self.model.load_state_dict(torch.load(f"{self.model_path}/{self.model_name}", map_location=AiComputer.CUDA_DEVICE))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_criterion = nn.MSELoss()

        self.legal_move_reward = 1
        # module_logger().info("Initialised the AiEngine") # example of logging

    def save_model(self, name=None):
        if self.model is not None:
            if name is not None:
                torch.save(self.model.state_dict(), f"{self.model_path}/{name}")
            else:
                torch.save(self.model.state_dict(), f"{self.model_path}/{self.model_name}")


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
        return cp.iloc[0]["move"]

    def __predict_single(self, fen, move_stack):
        ten, garb = self.tsfm(fen)
        ret = self.model(ten, garb)

        cp = self.moves.copy()
        temp = ret[0,:].tolist()
        ms = [m.uci() for m in move_stack[-7:]]
        temp = [1/10 * m if cp.iloc[i]["move"] in ms else m for i, m in enumerate(temp)]
        cp["prediction"] = temp
        legal = chess.Board(fen).legal_moves
        uci = list()
        for move in legal:
            uci.append(move.uci())

        cp["legal"] = cp["move"]
        cp["legal"] = cp["legal"].isin(uci)
        cp = cp.sort_values(by=[f"legal", f"prediction"], ascending=[False, False])
        return cp.iloc[0]["move"], ((ten, garb), ret)

    def predict_full(self, el):

        # multiple moves out of transform
        # el [0]    => fen
        # el [1]    => played move
        # el [2][0] => figure placement on the board
        # el [2][1] => en passant, castling
        # el [3]    => output of nn
        fen = el[0]
        model_ret = el[3]

        target = torch.zeros(model_ret.shape, device=model_ret.device)
        for b in range(model_ret.shape[0]):
            board = chess.Board(fen[b])

            if board.is_game_over():
                continue

            legales = list(board.legal_moves)
            uci_legales = [m.uci() for m in legales]
            for index, row in self.moves.iterrows():

                move = row["move"]
                if move in uci_legales:
                    target[b, 0, index] = self.legal_move_reward

                    # push move and check resulting position
                    m = chess.Move.from_uci(move)
                    turn = board.turn
                    board.push(m)
                    
                    # checkmate
                    if board.is_checkmate():
                        board.pop()
                        target[b, 0, index] = 5000 * (1 if turn else -1)
                        continue

                    # count pieces
                    score = 0
                    for key in self.piece_score:
                        score += (
                            len(board.pieces(key, turn)) - len(board.pieces(key, not turn))
                        ) * self.piece_score[key]
                    target[b, 0, index] += score if score > 0 else 0

                    # pawn advancement
                    score = 0
                    if turn:
                        score += sum([int(i/8) + 1 for i in board.pieces(1, True)])
                    else:
                        score += sum([(8 - int(i/8)) for i in board.pieces(1, False)])
                    target[b, 0, index] += score * 2

                    # piece activity
                    target[b, 0, index] += sum([len(board.attackers(turn, sq)) for sq in range(64)])

                    # king safety, count pieces around 
                    piece_sq = list(board.pieces(6, turn))[0] # king placement
                    sq = [piece_sq - i for i in range(-2, 3)]
                    squares = [j*8 + i for j in range(-2, 3) for i in sq if j*8+i >= 0 and j*8+i < 64 and j*8+i != piece_sq]
                    
                    score = 0
                    for s in squares:
                        if board.piece_at(s) is not None:
                            score += (1 if board.piece_at(s) == turn else -1)
                        else:
                            score -= 1
                    target[b, 0, index] += score * 10 if score > 0 else 0
                    board.pop()

            target[b, 0, :].mul_(1 / 5000)
        return model_ret, target

    @staticmethod
    def create_dataset(data):
        dataset = AiComputer.ChessMovesDataset(
            training_set=data,
        )

        return dataset

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

    def model_summary(self):
        model_stats = summary(self.model, [(1,8,8), (1,1,21)])
        if model_stats is not None:
            module_logger().info(f"\n{str(model_stats)}")

    def training(self):
        i = 1
        self.prev_game = None
        while True:
            training_set:List[Tuple[str, str, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = list()
            print(f"############################\nIteration: {i}")

            should_print = random.random() > 0.9
            ret = self.__single_game(should_print)
            training_set.extend(ret)
            
            dt = self.create_dataset(training_set)
            dl = DataLoader(dt, batch_size=len(dt), shuffle=True)
            i += 1
            mean_loss = list()
            for _, el in enumerate(dl):

                ret, target = self.predict_full(el)
                loss = self.loss_criterion(ret, target)
                mean_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # save model
            print("----------------------")
            print(f"Mean Loss: {np.mean(mean_loss)}")
            print("----------------------")

            self.save_model()


    def __single_game(self, p):
        board = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["White"] = "roofus"
        pgn.headers["Black"] = "doofus"
        pgn.setup(board)
        node = None
        
        game_set = list()
        while not board.is_game_over():
            m_uci, temp = self.__predict_single(board.fen(), board.move_stack)
            move = chess.Move.from_uci(m_uci)
            assert move in board.legal_moves
            
            mmmm = move if board.fullmove_number > 3 else random.choice(list(board.legal_moves))
            board.push(mmmm)
            a, b = temp
            game_set.append((board.fen(), move.uci(), a, b))

            if node is None:
                node = pgn.add_variation(mmmm)
            else:
                node = node.add_variation(mmmm)

        pgn.headers["Result"] = board.result()
        if p:
            print("################################################")
            print(pgn)
        # save the game
        if self.prev_game is not None:
            test = True
            for a, b in zip(self.prev_game.mainline_moves(), pgn.mainline_moves()):
                if a.uci() != b.uci():
                    test = False
                    break
            if test:
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        
        self.prev_game = pgn
        with open(f"{self.games_path}/{datetime.now().strftime('%d%m%Y_%H%M%S')}.pgn", mode="w") as f:
            f.write(str(pgn))

        return game_set

    class ChessMovesDataset(Dataset):
        # class for future building of chess moves dataset
        # multiple games into one 

        def __init__(self, training_set:List[Tuple[str, str, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]):
            self.games_positions = training_set

        def __len__(self):
            return len(self.games_positions)

        def __getitem__(self, idx):
            return self.games_positions[idx]

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

            output = dict()
            output["positions"] = list()
            output["played_move"] = list()
            board = game.board()
            counter = 0

            for move in game.mainline_moves():
                if self.side == board.turn:
                    counter += 1
                    output["positions"].append(board.fen())
                    output["played_move"].append(move.uci())
                # else: # LSTM MAYBE CONTEXT OF THE POSITION
                #     output["positions"].append(board.fen())
                #     output["played_move"].append("")
                board.push(move)

            output["counter"] = counter
            return output

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
            turn = temp.pop(0)
            castle = AiComputer.castle_move(fen_raw)
            en = AiComputer.en_passant(fen_raw)
            desc = torch.cat((castle, en), dim=1)
            desc = desc.to(AiComputer.CUDA_DEVICE)
            
            ranks = fen.split("/")
            if turn == "b":
                ranks.reverse()
            i = 0 
            rank_tensor = torch.zeros(
                8, 8, dtype=torch.float32, 
                device=AiComputer.CUDA_DEVICE
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
            rank_tensor = rank_tensor.unsqueeze(0)
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
            8, 8, dtype=torch.float32, device=AiComputer.CUDA_DEVICE
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

    @staticmethod
    def generate_moves_list():
        ret = list()

        # Queen moves
        base_fen_list = ["8", "8", "8", "8", "8", "8", "8", "8"]
        garbage = " w - - 0 1"
        for rank in range(8):
            for file in range(8):

                # create fen
                fen = base_fen_list.copy()
                first = "" if file == 0 else str(file)
                last = "" if file == 7 else str(7 - file)

                fen[rank] = f"{first}Q{last}"
                f = "/".join(fen) + garbage

                board = chess.Board(f)
                for move in list(board.legal_moves):
                    ret.append(move)

        # Knight moves
        base_fen_list = ["8", "8", "8", "8", "8", "8", "8", "8"]
        garbage = " w - - 0 1"
        for rank in range(8):
            for file in range(8):

                # create fen
                fen = base_fen_list.copy()
                first = "" if file == 0 else str(file)
                last = "" if file == 7 else str(7 - file)

                fen[rank] = f"{first}N{last}"
                f = "/".join(fen) + garbage

                board = chess.Board(f)
                for move in list(board.legal_moves):
                    ret.append(move)

        # promotions from 2-1 and from 7-8
        # 3 moves from each central square, 
        # 2 moves from the sides,
        # 4 promotion options
        temp = "8/PPPPPPPP/8/8/8/8/pppppppp/8 w - - 0 1"
        prom = chess.Board(temp).legal_moves

        # pawn pushes
        for move in list(prom):
            ret.append(move)

        temp = "nnnnnnnn/PPPPPPPP/8/8/8/8/pppppppp/8 w - - 0 1"
        prom = chess.Board(temp).legal_moves

        # pawn takes
        for move in list(prom):
            ret.append(move)

        # repeat for black side
        temp = "8/PPPPPPPP/8/8/8/8/pppppppp/8 b - - 0 1"
        prom = chess.Board(temp).legal_moves

        # pawn pushes
        for move in list(prom):
            ret.append(move)

        temp = "8/PPPPPPPP/8/8/8/8/pppppppp/NNNNNNNN b - - 0 1"
        prom = chess.Board(temp).legal_moves

        # pawn takes
        for move in list(prom):
            ret.append(move)

        return ret

if __name__ == "__main__":

    from api.engines.ai_engine.models.architecture3.net import Net
    eng = AiComputer(net=Net, model_name="model3.pt", load_model=True, )
    eng.training()
    
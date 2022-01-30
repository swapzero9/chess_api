import torch
import chess, chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import os, math
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from datetime import datetime
from api.utils.all_moves_generator.generate import generate_moves_list
from random import shuffle


@dataclass
class GameTree:
    hp: float = 0.5 # hyperparameter
    Qsa: Dict[Tuple[str, chess.Move], float] = field(default_factory=dict)      # Q values for state, action
    Nsa: Dict[Tuple[str, chess.Move], int] = field(default_factory=dict)        # Number of times state, action has been visited
    Ns:  Dict[str, int] = field(default_factory=dict)                           # Number of times state was visited
    Ps:  Dict[str, torch.Tensor] = field(default_factory=dict)                  # Initial policy for given state
    Es:  Dict[str, float] = field(default_factory=dict)                         # Stored values of finished games for given state
    Vs:  Dict[str, torch.Tensor] = field(default_factory=dict)                  # Stored legal moves for given state
    def eval(self, sa:Tuple[str, chess.Move], m:int):
        if sa in self.Qsa:
            return self.Qsa[sa] + self.hp * self.Ps[sa[0]][m].item() * math.sqrt(self.Ns[sa[0]]) / (1 + self.Nsa[sa])
        return self.hp * self.Ps[sa[0]][m].item() * math.sqrt(self.Ns[sa[0]] + 1)

class AiComputer2:

    CUDA_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MOVES:List[chess.Move] = generate_moves_list()

    def __init__(self, *, load_model=False, model_name="model.pt", hist_folder="", net=None):
        self.transform = AiComputer2.TransformToTensor()
        self.model = net(self.transform)
        self.model.to(self.CUDA_DEVICE)
        # self.model_summary()

        self.model_path = "./api/engines/ai_engine_new/models"
        # self.model_path = "./models"
        self.model_name = model_name
        self.game_path = f"./api/engines/ai_engine_new/{hist_folder}"

        if load_model:
            m = os.listdir(self.model_path)
            if self.model_name in m:
                print("load_model")
                self.model.load_state_dict(torch.load(f"{self.model_path}/{self.model_name}"))

        self.__mcts = AiComputer2.MonteCarloSearch(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_criterion = nn.MSELoss()

    def save_model(self, name=None):
        if self.model is not None:
            if name is not None: 
                torch.save(self.model.state_dict(), f"{self.model_path}/{name}")
            else:
                torch.save(self.model.state_dict(), f"{self.model_path}/{self.model_name}")

    def model_summary(self):
        model_stats = summary(self.model, [(1,85)])
        if model_stats is not None:
            print(model_stats)

    def think(self, fen):
        b = chess.Board(fen)
        self.__mcts(b)

    def learn(self):
        for i in range(0, 10000):
            print(f"#############################\nIteration: {i}")
            iteration_set = list()

            should_print = True
            for _ in range(5):
                temp = self.__execute_session(should_print)
                iteration_set.extend(temp)
                should_print = False
            
            shuffle(iteration_set)
            self.__train(iteration_set)
            self.save_model()
            self.__mcts.reset_game_tree()

    def __train(self, tset):
        pi_losses = list()
        v_losses = list()

        epoch_num = 2
        print(f"Entered training with {len(tset)} sets of data")

        for e in range(epoch_num):
            print(f"Epoch: {e}")
            dt = AiComputer2.TrainingDataset(tset)
            dl = DataLoader(dt, batch_size=1, shuffle=True)
            for _, b in enumerate(dl):

                brd, t_pi, t_v = b
                out_pi, out_v = self.model(brd)

                l_pi = self.loss_pi(t_pi, out_pi)
                l_v = self.loss_v(t_v, out_v)
                total = l_pi + l_v
                
                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

            print("-----------------------------")
            print(f"Policy Loss: {np.mean(pi_losses)}")
            print(f"Value Loss:  {np.mean(v_losses)}")


    def loss_pi(self, targets, outputs):
        return self.loss_criterion(targets, outputs)

    def loss_v(self, targets, outputs):
        return self.loss_criterion(targets, outputs)


    def create_moveset(self, arg):
        return self.ChessMoveset(arg)

    def __execute_session(self, p:bool):
        train_set = list()
        board = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["White"] = "roofus"
        pgn.headers["Black"] = "doofus"
        pgn.setup(board)
        node = None

        # run single training session, save all the positions and results from net
        result = None
        while not board.is_game_over() and board.fullmove_number < 75:
            
            probabilities, _ = self.__mcts.get_probabilities(board)
            train_set.append([board.fen(), probabilities, None])
            move_indx = np.random.choice(len(AiComputer2.MOVES), p=probabilities)
            move = AiComputer2.MOVES[move_indx]
            if move not in board.legal_moves:
                print(board.fen())
                print(move)
                assert move in board.legal_moves

            board.push(move)
            if node is None:
                node = pgn.add_variation(move)
            else:
                node = node.add_variation(move)
        
        if not board.is_game_over():
            pgn.headers["Result"] = "1/2-1/2"
        else:
            pgn.headers["Result"] = board.result()
        try:
            print(f"Game finished! \nResult: {board.result()}\nGame Val: {self.__mcts.gt.Es[AiComputer2.get_base_board(board)]}")
        except Exception as ex:
            print(ex)

        # save pgn to file
        with open(f"{self.game_path}/{datetime.now().strftime('%d%m%Y_%H%M%S')}.pgn", mode="w") as f:
            f.write(str(pgn))

        if p:
            print(f"#############################\nGame from iteration")
            print(f"#############################")
            print(str(pgn))

        result = AiComputer2.get_normalised_outcome(board)
        train_set = [[x[0], x[1], result] for x in train_set]
        return train_set

    class TrainingDataset(Dataset):

        def __init__(self, tset):
            self.tsfm = AiComputer2.TransformToTensor()
            self.db:List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list()

            for el in tset:
                pi = torch.FloatTensor(np.asarray(el[1]))
                pi = pi.to(AiComputer2.CUDA_DEVICE)

                v = torch.FloatTensor(np.asarray([el[2]]))
                v = v.to(AiComputer2.CUDA_DEVICE)

                temp = (
                    self.tsfm(el[0]),
                    pi, v
                )
                self.db.append(temp)
            pass

        def __getitem__(self, index):
            return self.db[index]

        def __len__(self):
            return len(self.db)


    class TransformToTensor(object):
        """
        takes a fen input or a list and creates tensorlist
        """

        def __init__(self):
            pass
        
        def __call__(self, inp, arg=False):
            assert isinstance(inp, list) or isinstance(inp, str) or isinstance(inp, tuple)

            if isinstance(inp, list) or isinstance(inp, tuple):
                t = list()
                for i in range(len(inp)):
                    t.append(AiComputer2.TransformToTensor.handle_single_fen(inp[i]), arg)
                return t
            else:
                t = AiComputer2.TransformToTensor.handle_single_fen(inp, arg)
                return t
        
        @staticmethod
        def handle_single_fen(fen_raw: str, arg=False):

            temp = fen_raw.split(" ") 
            fen = temp.pop(0)
            turn = temp.pop(0)
            castle = AiComputer2.castle_move(fen_raw)
            en = AiComputer2.en_passant(fen_raw)
            desc = torch.cat((castle, en), dim=1)
            desc = desc.to(AiComputer2.CUDA_DEVICE)
            
            ranks = fen.split("/")
            if turn == "b":
                ranks.reverse()
            i = 0 
            rank_tensor = torch.zeros(
                64, dtype=torch.float16, 
                device=AiComputer2.CUDA_DEVICE
            )
            for rank in ranks:
                j = 0
                for letter in rank:
                    if letter.isnumeric():
                        j += int(letter)
                    else:
                        piece = AiComputer2.piece_2_number(letter)
                        rank_tensor[i*8+j] = piece
                        j += 1
                i += 1
            rank_tensor.unsqueeze_(0)
            ret = torch.cat([rank_tensor, desc], dim=1)
            if arg:
                ret.unsqueeze_(0)
            return ret

    class MonteCarloSearch:

        def __init__(self, model) -> None:
            self.model:AiComputer2.Net = model
            self.gt = GameTree()
            self.mcts_am = 40

        def __call__(self, board:chess.Board):

            state = AiComputer2.get_base_board(board)
            if board.is_game_over():
                t = AiComputer2.get_normalised_outcome(board)
                self.gt.Es[state] = t
                return (-1) * t
            if board.fullmove_number > 75:
                # draw, took too long
                self.gt.Es[state] = 1e-4
                return (-1.0) * 1e-4

            if state not in self.gt.Ps:
                self.gt.Ps[state], value = self.model.predict(board.fen())
                self.gt.Ps[state].squeeze_(0)
                valid_moves = torch.FloatTensor(AiComputer2.get_masked_valid_moves(board))
                valid_moves = valid_moves.to(AiComputer2.CUDA_DEVICE)
                self.gt.Ps[state] *= valid_moves
                csum = torch.sum(self.gt.Ps[state])

                if csum > 0:
                    self.gt.Ps[state] /= csum
                else:
                    self.gt.Ps[state] += valid_moves
                    self.gt.Ps[state] /= torch.sum(self.gt.Ps[state])

                self.gt.Vs[state] = valid_moves
                self.gt.Ns[state] = 0
                return (-1.0) * value.item()

            # highest ucb
            best = -math.inf
            move = None
            for action in board.legal_moves:
                ucb_val = self.gt.eval((state, action), AiComputer2.get_move_index(action))
                if ucb_val > best:
                    best = ucb_val
                    move = action
            
            board.push(move)
            value = self.__call__(board)
            board.pop()

            if (state, move) in self.gt.Qsa:
                self.gt.Qsa[(state, move)] = (self.gt.Nsa[(state, move)] * self.gt.Qsa[(state, move)] + value) / (1 + self.gt.Nsa[(state, move)])
            else:
                self.gt.Qsa[(state, move)] = value
                self.gt.Nsa[(state, move)] = 1
            self.gt.Ns[state] += 1
            return (-1.0) * value


        def get_probabilities(self, board:chess.Board, temp=1):

            for _ in range(self.mcts_am):
                _ = self.__call__(board.copy())

            state = AiComputer2.get_base_board(board)
            legal_moves = list(board.legal_moves)
            counts = [self.gt.Nsa[(state, action)] if (state, action) in self.gt.Nsa else 0 for action in AiComputer2.MOVES]

            csum = float(sum(counts))
            if csum > 0:
                probs = [x / csum for x in counts]
            else:
                probs = [0 for _ in counts]
                probs[np.random.choice(len(probs))] = 1
            return probs, legal_moves

        def reset_game_tree(self):
            self.gt = GameTree()

    class ChessMoveset(Dataset):
        def __init__(self, nodelist):
            self.dataset = list()

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

    @staticmethod
    def get_base_board(b: chess.Board):
        f = b.fen().split(" ")
        f.pop()
        f.pop()
        f = "_".join(f)
        return f

    @staticmethod
    def get_normalised_outcome(board:chess.Board):
        if board.is_game_over():
            temp = board.outcome()
            if temp.winner is None:
                return 1e-4
            elif temp.winner == True:
                return 1.0
            elif temp.winner == False:
                return -1.0
        else:
            return 0
    
    @staticmethod
    def get_masked_valid_moves(board:chess.Board):
        masked = [0 for _ in range(len(AiComputer2.MOVES))]
        for move in board.legal_moves:
            i = AiComputer2.MOVES.index(move)
            masked[i] = 1
        return masked

    @staticmethod
    def get_move_index(move:chess.Move):
        return AiComputer2.MOVES.index(move)

if __name__ == "__main__":

    from api.engines.ai_engine_new.models.architecture1.net import Net as net1
    from api.engines.ai_engine_new.models.architecture2.net import Net as net2

    a = AiComputer2.get_base_board(chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"))
    # eng1 = AiComputer2(load_model=False, model_name="model1.pt", net=net1, hist_folder="games_history_a1")
    # eng2 = AiComputer2(load_model=False, model_name="model2.pt", net=net2, hist_folder="games_history_a2")
    

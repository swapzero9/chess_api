from time import time
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
import random
import torch.multiprocessing as mp


@dataclass
class GameTree:
    hp: float = 0.3 # hyperparameter
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
                t.append(self.handle_single_fen(inp[i]), arg)
            return t
        else:
            t = self.handle_single_fen(inp, arg)
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

class AiComputer2:
    CUDA_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MOVES:List[chess.Move] = generate_moves_list()
    tsfm = TransformToTensor()

    def __init__(self, *, load_model=False, model_name="model.pt", hist_folder="", net=None):
        self.model:nn.Module = net()
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
                self.model.load_state_dict(torch.load(f"{self.model_path}/{self.model_name}", map_location=AiComputer2.CUDA_DEVICE))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_criterion = nn.MSELoss()

    def save_model(self, name=None):
        if self.model is not None:
            if name is not None: 
                torch.save(self.model.state_dict(), f"{self.model_path}/{name}")
            else:
                torch.save(self.model.state_dict(), f"{self.model_path}/{self.model_name}")

    def model_summary(self):
        model_stats = summary(self.model, [(1,1,85)])
        if model_stats is not None:
            print(model_stats)

    def think(self, fen):
        b = chess.Board(fen)
        temp_mcts = self.MonteCarloSearch(self.model, 200)
        probab, _ = temp_mcts.get_probabilities(b)
        move_indx = np.random.choice(len(AiComputer2.MOVES), p=probab)
        move = AiComputer2.MOVES[move_indx]
        if move not in b.legal_moves:
            print(b.fen())
            print(move)
            assert move in b.legal_moves

        return move

    def learn(self):
        proc_num = 5
        games_in_batch = 20
        self.model.share_memory()

        # start game processes
        queue = mp.JoinableQueue(maxsize=games_in_batch)
        lock = mp.Lock()
        processes = list()
        shared_list = mp.Manager().list()
        for i in range(proc_num):
            p = mp.Process(target=self.__execute_session, args=(queue, lock, shared_list))
            processes.append(p)
            p.start()

        for i in range(0, 100000):
            print(f"#############################\nIteration: {i}")
            iteration_set = list()

            seed_list = list()
            while True:
                seed_list = np.random.randint(0, 10000, size=games_in_batch).tolist()
                if len(seed_list) == len(list(set(seed_list))):
                    break
            for j in range(games_in_batch):
                queue.put((seed_list[j] > 1000, j, seed_list[j]))
            
            print("Queued games")
            queue.join()
            print("Games finished")

            iteration_set = shared_list[:]
            lock.acquire()
            try:
                shared_list[:] = []
            except Exception as ex:
                print(ex)
            finally:
                lock.release()

            shuffle(iteration_set)
            self.__train(iteration_set)
            self.save_model()

    def __train(self, tset):
        epoch_num = 2
        print(f"Entered training with {len(tset)} sets of data")
        for e in range(epoch_num):
            pi_losses = list()
            v_losses = list()
            dt = AiComputer2.TrainingDataset(tset)
            dl = DataLoader(dt, batch_size=1, shuffle=True)
            for _, b in enumerate(dl):
                brd, t_pi, t_v = b
                out_pi, out_v = self.model(brd)
                t_v.unsqueeze_(0)
                l_pi = self.loss_pi(t_pi, out_pi)
                l_v = self.loss_v(t_v, out_v)
                total = l_pi + l_v
                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()
            
            print("-----------------------------")
            print(f"Epoch: {e+1}")
            print(f"Policy Loss: {np.mean(pi_losses)}")
            print(f"Value Loss:  {np.mean(v_losses)}")


    def loss_pi(self, targets, outputs):
        return self.loss_criterion(targets, outputs)

    def loss_v(self, targets, outputs):
        return self.loss_criterion(targets, outputs)

    def __execute_session(self, queue, lock, res):
        while True:
            data = queue.get()
            # unpack data
            p = data[0]
            id = data[1]
            seed = data[2]

            # init variables
            train_set = list()
            board = chess.Board()
            pgn = chess.pgn.Game()
            pgn.headers["White"] = "roofus"
            pgn.headers["Black"] = "doofus"
            pgn.setup(board)
            node = None
            np.random.seed(seed)
            random.seed(seed)
            mcts = AiComputer2.MonteCarloSearch(self.model, 50, True)
            print(f'-- {datetime.today().strftime("%d/%m/%Y %H:%M:%S")} -- Running the game {id+1}')
            # run single training session, save all the positions and results from net
            result = None
            while not board.is_game_over() and board.fullmove_number < 170:            
                probabilities, _ = mcts.get_probabilities(board)
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
                print(f"--------------------\nGame finished! \nResult: {board.result() if board.is_game_over() else '1/2-1/2'}\nNumber of Moves: {board.fullmove_number}")
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
            lock.acquire()
            try: 
                res.extend(train_set)
            except Exception as ex:
                print(ex)
            finally:
                queue.task_done()
                lock.release()

    class TrainingDataset(Dataset):

        def __init__(self, tset):
            self.db:List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list()

            for el in tset:
                pi = torch.FloatTensor(np.asarray(el[1]))
                pi = pi.to(AiComputer2.CUDA_DEVICE)

                v = torch.FloatTensor(np.asarray([el[2]]))
                v = v.to(AiComputer2.CUDA_DEVICE)

                temp = (
                    AiComputer2.tsfm(el[0]),
                    pi, v
                )
                self.db.append(temp)
            pass

        def __getitem__(self, index):
            return self.db[index]

        def __len__(self):
            return len(self.db)

    class MonteCarloSearch:

        def __init__(self, model, am=50, is_training=False) -> None:
            self.model:AiComputer2.Net = model
            self.gt = GameTree()
            self.mcts_am = am
            self.is_training = is_training

        def __call__(self, board:chess.Board):

            state = AiComputer2.get_base_board(board)
            turn_mult = (1.0 if board.turn else -1.0)
            if board.is_game_over():
                t = AiComputer2.get_normalised_outcome(board)
                self.gt.Es[state] = t
                return turn_mult * t
            if self.is_training:
                if board.fullmove_number >= 175:
                    # draw, took too long
                    self.gt.Es[state] = 1e-4
                    return turn_mult * 1e-4

            if state not in self.gt.Ps:
                self.gt.Ps[state], value = self.model.predict(AiComputer2.tsfm(board.fen()))
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
                return turn_mult * value.item()

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
                self.gt.Nsa[(state, move)] += 1
            else:
                self.gt.Qsa[(state, move)] = value
                self.gt.Nsa[(state, move)] = 1
            self.gt.Ns[state] += 1
            return turn_mult * value


        def get_probabilities(self, board:chess.Board):

            for _ in range(self.mcts_am):
                _ = self.__call__(board.copy())

            legal_moves = list(board.legal_moves)
            if self.is_training:
                for m in board.legal_moves:
                    board.push(m)
                    if board.is_checkmate():
                        custom_probabilities = np.zeros(len(AiComputer2.MOVES)).tolist()
                        custom_probabilities[AiComputer2.MOVES.index(m)] = 1
                        board.pop()
                        return custom_probabilities, legal_moves
                    board.pop()

            state = AiComputer2.get_base_board(board)
            dis = [self.gt.Qsa[(state, action)] if (state, action) in self.gt.Qsa else 0 for action in AiComputer2.MOVES]
            max_dis = max(dis)
            min_dis = min(dis)
            dis_interp = [np.interp(v, [min_dis, max_dis], [0.5, 1]) for v in dis]
            counts = [self.gt.Nsa[(state, action)] * v if (state, action) in self.gt.Nsa else 0 for action, v in zip(AiComputer2.MOVES, dis_interp) ]
            # counts = [self.gt.Nsa[(state, action)] if (state, action) in self.gt.Nsa else 0 for action in AiComputer2.MOVES]

            if self.is_training:
                last_moves = [AiComputer2.MOVES.index(m) for m in board.move_stack[-9:]]
                for m in last_moves:
                    counts[m] *= 1/10

            csum = float(sum(counts))
            if csum > 0:
                probs = [x / csum for x in counts]
            else:
                probs = [0 for _ in counts]
                probs[np.random.choice(len(probs))] = 1
            return probs, legal_moves

        def reset_game_tree(self):
            self.gt = GameTree()


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
        masked = np.zeros(len(AiComputer2.MOVES)).tolist()
        for move in board.legal_moves:
            i = AiComputer2.MOVES.index(move)
            masked[i] = 1
        return masked

    @staticmethod
    def get_move_index(move:chess.Move):
        return AiComputer2.MOVES.index(move)

if __name__ == "__main__":

    from api.engines.ai_engine_new.models.l5.net import Net as net

    a = AiComputer2(load_model=False, net=net)
    a.model_summary()
    # eng1 = AiComputer2(load_model=False, model_name="model1.pt", net=net1, hist_folder="games_history_a1")
    # eng2 = AiComputer2(load_model=False, model_name="model2.pt", net=net2, hist_folder="games_history_a2")
    

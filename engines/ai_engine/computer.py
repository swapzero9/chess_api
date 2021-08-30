import torch.nn as nn
import torch
from api.engines.template_computer import Computer
from torch.utils.data import Dataset
import chess, chess.pgn

import io


class AiComputer(Computer):

    # global cuda_devie
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):

        self.model = AiComputer.Net(self)
        fen_raw = "rn2kbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        fen = fen_raw.split(" ")[0]
        # ten = self.fen_2_tensor(fen)
        # ten.unsqueeze_(0)
        # ten.unsqueeze_(0)
        # a = self.model(ten)

        pgn = """
[Event "Live Chess"]
[Site "Chess.com"]
[Date "2021.04.16"]
[Round "-"]
[White "swapzero1111"]
[Black "chesslover240"]
[Result "1-0"]
[WhiteElo "1142"]
[BlackElo "1105"]
[TimeControl "180"]
[EndTime "6:11:44 PDT"]
[Termination "swapzero1111 won by resignation"]

1. c4 e5 2. Nc3 Bb4 3. a3 Bxc3 4. dxc3 Nf6 5. b4 a6 6. g3 d6 7. Bg2 O-O 8. b5
axb5 9. cxb5 Ra7 10. Be3 Ra8 11. a4 Qd7 12. Qb3 c6 13. Nh3 cxb5 14. a5 Na6 15.
Bb6 e4 16. e3 d5 17. O-O Qd6 18. Nf4 Nc5 19. Bxc5 Qxc5 20. Nxd5 Nxd5 21. Bxe4
Be6 22. Rfd1 Nxe3 23. c4 Bxc4 24. Qxe3 Rfe8 25. Qxc5 Rxe4 26. Rd7 Rae8 27. Qf5
Be6 28. Qxe4 Bxd7 29. Qxb7 1-0
        """
        print(pgn)
        aa = Computer.pgn_2_fenlist(pgn)
        bb = Computer.fenlist_2_pgn(aa)
        print(bb)

    @classmethod
    def create_dataset(tensor_list):

        pass

    class Net(nn.Module):
        def __init__(self, outer):
            super(AiComputer.Net, self).__init__()

            # outer class
            self.outer = outer

            # define architecture
            self.conv1 = torch.nn.Conv2d(1, 25, 4, device=AiComputer.cuda_device)

        def forward(self, x):
            return self.conv1(x)

    class ChessMovesDataset(Dataset):
        # class for future building of chess moves dataset

        def __init__(self):
            pass

        def __len__(self):
            pass

        def __getitem__(self, idx):
            pass

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

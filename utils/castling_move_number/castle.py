import torch

def castle_move(fen):

    # r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1
    # r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1
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

if __name__ == "__main__":

    r = castle_move("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    print(r)
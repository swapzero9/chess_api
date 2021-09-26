import torch

def en_passant(fen):
    
    # rnbqkbnr/pp1p1ppp/8/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3
    # a3 a6
    # b3 b6
    # c3 c6
    # d3 d6
    # e3 e6
    # f3 f6
    # g3 g6
    # h3 ah6
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

    print(en_passant("rnbqkbnr/pp1p1ppp/8/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3"))
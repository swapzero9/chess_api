from api.engines.ai_engine.computer import AiComputer
from api.engines.training import TrainingSession
import api.utils.decorators as d
from api.engines.ai_engine.models.architecture1.net import Net as n1
from api.engines.ai_engine.models.architecture3.net import Net as n2
from multiprocessing import Process

@d.timer_log
@d.debug_log
def main():
    # player = AiComputer(net=Net)
    # player = AiComputer(load_model=False, model_name="model_example.py")

    # t = TrainingSession("Ai_1", player)
    # t.train()
    ar = [
        # ("Test Training", "model", n2),
    ]
    for el in ar: 
        p = Process(target=single_session, args=el)
        p.start()

    pass

def single_session(name, model_name, net):
    player1 = AiComputer(model_name=f"{model_name}1.pt", net=net)
    player2 = AiComputer(model_name=f"{model_name}2.pt", net=net)
    t = TrainingSession(name, player1, player2)
    t.train()

if __name__ == "__main__":
    main()

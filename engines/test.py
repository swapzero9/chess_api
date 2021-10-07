from py2neo import Graph, Node
import api.utils.decorators as d
from api.engines.random_engine.computer import RandomComputer
from api.engines.training import TrainingSession
from api.engines.ai_engine.computer import AiComputer
from multiprocessing import Process

@d.timer
@d.debug
def main():
    white = AiComputer(model_name="white_model.pt")
    black = AiComputer(model_name="black_model.pt")

    t = TrainingSession("Ai_training_new_model", white, black)
    t.train()

if __name__ == "__main__":
    main()

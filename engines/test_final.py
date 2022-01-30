
from api.engines.ai_engine_new.models.architecture4.net import Net as net1
from api.engines.ai_engine_new.computer import AiComputer2

if __name__ == "__main__":
    eng = AiComputer2(load_model=True, model_name="model4.pt", net=net1, hist_folder="games_history_a4")
    eng.learn()
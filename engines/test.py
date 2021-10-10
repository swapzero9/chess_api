from api.engines.ai_engine.computer import AiComputer
from api.engines.training import TrainingSession
import api.utils.decorators as d

@d.timer_log
@d.debug_log
def main():
    player = AiComputer()
    # player = AiComputer(load_model=False, model_name="model_example.py")

    t = TrainingSession("Ai_with_validation", player)
    t.train()

if __name__ == "__main__":
    main()

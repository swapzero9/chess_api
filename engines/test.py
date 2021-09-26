from py2neo import Graph, Node
import api.utils.decorators as d
from api.engines.random_engine.computer import RandomComputer
from api.engines.training import TrainingSession
from api.engines.ai_engine.computer import AiComputer


@d.timer
@d.debug
def main():
    # db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))
    # db.delete_all()
    # db.commit()

    # tx = db.begin()
    # a = Node("Penice", name="asd")
    # tx.create(a)
    # db.commit(tx)
    white = AiComputer()
    black = RandomComputer()


    t = TrainingSession("First_Ai_game", white, black, 10)
    t.train()

if __name__ == "__main__":
    main()

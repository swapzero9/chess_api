from py2neo import Graph, Node
import api.utils.decorators as d


@d.timer
@d.debug
def main():
    db = Graph("bolt://localhost:7687", auth=("neo4j", "s3cr3t"))
    # db.delete_all()
    # db.commit()

    tx = db.begin()
    a = Node("Penice", name="asd")
    tx.create(a)
    db.commit(tx)


if __name__ == "__main__":
    main()

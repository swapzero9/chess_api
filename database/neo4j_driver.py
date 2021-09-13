from neo4j import GraphDatabase
import json

####################################################################
# node class for neo4j
class NeoNode:
    def __init__(self, label: str, props: dict = None, ret: list = None):
        self.label = label
        if props == None:
            self.props = dict()
        else:
            self.props = props
        if ret == None:
            self.ret = list()
        else:
            self.ret = ret

        self.str_props = self.get_props()

    def get_props(self):
        properties = list()
        for property in self.props:
            properties.append(f"{property}: '{self.props[property]}'")

        if len(properties) > 0:
            props = " {" + ", ".join(properties) + "}"
        else:
            props = ""
        return props

    def get_rets(self, id: str):
        # ["asd", "asd"]
        t = list()
        t.append(f"ID({id})")
        for el in self.ret:
            t.append(f"{id}.{el}")
        return ", ".join(t)


####################################################################
# relationship class for neo4j
# depends on node
# fuck you to do
class NeoRelationship:
    def __init__(self, label: str, props: dict = None, ret: list = None):
        self.label = label
        if props == None:
            self.props = dict()
        else:
            self.props = props

        self.str_props = self.get_props()

    def get_props(self):
        properties = list()
        for property in self.props:
            properties.append(f"{property}: '{self.props[property]}'")

        if len(properties) > 0:
            props = " {" + ", ".join(properties) + "}"
        else:
            props = ""
        return props


####################################################################
# driver for executin neo4j queries
# depends on Node and Relationship Class
class Neo4j_Driver:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, node: NeoNode, ret=False):
        string = ""
        if ret:
            t = node.get_rets("n")
            string = f" RETURN {t}"

        query = f"CREATE (n:{node.label}{node.str_props}){string}"

        with self.driver.session() as sess:
            res = sess.write_transaction(self.execute_command, query)
            return res

    # reads data inside specified node
    # node must have specified rets property
    def read_node(self, node: NeoNode, id: int = None):
        rets = node.get_rets("n")
        if id != None:
            query = f"MATCH (n:{node.label}{node.str_props}) WHERE ID(n) = {id} RETURN {rets}"
        else:
            query = f"MATCH (n:{node.label}{node.str_props}) RETURN {rets}"

        with self.driver.session() as sess:
            ret = sess.write_transaction(self.execute_command, query)
            return ret

    # creates node with relation to existing node
    def create_node_rel(
        self, a: NeoNode, b: NeoNode, r: NeoRelationship, a_id: int = None
    ):
        a_rets = a.get_rets("a")
        b_rets = b.get_rets("b")

        query = f"MATCH "
        if a_id != None:
            query += f"(a:{a.label}{a.str_props}) WHERE ID(a) = {a_id}"
        else:
            query += f"(a:{a.label}{a.str_props})"
        query += f" CREATE (b:{b.label}{b.str_props}) "
        query += f"CREATE (a)-[r:{r.label}{r.str_props}]->(b) RETURN {a_rets}, {b_rets}"
        with self.driver.session() as sess:
            ret = sess.write_transaction(self.execute_command, query)
            return ret

    # create relation between existing nodes ``
    def match_nodes_rel(
        self,
        a: NeoNode,
        b: NeoNode,
        r: NeoRelationship,
        a_id: int = None,
        b_id: int = None,
    ):
        a_rets = a.get_rets("a")
        b_rets = b.get_rets("b")

        query = f"MATCH "
        query += f"(a:{a.label}{a.str_props})"
        query += f", (b:{b.label}{b.str_props})"

        if a_id != None and b_id != None:
            query += f"WHERE ID(a) = {a_id} and ID(b) = {b_id}"
        elif a_id != None and b_id == None:
            query += f"WHERE ID(a) = {a_id}"
        elif a_id == None and b_id != None:
            query += f"WHERE ID(b) = {b_id}"
        query += (
            f" CREATE (a)-[r:{r.label}{r.str_props}]->(b) return {a_rets}, {b_rets}"
        )

        with self.driver.session() as sess:
            ret = sess.write_transaction(self.execute_command, query)
            return ret

    # create nodes with relation to eachother
    def create_nodes_rel(self, a: NeoNode, b: NeoNode, r: NeoRelationship):
        a_rets = a.get_rets("a")
        b_rets = b.get_rets("b")

        query = f"CREATE (a:{a.label} {a.str_props}), (b:{b.label} {b.str_props}) "
        query += f"CREATE (a)-[r:{r.label} {r.str_props}]->(b) "
        query += f"RETURN {a_rets}, {b_rets}"

        with self.driver.session() as session:
            ret = session.write_transaction(self.execute_command, query)
            return ret

    # execute custom method
    def run_query(self, query):
        with self.driver.session as session:
            ret = session.write_transaction(self.execute_command, query)
            return ret

    # just for executing methods
    @staticmethod
    def execute_command(tx, query: str):
        result = tx.run(query)
        return result.values()

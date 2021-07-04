from neo4j_driver import Neo4j_Driver, NeoNode, NeoRelationship

if __name__ == "__main__":

	d1 = {
	}
	d2 = {
	}
	l = ["name"]

	temp1 = NeoNode("shorts", d1, l)
	temp2 = NeoNode("shorts", d2, l)

	r = NeoRelationship("Relates", {"since": "you do not suck"})

	db = Neo4j_Driver("bolt://localhost:7687","neo4j","s3cr3t")

	a = db.match_nodes_rel(temp1, temp2, r, 18, 20)

	print(a)
	db.close()
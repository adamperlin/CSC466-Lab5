class Node:
    def __init__(self):
        self.var = ""
        self.edges = []


class Edge:
    def __init__(self):
        self.value = ""
        self.node = None


class Leaf:
    def __init__(self, decision, p):
        self.decision = decision
        self.p = p

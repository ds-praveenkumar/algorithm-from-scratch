# graph.py
from collections import OrderedDict

class Graph:
    """ 
    container class for storing graph data structure
    
    
    """
    def __init__(self):
        pass

    def generate_graph(self): #graph/graph.txt
        with open( "graph/graph.txt", "r") as f:
            lines = f.readlines()
        lines = [line.strip().split(',') for line in lines]
        od = OrderedDict()
        for line in lines:
            root = int(line.pop(0))
            od[root] = line
        return od



if __name__ == "__main__":
    g = Graph()
    graph =  g.generate_graph()
    print( graph)
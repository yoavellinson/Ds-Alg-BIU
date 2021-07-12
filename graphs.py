from _typeshed import Self
import numpy as np


INF = 999999999999999999

class Node:
	def __init__(self, val):
        self.val = val
        self.edges = []
		self.d = None
		self.p = None

class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes

    def add_node(self, val):
        newNode = Node(val)
        self.nodes.append(newNode)

    def add_edge(self, node1, node2):
        node1.edges.append(node2)
        node2.edges.append(node1)
	

def Floyed_Marshell(W): #all pairs shortest paths
    d = np.array(W)
    n = d.shape[1]
    for k in range(0,n):
         for i in range(0,n):
            for j in range(0,n):
                d[i][j]=min(d[i][j],d[i][k]+d[k][j])
    print(d)


def dijkstra(G,s,t):
	for v in G:
		v.d = INF
		v.p = None
	s.d = 0
	S = []
	for node in G:
		if node == s:
			Q = G.pop(node)
			break
	

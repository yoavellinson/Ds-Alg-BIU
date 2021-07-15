
import copy
import ast
import numpy as np
# finding  the negative cycles
# edge in graph
class Edge:
    def __init__(self):
        self.src = 0
        self.dest = 0
        self.weight = 0


# Structure to represent a directed
# and weighted graph
class Graph:

    def __init__(self):
        # V. Number of vertices, E.
        # Number of edges
        self.V = 0
        self.E = 0

        # Graph is represented as
        # an array of edges.
        self.edge = []


# Creates a new graph with V vertices
# and E edges
def createGraph(V, E):
    graph = Graph();
    graph.V = V;
    graph.E = E;
    graph.edge = [Edge() for i in range(graph.E)]
    return graph;


# Function runs Bellman-Ford algorithm
# and prints negative cycle(if present)
def NegCycleBellmanFord(graph, src):
    V = graph.V;
    E = graph.E;
    dist = [1000000 for i in range(V)]
    parent = [-1 for i in range(V)]
    dist[src] = 0;

    # Relax all edges |V| - 1 times.
    for i in range(1, V):
        for j in range(E):

            u = graph.edge[j].src;
            v = graph.edge[j].dest;
            weight = graph.edge[j].weight;

            if (dist[u] != 1000000 and
                    dist[u] + weight < dist[v]):
                dist[v] = dist[u] + weight;
                parent[v] = u;

    # Check for negative-weight cycles
    C = -1;
    for i in range(E):
        u = graph.edge[i].src;
        v = graph.edge[i].dest;
        weight = graph.edge[i].weight;

        if (dist[u] != 1000000 and
                dist[u] + weight < dist[v]):
            # Store one of the vertex of
            # the negative weight cycle
            C = v;
            break;

    if (C != -1):
        for i in range(V):
            C = parent[C];

        # To store the cycle vertex
        cycle = []
        v = C

        while (True):
            cycle.append(v)
            if (v == C and len(cycle) > 1):
                break;
            v = parent[v]
        '''
        the folowing loop is specific for a praph with 15 nodes, returns how many nodes can get to the negative cycles.
        '''
        print(cycle)
        num_of_nodes = 0
        for i in (set(cycle)):
            for j in range(15):
                if matrix_check[i][j] != 0:
                    num_of_nodes += 1
                    matrix_check[i][j] = 0
        return num_of_nodes



def noSpace(str):
    str=str.replace('[ ','[')
    str=str.replace(']',']')
    str=str.replace(' ',',')
    return ast.literal_eval(str)

# Driver Code
if __name__ == '__main__':
    # Number of vertices in graph
    V = 15
    # Number of edges in graph
    E = 0
    # Given Graph
matrix = noSpace("[[ 0 0 0 1 0 0 0 0 3 0 0 0 0 0 0] [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] [ 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0] [ 0 0 0 0 0 3 2 0 0 0 -5 0 0 0 0] [ 0 0 0 0 0 0 5 3 0 0 5 0 3 4 0] [ 0 0 1 0 0 0 0 0 0 0 0 0 0 -2 0] [ 1 0 5 2 0 -3 0 0 0 0 3 0 0 0 -4] [ 0 0 4 -2 0 5 0 0 3 1 0 0 0 0 0] [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5] [ 0 0 0 -4 0 0 0 0 0 0 3 0 0 0 0] [ 4 0 0 0 0 0 0 -3 4 0 0 0 0 0 2] [ 0 0 0 0 0 0 0 0 0 0 0 0 0 -3 0] [ 0 3 0 0 0 0 0 0 4 0 0 0 0 0 0] [-1 0 0 0 0 0 1 0 0 0 1 5 0 0 0] [ 0 0 0 0 1 -1 1 0 0 0 0 0 0 0 0]]")
matrix_check = matrix
print(np.array(matrix_check))
for i in range(len(matrix)):
     for j in range(len(matrix[0])):
         if matrix[i][j] != 0:
             E = E +1

graph = createGraph(V, E)


counter = 0 # counting edges
for i in range(0,V):
    for j in range(0,V):
            #checking if we set already all the vertexes
         if counter == E:
             break;
         if matrix[i][j]!=0:
                graph.edge[counter].src=i;
                graph.edge[counter].dest = j;
                graph.edge[counter].weight = matrix[i][j];
                counter=counter+1;

    # Function Call
print(NegCycleBellmanFord(graph, 0))

  
"קרדיט לאיתי חן, יואב אלינסון ולאשלי הרכבי על הקוד"
'תפריט:'
'1: שורה 149'
'2: שורה 242'
'3: שורה 400'
'4: שורה 475'
import ast
import copy
from collections import defaultdict

class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.org_graph = [i[:] for i in graph]
        self.ROW = len(graph)
        self.COL = len(graph[0])


    def BFS(self, s, t, parent):


        visited = [False] * (self.ROW)

        queue = []
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:
            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    def dfs(self, graph, s, visited):
        visited[s] = True
        for i in range(len(graph)):
            if graph[s][i] > 0 and not visited[i]:
                self.dfs(graph, i, visited)

    # Returns the min-cut of the given graph
    def minCut(self, source, sink):


        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially
        while self.BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]


            max_flow += path_flow

            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        visited = len(self.graph) * [False]
        self.dfs(self.graph, s, visited)

        min_cutes_v = []
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and \
                        self.org_graph[i][j] > 0 and visited[i]:
                    print
                    str(i) + " - " + str(j)
                    min_cutes_v.append([i, j])
        return min_cutes_v

    def BFS_F(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:

                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True

        return False

    def FordFulkerson(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        while self.BFS_F(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow

def noSpace_1(str):
    str = str.replace('[ ', '[')
    str = str.replace(']', ']')
    str = str.replace(' ', ',')
    return ast.literal_eval(str)

# הכנסת שאלה 1
#ex_1
graph = Adjacemcy_matrix_G = noSpace_1("[[ 0 0 11 13 15 35 17 0 29 0] [16 0 37 0 19 19 27 17 6 39] [ 0 0 0 21 25 21 0 0 23 3] [ 0 13 0 0 22 0 0 19 17 31] [ 0 0 0 0 0 11 2 16 10 0] [ 0 0 0 36 0 0 5 0 30 0] [ 0 0 21 9 0 0 0 28 36 37] [19 0 20 0 0 20 0 0 5 0] [ 0 0 0 0 0 0 0 0 0 0] [28 0 0 0 1 8 0 16 17 0]]")
source = 0
sink = 9
new_cable = 55
graph_copy = copy.deepcopy(graph)
g = Graph(graph_copy)

min_cutes_v = g.minCut(source, sink)
print("********* ex. max flow *******")
#print("the min cut is: ", min_cutes_v)
'''for i in range(0,len(min_cutes_v)):
    index_i = min_cutes_v[i][0]
    index_j = min_cutes_v[i][1]
    print(graph[index_i][index_j])'''
graph_copy = copy.deepcopy(graph)
g = Graph(graph_copy)
#print("original max flow= ", g.FordFulkerson(source, sink))

min_max_flow = []
for i in range(0, len(min_cutes_v)):
    graph_copy = copy.deepcopy(graph)
    # cutting all the vertexes of the min cut
    for j in range(0, len(min_cutes_v)):
        index_i = min_cutes_v[j][0]
        index_j = min_cutes_v[j][1]
        graph_copy[index_i][index_j] = 0
    # replacing each time another min cut
    index_i = min_cutes_v[i][0]
    index_j = min_cutes_v[i][1]
    graph_copy[index_i][index_j] = new_cable
    g = Graph(graph_copy)
    min_max_flow.append(g.FordFulkerson(source, sink))

#print("the fixed electric matrix is:", min_max_flow)
# finding min flow
minimum = float('inf')
for i in range(0, len(min_max_flow)):
    if minimum > min_max_flow[i] and min_max_flow[i] != 0:
        minimum = min_max_flow[i]
print("the minimum max flow for G_fixed= ", minimum)
print("--------------end q 1--------------")

# QUESTION 2:
inf = float('inf')

def zeroToInf(mat):
    n = len(mat)
    for i in range(n):
        for j in range(n):
            if i != j and mat[i][j] == 0:
                mat[i][j] = inf
    return mat


def FloydWarshall(lengths):
    n = len(lengths)
    delta = [[[inf for _ in range(n)] for _ in range(n)] for _ in range(n + 1)]  # n matrices of nxn
    for i in range(n):
        if lengths[i][i] < 0:
            delta[0][i][i] = lengths[i][i]
        else:
            delta[0][i][i] = 0
    for i in range(n):
        for j in range(n):
            if lengths[i][j] != 0:
                delta[0][i][j] = lengths[i][j]

    for k in range(1, n + 1):
        for i in range(n):
            for j in range(n):
                delta[k][i][j] = min(delta[k - 1][i][j], delta[k - 1][i][k - 1] + delta[k - 1][k - 1][j])
    return delta[n][:][:]


def question2(adj):
    '''
    Input: adjacency matrix given in question
    Output: number of vertices in negative cycles
    '''
    print("\n*** QUESTION 2 ***")
    n = len(adj)
    adj = zeroToInf(adj)
    minDis = FloydWarshall(adj)
    V = []
    for i in range(n):
        if minDis[i][i] < 0:
            V.append(i)
    # V now contains vertices that are in negative cycles

    print("Number of vertices in negative cycles is: ", end="")
    print(len(V))

#שימו פה את 2
adj = noSpace_1("[[ 0 1 0 0 0 0 0 0 2 0 0 5 0 0 0] [ 0 0 0 -1 0 0 0 3 0 0 0 -5 0 4 0] [ 0 0 0 1 0 0 0 0 4 0 1 0 0 0 0] [ 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0] [ 0 0 4 0 0 0 0 0 5 0 0 4 4 0 0] [ 0 2 0 0 0 0 2 0 1 1 0 3 5 0 0] [ 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0] [ 0 3 0 1 4 0 0 0 4 -3 0 0 0 -5 0] [ 0 -2 -2 0 0 0 0 1 0 3 0 0 0 2 0] [ 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0] [ 0 0 3 0 0 0 0 0 0 5 0 0 2 0 0] [ 0 0 0 0 0 1 0 3 0 0 0 0 -4 0 0] [ 1 0 0 0 0 4 1 0 0 0 0 0 0 5 0] [ 3 1 0 2 0 0 1 0 0 0 0 0 1 0 0] [-2 0 0 0 0 5 0 -3 0 0 0 -2 0 0 0]]")
question2(adj)

print("--------------end q 2--------------")


import ast
import math
import numpy as np


def matToPerm(A):
    '''
    assuming A is a permutation matrix, returns array of length len(A) which is the permutation of rows:
    A*X causes:
    row 0 is now row S[0]
    row 1 is now row S[1]
    ...
    row n-1 is now row S[n-1]
    OUTPUT: array L
    '''
    n = len(A)
    L = [None] * n
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:  # row i was row j in I matrix
                L[j] = i
    return L


def permProduct(L1, L2):
    '''
    INPUT: 2 lists that resemble permutations: L1 and L2 which are for matrices A1,A2
    OUTPUT: list for permutation that is L1*L2
    '''
    if L2 is None:
        return None

    n = len(L1)
    ans = [None] * n
    for i in range(n):
        ans[i] = L1[L2[i]]
        # initially row i was moved to row L2[i], and now moved to L1[L2[i]]
    return ans


def add(list, tree):
    if tree.perm not in list and tree is not None:
        list.append(tree.perm)
        return tree
    else:
        return None


def enqueue(Q, node):
    if node is not None:
        Q.append(node)


class pTree:
    # children are product of self and each permutation matrix
    def __init__(self, LX, d):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.perm = LX
        self.dep = d

    def update(self, La, Lb, Lc, Ld, list):
        self.a = add(list, pTree(permProduct(La, self.perm), self.dep + 1))
        self.b = add(list, pTree(permProduct(Lb, self.perm), self.dep + 1))
        self.c = add(list, pTree(permProduct(Lc, self.perm), self.dep + 1))
        self.d = add(list, pTree(permProduct(Ld, self.perm), self.dep + 1))


def isChildEye(tree):
    boo = False
    n = len(tree.perm)
    if tree.a is not None:
        boo = boo or tree.a.perm == [*range(n)]
    if tree.b is not None:
        boo = boo or tree.b.perm == [*range(n)]
    if tree.c is not None:
        boo = boo or tree.c.perm == [*range(n)]
    if tree.d is not None:
        boo = boo or tree.d.perm == [*range(n)]
    return boo


def updateTree(root: pTree, La, Lb, Lc, Ld, list):
    n = len(root.perm)
    # using queue to update one level each time
    Q = [root]
    while Q is not None and len(Q) > 0:
        node = Q.pop(0)
        node.update(La, Lb, Lc, Ld, list)
        if isChildEye(node):
            return node.dep + 1

        enqueue(Q, node.a)
        enqueue(Q, node.b)
        enqueue(Q, node.c)
        enqueue(Q, node.d)
    return math.inf


def question3(A, B, C, D, X):
    '''
    we want to find X=A1*A2*A3*...*An (Ai are matrices A B C or D)
    => (An^-1*...*A2^-1*A1^-1)*X=I
    A,B,C,D are permutation matrices so inverse matrix is transpose
    we will check all options until:
    1. we reach I
    2. we get stuck in a loop (get to a matrix we already found)
    '''

    print("\n*** QUESTION 3 ***")
    A_t = np.transpose(A)
    B_t = np.transpose(B)
    C_t = np.transpose(C)
    D_t = np.transpose(D)

    La = matToPerm(A_t)
    Lb = matToPerm(B_t)
    Lc = matToPerm(C_t)
    Ld = matToPerm(D_t)
    Lx = matToPerm(X)

    # list of permutations we got to (as lists)
    checked = [Lx]

    root = pTree(Lx, 0)
    ans = updateTree(root, La, Lb, Lc, Ld, checked)

    print("Min number of matrices that their product is X is: ", end="")
    print(ans)


def noSpace(str):
    str = str.replace('{', '[')
    str = str.replace('}', ']')
    return ast.literal_eval(str)


'''






נרצה לכתוב את מטריצה X


נרצה לכתוב את מטריצה X
'''
A = noSpace(
    "{{0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}}")
B = noSpace(
    "{{0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1,0}, {0, 0, 0, 0, 0, 0, 1}}")
C = noSpace(
    "{{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0,  0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}}")
D = noSpace(
    "{{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}}")
#הכנס כאן את 3
X = noSpace(
    "{{0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 0, 0}}")
print("answer: ", end="")
question3(A, B, C, D, X)

print("--------------end q 3--------------")


# cities distances problem
print("\n********* ex. cities distances *******")
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph
from collections import defaultdict


class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary


    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    def KruskalMST(self, cities):
        result = []  # This will store the resultant MST
        i = 0
        e = 0
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:

            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)


            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge

        minimumCost = 0

        for u, v, weight in result:
            minimumCost += weight

        print("Minimum round-trip road", minimumCost * 2)

# 4
matrix = noSpace("{{0., 484.649, 1865.64, 1882.11, 1671.56, 1621.8, 1937.25, 2087.96, 1237.22}, {484.649, 0., 1530.37, 1387.71, 1429.98, 1127.68, 1443.19, 1663.76, 760.434}, {1865.64, 1530.37, 0., 1326.45, 379.099, 1292.7, 1380.55, 2232.65, 993.518}, {1882.11, 1387.71, 1326.45, 0., 1595.49, 262.865, 68.3604, 943.717, 723.737}, {1671.56, 1429.98, 379.099, 1595.49, 0., 1503.12, 1656.86, 2439.44, 1094.11}, {1621.8, 1127.68, 1292.7, 262.865, 1503.12, 0., 315.513, 942.56, 509.999}, {1937.25, 1443.19, 1380.55, 68.3604, 1656.86, 315.513, 0., 897.29, 791.183}, {2087.96, 1663.76, 2232.65, 943.717, 2439.44, 942.56, 897.29, 0., 1377.13}, {1237.22, 760.434, 993.518, 723.737, 1094.11, 509.999, 791.183, 1377.13, 0.}}")
cities = ["London", "Luxemburg", "Minsk", "Pristina", "Riga", "Sarajevo", "Skopje", "Valletta", "Vienna"]
v = 9
g = Graph(v)
# set the ne edges by the matrix
for i in range(0, v):
    for j in range(0, v):
        if matrix[i][j] != 0:
            g.addEdge(i, j, matrix[i][j])

g.KruskalMST(cities)
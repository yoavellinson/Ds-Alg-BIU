# # written and collect by Itai Henn
# Python3 program for the above approach
# cities distances problem
print("\n********* ex. cities distances *******")
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph
from collections import defaultdict
import ast
# Class to represent a graph
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self, cities):

        result = []  # This will store the resultant MST

        # An index variable, used for sorted edges
        i = 0

        # An index variable, used for result[]
        e = 0

        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge does't
            #  cause cycle, include it in result
            #  and increment the indexof result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge

        minimumCost = 0
        print()
        print("Welcome abord, our road trip map is:")
        for u, v, weight in result:
            minimumCost += weight
            print("%s (%d)-> %s (%d) == %d" % (cities[u],u,cities[v], v, weight))
        print()
        print("Minimum one-way road", minimumCost)
        print("Minimum round-trip road THE QUIZ ANSWER", minimumCost*2)

def noSpace(str):
    str=str.replace('{','[')
    str=str.replace('}',']')
    return ast.literal_eval(str)

'''

C={Amsterdam, Andorra la Vella, Athens, Belgrade, Bern, Dublin, Gibraltar, Helsinki, Saint Peter Port}



כדי לתכנן את הטיול דני חישב את המטריצת המרחקים בין הערים הללו


'''
# Driver code
matrix = noSpace("{{0., 1125.15, 2166.57, 1419.38, 630.207, 748.514, 1975.59, 1506.04, 614.19}, {1125.15, 0., 1950.1, 1549.37, 680.866, 1330.4, 921.301, 2526.37, 834.797}, {2166.57, 1950.1, 0., 807.109, 1664.14, 2852.61, 2583.96, 2469.22, 2453.45}, {1419.38, 1549.37, 807.109, 0., 1039.36, 2146.25, 2381.34, 1732.42, 1814.52}, {630.207, 680.866, 1664.14, 1039.36, 0., 1200.62, 1602.15, 1861.15, 792.836}, {748.514, 1330.4, 2852.61, 2146.25, 1200.62, 0., 1908.21, 2019.89, 497.145}, {1975.59, 921.301, 2583.96, 2381.34, 1602.15, 1908.21, 0., 3434.51, 1498.14}, {1506.04, 2526.37, 2469.22, 1732.42, 1861.15, 2019.89, 3434.51, 0., 2106.54}, {614.19, 834.797, 2453.45, 1814.52, 792.836, 497.145, 1498.14, 2106.54, 0.}}")

cities= ["Amsterdam", "Andorra la Vella", "Athens", "Belgrade", "Bern", "Dublin", "Gibraltar", "Helsinki", "Saint Peter Port"] #paste match this "city"

v = 9
g = Graph(v)
#set the ne edges by the matrix
for i in range(0,v):
    for j in range(0,v):
        if matrix[i][j]!=0:
            g.addEdge(i, j, matrix[i][j])

# Function call
g.KruskalMST(cities)

# This code is contributed by Neelam Yadav

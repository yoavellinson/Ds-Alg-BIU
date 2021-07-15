import numpy as np
from collections import defaultdict
import ast

from numpy.lib.utils import source

class Graph:

	def __init__(self, graph):
		self.graph = graph # residual graph
		self. ROW = len(graph)
		# self.COL = len(gr[0])

	'''Returns true if there is a path from source 's' to sink 't' in
	residual graph. Also fills parent[] to store the path '''

	def BFS(self, s, t, parent):

		# Mark all the vertices as not visited
		visited = [False]*(self.ROW)

		# Create a queue for BFS
		queue = []

		# Mark the source node as visited and enqueue it
		queue.append(s)
		visited[s] = True

		# Standard BFS Loop
		while queue:

			# Dequeue a vertex from queue and print it
			u = queue.pop(0)

			# Get all adjacent vertices of the dequeued vertex u
			# If a adjacent has not been visited, then mark it
			# visited and enqueue it
			for ind, val in enumerate(self.graph[u]):
				if visited[ind] == False and val > 0:
					# If we find a connection to the sink node,
					# then there is no point in BFS anymore
					# We just have to set its parent and can return true
					queue.append(ind)
					visited[ind] = True
					parent[ind] = u
					if ind == t:
						return True

		# We didn't reach sink in BFS starting
		# from source, so return false
		return False
			
	
	# Returns tne maximum flow from s to t in the given graph
	def FordFulkerson(self, source, sink):

		# This array is filled by BFS and to store path
		parent = [-1]*(self.ROW)

		max_flow = 0 # There is no flow initially

		# Augment the flow while there is path from source to sink
		while self.BFS(source, sink, parent) :

			# Find minimum residual capacity of the edges along the
			# path filled by BFS. Or we can say find the maximum flow
			# through the path found.
			path_flow = float("Inf")
			s = sink
			while(s != source):
				path_flow = min (path_flow, self.graph[parent[s]][s])
				s = parent[s]

			# Add path flow to overall flow
			max_flow += path_flow

			# update residual capacities of the edges and reverse edges
			# along the path
			v = sink
			while(v != source):
				u = parent[v]
				self.graph[u][v] -= path_flow
				self.graph[v][u] += path_flow
				v = parent[v]

		return max_flow

def noSpace(str):
    str=str.replace('[ ','[')
    str=str.replace(']',']')
    str=str.replace(' ',',')
    return ast.literal_eval(str)



# Create a graph given in the above diagram
S = 0
T = 9
E = 55
min_flow = 999999999999
source_n = []

for i in range(9):
   graph_check = noSpace("[[ 0 21 3 0 3 23 23 6 37 0] [ 0 0 1 22 9 0 0 21 25 29] [ 0 0 0 37 29 34 23 0 13 33] [17 0 0 0 22 34 18 21 0 9] [ 0 0 0 0 0 1 0 10 25 23] [ 0 14 0 0 0 0 0 0 0 18] [ 0 1 0 0 7 36 0 36 0 3] [ 0 0 30 0 0 20 0 0 0 0] [ 0 0 0 39 0 25 16 26 0 22] [38 0 0 0 0 0 0 4 0 0]]")
   for j in range(9):
	   if graph_check[j][T] != 0:
		   source_n.append(j)
		   graph_check[j][T] = 0
	   if i in range(9):
	        graph_check[i][T] = E
	        min_flow = min(min_flow,Graph(graph_check).FordFulkerson(S,T)); print(min_flow)
	   source_n.clear
print("Question 1: ")
print("the minimal max flow is:", min_flow)









	


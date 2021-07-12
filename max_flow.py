# Python program for implementation
# of Ford Fulkerson algorithm
from collections import defaultdict

# This class represents a directed graph
# using adjacency matrix representation
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


# Create a graph given in the above diagram
Adjacemcy_matrix_G = [[0,29,36,16,4,38,0,21,19,0],
[0,0,31,36,22,7,0,29,11,0],
[0,0,0,0,21,16,25,20,22,0],
[0,0,21,0,16,26,0,13,10,34],
[0,0,0,0,0,13,26,0,2,34],
[0,0,0,0,0,0,20,36,0,0],
[15,25,0,1,0,0,0,29,0,34],
[0,0,0,0,29,0,0,0,0,0],
[0,0,0,0,0,12,32,22,0,0],
[49,12,10,0,0,27,0,39,38,0]]


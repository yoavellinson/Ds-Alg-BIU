import numpy as np
from numpy.lib.function_base import sinc
from numpy.lib.utils import source
from max_flow import FordFulkerson
#שאלה,1
S = 0
V = 9
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

adj = np.array(Adjacemcy_matrix_G)

print(FordFulkerson(Adjacemcy_matrix_G,0,9))
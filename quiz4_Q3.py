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
    n=len(A)
    L=[None]*n
    for i in range(n):
        for j in range(n):
            if A[i][j]==1: # row i was row j in I matrix
                L[j]=i
    return L

def permProduct(L1,L2):
    '''
    INPUT: 2 lists that resemble permutations: L1 and L2 which are for matrices A1,A2
    OUTPUT: list for permutation that is L1*L2
    '''
    if L2 is None:
        return None
    
    n=len(L1)
    ans=[None]*n
    for i in range(n):
        ans[i]=L1[L2[i]]
        # initially row i was moved to row L2[i], and now moved to L1[L2[i]]
    return ans

def add(list,tree):
    if tree.perm not in list and tree is not None:
        list.append(tree.perm)
        return tree
    else:
        return None

def enqueue(Q,node):
    if node is not None:
        Q.append(node)

class pTree:
    # children are product of self and each permutation matrix
    def __init__(self,LX,d):
        self.a=None
        self.b=None
        self.c=None
        self.d=None
        self.perm=LX
        self.dep=d

    def update(self,La,Lb,Lc,Ld,list):
        self.a=add(list,pTree(permProduct(La,self.perm),self.dep+1))
        self.b=add(list,pTree(permProduct(Lb,self.perm),self.dep+1))
        self.c=add(list,pTree(permProduct(Lc,self.perm),self.dep+1))
        self.d=add(list,pTree(permProduct(Ld,self.perm),self.dep+1))


def isChildEye(tree):
    boo=False
    n=len(tree.perm)
    if tree.a is not None:
        boo=boo or tree.a.perm==[*range(n)]
    if tree.b is not None:
        boo=boo or tree.b.perm==[*range(n)]
    if tree.c is not None:
        boo=boo or tree.c.perm==[*range(n)]
    if tree.d is not None:
        boo=boo or tree.d.perm==[*range(n)]
    return boo

def updateTree(root : pTree,La,Lb,Lc,Ld,list):
    n=len(root.perm)
    # using queue to update one level each time
    Q=[root]
    while Q is not None and len(Q)>0:
        node=Q.pop(0)
        node.update(La,Lb,Lc,Ld,list)
        if isChildEye(node):
            return node.dep+1

        enqueue(Q,node.a)
        enqueue(Q,node.b)
        enqueue(Q,node.c)
        enqueue(Q,node.d)
    return math.inf 


def question3(A,B,C,D,X):
    '''
    we want to find X=A1*A2*A3*...*An (Ai are matrices A B C or D)
    => (An^-1*...*A2^-1*A1^-1)*X=I
    A,B,C,D are permutation matrices so inverse matrix is transpose
    we will check all options until:
    1. we reach I
    2. we get stuck in a loop (get to a matrix we already found)
    '''

    print("\n*** QUESTION 3 ***")
    A_t=np.transpose(A)
    B_t=np.transpose(B)
    C_t=np.transpose(C)
    D_t=np.transpose(D)

    La=matToPerm(A_t)
    Lb=matToPerm(B_t)
    Lc=matToPerm(C_t)
    Ld=matToPerm(D_t)
    Lx=matToPerm(X)
    
    # list of permutations we got to (as lists)
    checked=[Lx]

    root=pTree(Lx,0)
    ans=updateTree(root,La,Lb,Lc,Ld,checked)

    
    print("Min number of matrices that their product is X is: ",end="")
    print(ans)

def noSpace(str):
    str=str.replace('{','[')
    str=str.replace('}',']')
    return ast.literal_eval(str)

'''







נרצה לכתוב את מטריצה X

'''

A = noSpace("{{0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}}")
B = noSpace("{{0, 0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1,0}, {0, 0, 0, 0, 0, 0, 1}}")
C = noSpace("{{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0,  0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}}")
D = noSpace("{{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}}")
X = noSpace("{{0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0}}")
print("answer: ",end="")
question3(A,B,C,D,X)


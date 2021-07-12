def bubble_sort(my_list):
    print(my_list)
    flag = True
    while flag:
     flag = False
     for i in range(len(my_list)-1):
        if my_list[i] > my_list[i+1]:
            my_list = swap(i,i+1,my_list)
            flag = True
            print(my_list)



def swap(x,y,List):
    temp = List[x]
    List[x] = List[y]
    List[y] = temp
    return List


def counting_sort(A,Range): #A is the list , Range is the known range of numbers in A
    B = [0]*len(A)
    C = [0]*(Range+1)
    for i in A:
        C[i] += 1
    for i in range(1,(Range+1)):
        C[i] += C[i-1]
    for i in range(len(A)-1,0,-1):
        B[C[A[i]]-1] = A[i]
        C[A[i]] -= 1
    return B

    


import numpy as np

def Fib(n): #dynamic prograiimng
    d ={"curr":2,"next":3} #dict to save the two elemnts that calc the next value
    if n == 0: return 0 #three base cases for  n=0,1,2
    if n == 1: return 1
    if n == 2: return 1
    else: #every other n that is greater that 2
        for i in range(3,n): #loop to run from 3 to n that clacs the value until it reaches n
            temp = d["curr"]
            d["curr"] = d["next"]
            d["next"] += temp
    return d["curr"]


def Steps(n):
    d = {"one": 2,"two": 1}
    if n == 0: return 0 #three base cases for  n=0,1,2
    if n == 1: return 1
    if n == 2: return 2
    else:
        for i in range(3,n):
            temp = d["two"]
            d["two"] = d["one"]
            d["one"] += temp
    return d["one"]
        
def in_dict(s , d):
    counter = 0
    for key in d: #for every key in the dict
        for j in range(len(s)): 
            if d[key][0] == s[j]: #finds a cummon start
                counter += 1
                for i in range(1,len(d[key])): 
                    if d[key][i] == s[j+i]: #detrments if the whole key is in the string
                        counter += 1
            if counter == len(d[key]): #if the whole key is in the string it replaces it with zeros
                 s = s[:j]+ '0'*counter  +s[(j+counter):]
                 counter = 0 
    if s == '0'*len(s): return True #if the whole string is repleced with zeros it returns True
    else: return False

    
def best_value(rest_list,k):
    counter = 0
    d = {"max_value":0,"last_value":0}
    arr = np.array(rest_list)
    km = np.max(arr[:,0])
    road = [0]*(km+1)
    for i in rest_list:
        road[i[0]] = i[1]

    print(road)
    for j in range(km+1):
        for check in range(k):
            if (j-k-1) >= 0 and (road[j-k] == road[j-k-1]): counter += 1
            if counter >= k: 
                d["max_value"] = 0
            counter = 0
        road[j] = max(d["max_value"],road[j])
        if road[j] > d["max_value"]:
             d["max_value"] = road[j]
             for back in range(k):
                 if j-back >= 0:
                     road[j-back] = d["last_value"]
    print(road)
   

 
        
def max_profit(list,k,n):
    road = []
    for km in range(n):
        if km-k-1 >= 0: road.append(max(list[km][1], list[km-k-1][1]))
    print(road)


        
def Floyed_Marshell(W): #take w as wight matrix on a graph
    d = np.array(W)
    n = d.shape[1]
    for k in range(0,n):
         for i in range(0,n):
            for j in range(0,n):
                d[i][j]=min(d[i][j],d[i][k]+d[k][j])
    print(d)

    

Floyed_Marshell([[0, 3, 1], [3, 0, 1], [1, 1, 0]])



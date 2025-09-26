import numpy as np
import math

# PART A.

def power_iteration(A, x0):
    last_m=0
    m=max(x0, key=abs)[0]
    x = x0
    iteration = 0

    while not math.isclose(m, last_m, rel_tol=1e-14): # matching to 14 digits
        last_m = m
        iteration += 1
        x = A@x
        m=max(x, key=abs)[0]
        x=x/m
    return x,m

adjacency_matrix = np.array([[0,0,1,0,0,0],
                               [1,0,1,0,0,0],
                               [0,1,0,0,0,0],
                               [0,0,1,0,0,1],
                               [1,1,0,0,0,1],
                               [1,1,1,1,0,0]]).T 

col_sum = 1/np.sum(adjacency_matrix, axis=0)

print(adjacency_matrix*col_sum)


#x0 = np.ones((5,1), dtype=float)

#x,m = power_iteration(A,x0)



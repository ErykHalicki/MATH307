import numpy as np
import math

def power_iteration(A, x0,max_iterations):
    last_m=0
    m=max(x0, key=abs)[0]
    x = x0
    iteration = 0

    while iteration<max_iterations: 
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

inv_col_sum = 1/np.sum(adjacency_matrix, axis=0)

P = adjacency_matrix*inv_col_sum

x0 = np.ones((P.shape[0],1), dtype=float)/P.shape[0]
x = x0

# PART A
for i in range(50):
    x = np.round(P@x, 5)

print(f"Q6. A)")
print(f"P = \n{P}\n")
print(f"x(50) = \n{x}")
print("------------------------")


#PART B
dtype = [('importance', float), ('index', int)]
alpha = 0.85

M = np.ones(P.shape, dtype=float)*(1-alpha)/P.shape[0] + P*alpha
x = x0

for i in range(50):
    x = np.round(M@x, 5)

print(f"Q6. B)")
print(f"M = \n{M}\n")
print(f"x(50) = \n{x}\n")

#adding index and sorting matrix
indexed_results = np.empty(M.shape[0], dtype=dtype)
indexed_results['importance'] = x.flatten()
indexed_results['index'] = np.arange(M.shape[0])

sorted_results = np.sort(indexed_results, order='importance')[::-1].reshape(M.shape[0],1)
print(sorted_results.dtype)
print(sorted_results)
print("------------------------")

#PART C
x,m = power_iteration(M,x0,15)
print(f"Q6. C)")
print(f"----------------------")
print(f"m = {m}")
print(f"x = \n{x}")
print(f"----------------------")

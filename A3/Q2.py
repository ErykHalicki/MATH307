import numpy as np 

#Q2 part a ii
Q = np.array([[1/6**(1/2), -1/6**(1/2), -1/3**(1/2)],
              [0, 2/6**(1/2), 0],
              [2/6**(1/2), 0, 1/3**(1/2)],
              [1/6**(1/2), 1/6**(1/2), -1/3**(1/2)]], dtype=float)

P = Q@Q.T
print(np.round(Q, decimals = 2))
print(np.round(P, decimals = 2))

K = np.eye(4, dtype=float) - P

print(f"K: \n{np.round(K, decimals=2)}")

b = np.array([[1,2,0,2]],dtype=float).T

proj_b = K@b
print(np.round(proj_b, decimals=2))

#check orthogonal to all basis vectors
for i in range(3):
    print(np.round(proj_b.T@Q[:,i],decimals=2))# proj_b dot Qi



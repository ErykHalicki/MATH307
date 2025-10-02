import numpy as np

v1 = np.array([[3],[-1],[2]],dtype=float)
v2 = np.array([[1],[-1],[-2]],dtype=float)

#P = standard projection matrix of subspace W
#Px = b, where x is a vector in Rn and b is x projected onto subspace W (in Rn)
#also, b is the closest point to x in W

P = (v1@v1.T) / np.linalg.norm(v1)**2 + (v2@v2.T) / np.linalg.norm(v2)**2

print(P)

u = np.array([[-7],[6],[-3]],dtype=float)

u1 = P@u #u projected onto W, using P

u2 = u - u1

print(u1)
print(u2)

# Q = projection onto Wt (orthogonal complement of W)
# then Q = I - P
# so if we do Qu, we should get u2

Q = np.eye(3, dtype = float) - P 

print(np.round(Q@u - u2, decimals=2)) #will be all 0's since they are the same


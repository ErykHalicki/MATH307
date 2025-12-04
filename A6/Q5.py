import numpy as np

A = np.array([[6,-8,-4,5,-4],
              [2,7,-5,-6,4],
              [0,-1,-8,2,2],
              [-1,-2,4,4,-8]])

print(A)

U, S, VT = np.linalg.svd(A)

print(U)
print(A)
print(VT)

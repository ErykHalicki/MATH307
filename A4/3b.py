import numpy as np 
import math

w = np.array([[44,61,81,113,131]]).T
p = np.array([[91,98,103,110,112]]).T

ln_w = np.log(w)

W = np.hstack((np.ones((w.shape[0],1)), ln_w))
print(W)

B = np.linalg.solve(W.T@W, W.T@p)
print(B)

child_estimate = B[0,0] + B[1,0] * math.log(100)
print(child_estimate)

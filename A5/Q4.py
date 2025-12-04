import numpy as np

#Q(x) = −6x1^2 − 10x2^2 − 13x3^2 − 13x4^2 − 4x1x2 − 4x1x3 − 4x1x4 + 6x3x4

A = [[-6,-2 ,-2 , -2],
     [-2 ,-10, 0, 0],
     [-2, 0, -13,3 ],
     [-2, 0, 3, 13]]

A = np.array(A)

eigvalues,eigvectors=np.linalg.eig(A)

print(f"max(Q(x) | ||x|| = 1) = {np.max(eigvalues)}")
print(f"min(Q(x) | ||x|| = 1) = {np.min(eigvalues)}")

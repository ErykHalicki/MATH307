import numpy as np 

t = np.atleast_2d(np.arange(13)).T
p = np.array([[0, 8.8, 29.9, 62.0, 104.7, 159.1, 222.0, 294.5, 380.4, 471.1, 571.7, 686.8, 809.2]]).T

T = np.hstack((np.ones((t.shape[0],1)), t, t*t, t*t*t))
print(T)

B = np.linalg.solve(T.T@T, T.T@p)
print(B)

velocity_estimate = B.T@np.array([[1, 4.5, 4.5**2, 4.5**3]]).T

print(velocity_estimate)

# It was predictable that there would be a unique least squares solution
# because the system is over determined (more equations than variables)
# since there are 13 equations and 4 variables (B0, B1, B2, B3) 


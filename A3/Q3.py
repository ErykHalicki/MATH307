import numpy as np 

q1 = np.array([[1,0,1,1]]).T
q1 = q1 / np.linalg.norm(q1)

q2 = np.array([[-1,2,-1,2]]).T
q2 = q2 / np.linalg.norm(q2)

q3 = np.array([[-3,-3,1,2]]).T
q3 = q3 / np.linalg.norm(q3)

Q = np.concatenate((q1,q2,q3), axis=1)

print(Q)

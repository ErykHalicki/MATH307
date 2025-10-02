import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

print("Welcome to Assignment 1 for MATH307")

C = 1

temp_array = []

for i in range(5):
    temp_array.append([])
    for j in range(5):
        temp_array[i].append(0)

for i in range(5):
    for j in range(5):
        if(i == j):
            temp_array[i][j] = 1 + 2*C
        if(abs(j-i) == 1):
            temp_array[i][j] = -C

A = np.array(temp_array)

print(f"matrix A:\n {A}")

_, L, U = lu(A) #ignore P for this question, A is LU factorizable

print(f"matrix L:\n {L}")
print(f"matrix U:\n {U}")

# what is the difference between lu and lu_factor?
# is LU and object that contains both L and U together?

LU, p = lu_factor(A)

t = []
t.append(np.array([10,12,12,12,10]))
t[0].shape = (5,1)

print(f"t0:\n {t[0]}")

#Atk+1 = tk
for i in range(4):
    t.append(lu_solve((LU,p), t[i])) #Atk+1 = tk
    print(f"t{i+1}:\n {t[i+1]}")



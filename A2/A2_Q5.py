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

A = np.array([
    [245, -254, -252, -46, -224],
    [161, -168, -174, -32, -148],
    [-39, 40, 45, 7, 38],
    [27, -28, -32, -6, -26],
    [110, -113, -110, -21, -101]
])

x0 = np.ones((5,1), dtype=float)

x,m = power_iteration(A,x0)

print(f"Q5. A)")
print(f"----------------------")
print(f"m = {m}")
print(f"x = {x}")
print(f"----------------------")

lambda1 = m

#part B
B = A-np.eye(5,dtype=float)*lambda1
x,m = power_iteration(B,x0)

print(f"Q5. B)")
print(f"----------------------")
print(f"m = {m + lambda1}")
print(f"x = {x}")
print(f"----------------------")

#part C
def shifted_inverse_power_iteration(A, x0, alpha):
    y = x0
    B = A-np.eye(5,dtype=float)*alpha
    last_m=0
    m=max(x0, key=abs)[0]
    iteration = 0
    while not math.isclose(m, last_m, rel_tol=1e-14): # matching to 14 digits
        iteration+=1
        last_m = m
        x = np.linalg.solve(B,y)
        beta=max(x, key=abs)[0]
        m = alpha + 1./beta
        y = 1./beta*x

    return x,m

#part D
x,m = shifted_inverse_power_iteration(A, x0, 0)
print(f"Q5. D)")
print(f"----------------------")
print(f"m = {m}")
print(f"x = {x}")
print(f"----------------------")

#part E
x,m = shifted_inverse_power_iteration(A, x0, 7)
print(f"Q5. E)")
print(f"----------------------")
print(f"m = {m}")
print(f"x = {x}")
print(f"----------------------")



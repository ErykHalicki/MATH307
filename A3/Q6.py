import numpy as np 

arr_a = np.array([[2,3],[2,1]], dtype=float)
arr_b = np.array([[1,0,-1],[1,2,1],[-4,0,1]], dtype=float)

def qr_algorithm(A, iterations):
    Ak = A 
    for i in range(iterations):
        Q,R = np.linalg.qr(Ak)
        Ak = R@Q
    return Ak

print(f"Eigenvalues of \n{arr_a}\nare\n {np.diagonal(qr_algorithm(arr_a, 100).round(2))}")
print(f"Eigenvalues of \n{arr_b}\nare\n {np.diagonal(qr_algorithm(arr_b, 100).round(2))}")


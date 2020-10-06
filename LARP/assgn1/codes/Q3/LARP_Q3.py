import numpy as np
#import sympy
import math


def correctVec(x):
	for i in range(x.shape[0]):
		x[i] = int(round(x[i,0]))%2
	return x


n = 2

A = np.zeros([n*n, n*n], dtype=int)
for i in range(n*n):
	A[i][i] = 1
	if(i >= n):
		A[i][i-n] = 1
		A[i-n][i] = 1
	if(i < (n*n - n)):
		A[i][i+n] = 1
		A[i+n][i] = 1
	if(i % n != 0):
		A[i][i-1] = 1
		A[i-1][i] = 1
	if(i % n != n-1):
		A[i][i+1] = 1
		A[i+1][i] = 1

detA = np.linalg.det(A)

y = np.array([[1 for i in range(n*n)]])
#B = np.concatenate((A, y.T), axis=1)
#B,a = sympy.Matrix(B).rref()
#B = np.array(B)
#z = detA * B[:,n*n:n*n+1]
#B = B[:,0:n*n]
print("A: ")
print(A)
#y = [1, 1, 1, 1]
print("y: ")
print(y)

#print("After Converting to Row Reduced Echelon Form")
#print("A:")
#print(B)
#print("y:")
#print(correctVec(z).T)
#print(np.matmul(np.transpose(A),A))
x = detA * np.matmul(np.linalg.inv(A), y.T)
correctVec(x)
print("x: ")
print(x.T)

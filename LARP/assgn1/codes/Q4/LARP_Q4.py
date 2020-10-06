

import numpy as np
import math

X_1 = np.array([[1,2,6,3],[12,1,2,4],[1,10,11,30],[13,24,2,13]])
X_2 = np.array([[1,2,6,3],[2,4,12,4],[1,2,6,30],[7,26,42,13]])

def encodeWith(msg, X):
    if(len(msg)%4 != 0):
        msg_num = np.zeros(len(msg) + (X.shape[0] - len(msg)%X.shape[0]), dtype=int)
    else:
        msg_num = np.zeros(len(msg), dtype=int)
    for i in range(len(msg)):
        if(i < len(msg)):
            msg_num[i] = ord(msg[i])
        else:
            msg_num[i] = 0
    res = np.zeros(len(msg_num), dtype=int)
    result = np.zeros(len(msg), dtype=int)
    for i in range(int(len(msg_num)/4)):
        res[X.shape[0]*i:X.shape[0]*i+X.shape[0]] = np.matmul(X, msg_num[X.shape[0]*i:X.shape[0]*i+X.shape[0]])
    print(res)
    for i in range(len(msg)):
        result[i] = res[i]
    return result

def decodeWith(msg_num, X):
    n = len(msg_num)
    if(len(msg_num)%4 != 0):
        for i in range(4 - (len(msg_num)%4)):
            msg_num.append(0)
        print(msg_num)
    res1 = np.zeros(len(msg_num), dtype=float)
    res = np.linalg.inv(np.matmul(np.transpose(X), X))
    for i in range(int(len(msg_num)/4)):
        #res1[X.shape[0]*i:X.shape[0]*i+X.shape[0]] = np.matmul(np.transpose(X), msg_num[X.shape[0]*i:X.shape[0]*i+X.shape[0]])
    #res1 = np.matmul(np.transpose(X), msg_num)
        #res1[X.shape[0]*i:X.shape[0]*i+X.shape[0]] = np.matmul(res, res1[X.shape[0]*i:X.shape[0]*i+X.shape[0]])
        res1[X.shape[0]*i:X.shape[0]*i+X.shape[0]] = np.matmul(np.linalg.inv(X),msg_num[X.shape[0]*i:X.shape[0]*i+X.shape[0]])
    #print(res)
    print(res1)
    result = ''
    print(n)
    for i in range(n):
        a = chr(abs(int(round(res1[i]))))
        result += a
    return result


print("Encoding Matrix 1: ")
print(X_1)
print("Encoding Matrix 2: ")
print(X_2)
print("Encoding of LARP with Matrix 1: ")
print(encodeWith("LARP", X_1))
print("Encoding of LARP with Matrix 2: ")
print(encodeWith("LARP", X_2))
print("Encoding of CIPHER with Matrix 1: ")
print(encodeWith("CIPHER", X_1))
print("Encoding of LINEAR with Matrix 1: ")
print(encodeWith("LINEAR", X_1))
print("Encoding of LINEAR with Matrix 2: ")
print(encodeWith("LINEAR", X_2))


print(decodeWith([927, 1345, 4006, 3913], X_1))
print(decodeWith([927, 1445, 3811, 3665, 708, 1081, 1778], X_1))


print(np.matmul(X_2,np.transpose(X_2)))
print(np.linalg.inv(np.matmul(X_2,np.transpose(X_2))))






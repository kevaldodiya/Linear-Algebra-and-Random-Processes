


import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

sns.set()

dataFile = "Dataset_2_Team_31.csv"

data = []

x = []
y = []

with open(dataFile, 'r') as csvFile:
    csvreader = csv.reader(csvFile)
    for rows in csvreader:
        #print(rows)
        row = []
        if rows[0] == '':
            continue
        for field in range(11):
            if(field == 0):
                continue
            row.append(float(rows[field]))
        data.append(row)

data = np.array(data)
print("Dataset: ")
print(data)



def calculateMu(data):
    return sum(data)/len(data)

def calculateSigma(data, mu):
    n = len(data)
    res = 0.0;
    for i in data:
        res = res + (1.0/n)*(i - mu)*(i - mu)
    res = math.sqrt(res)
    return res

def sortInd(eigs, eigVs):
    for i in range(eigs.shape[0]):
        for j in range(i + 1, eigs.shape[0]):
            if(eigs[i] < eigs[j]):
                eigs[i],eigs[j] = eigs[j],eigs[i]
                eigVs[:,[i,j]] = eigVs[:,[j,i]]

def norm(x):
    tmp_sum = 0
    for i in range(x.shape[0]):
        tmp_sum += x[i]**2
    return math.sqrt(tmp_sum)




ata = np.matmul(np.transpose(data),data)
aat = np.matmul(data, np.transpose(data))

eigV_ata, eigVec_ata = np.linalg.eigh(ata)
eigV_aat, eigVec_aat = np.linalg.eigh(aat)

#data_mu = np.array([calculateMu(data[:,0]),calculateMu(data[:,1])])

sortInd(eigV_ata, eigVec_ata)
print("Sorted Squares of Singular Values: ")
print(str(eigV_ata))
print("Singular Vectors (Left): ")
print(str(eigVec_ata))


data_project = np.matmul(data, eigVec_ata[:,0:2])
data_projected = np.zeros(data_project.shape, dtype=float)
for i in range(data_projected.shape[0]):
    data_projected[i] = data_project[i]# * np.transpose(eigVec_ata[:,0])

plt.scatter(data_projected[:,0], data_projected[:,1], facecolor='none', edgecolor='r')
plt.title("Data Set Projected onto Top 2 Singular Vectors")
plt.xlabel("X1 (First Coordinate)")
plt.ylabel("X2 (Second Coordinate)")
plt.show()


per_info_lost = 0
total_info = 0
for i in range(eigV_ata.shape[0]):
    if(i >= 2):
        per_info_lost += math.sqrt(eigV_ata[i])
    total_info += math.sqrt(eigV_ata[i])

per_info_lost = per_info_lost * 100 / total_info

print("Percentage Information Lost: " + str(per_info_lost))

info_gained = np.zeros(eigV_ata.shape, dtype=float)
num_of_Vecs = 0
for i in range(eigV_ata.shape[0]):
    if(i == 0):
        info_gained[i] = math.sqrt(eigV_ata[i])
    else:
        info_gained[i] = math.sqrt(eigV_ata[i]) + info_gained[i - 1]

for i in range(info_gained.shape[0]):
    if((info_gained[i]/total_info) >= 0.9):
        num_of_Vecs = i + 1
        break

print("Number of Singular Vectors Required: " + str(num_of_Vecs))






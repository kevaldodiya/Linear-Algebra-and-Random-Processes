import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

sns.set()

dataFile = "Dataset_1_Team_31.csv"

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
        for field in range(3):
            if(field == 0):
                continue
            row.append(float(rows[field]))
        data.append(row)

data = np.array(data)
#print(data)
plt.scatter(data[:,0],data[:,1], facecolor='none', edgecolor='b')
plt.xlabel("X1 (First Coordinate)")
plt.ylabel("X2 (First Coordinate)")
plt.title("Dataset")
#plt.legend(["Data Points"])
axs = plt.gca()
axs.set_xlim([-5, 125])
axs.set_ylim([-5, 125])
plt.show()



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

def norm(x, y):
    return math.sqrt(x**2 + y**2)

def error(x, y):
    return norm(y[0] - x[0], y[1] - x[1]) ** 2




ata = np.matmul(np.transpose(data),data)
aat = np.matmul(data, np.transpose(data))

eigV_ata, eigVec_ata = np.linalg.eigh(ata)
eigV_aat, eigVec_aat = np.linalg.eigh(aat)

data_mu = np.array([calculateMu(data[:,0]),calculateMu(data[:,1])])
#origin = [0],[0]

#eigVec_ata_scaled = np.zeros(eigVec_ata.shape, dtype=float)
#eigVec_ata_scaled[:,0] = 3 * eigVec_ata[:,0]
#eigVec_ata_scaled[:,1] = 7 * eigVec_ata[:,1]
dp = plt.scatter(data[:,0],data[:,1], facecolor='none', edgecolor='b')
plt.xlabel("X1 (First Coordinate)")
plt.ylabel("X2 (First Coordinate)")
plt.title("Dataset")
ax = plt.axes()
axs = plt.gca()
axs.set_xlim([-5, 125])
axs.set_ylim([-5, 125])
print("Data Mean: ")
print(data_mu)
ar1 = ax.arrow(data_mu[1], data_mu[1], 60*eigVec_ata[1,0], 60*eigVec_ata[1,1], head_width=3, head_length=3, fc='r', ec='r', label="First Singular Vector")
ar2 = ax.arrow(data_mu[1], data_mu[1], 20*eigVec_ata[0,0], 20*eigVec_ata[0,1], head_width=3, head_length=3, fc='g', ec='g', label="Second Singular Vector")
plt.legend([dp,ar1,ar2,], ["Data Points", "First Singular Vector", "Second Singular Vector"])
#plt.quiver(*origin, eigVec_ata_scaled[0,:], eigVec_ata_scaled[1,:], color=['r','g'], scale=25)
plt.show()


ax = plt.axes()
axs = plt.gca()
axs.set_xlim([-5, 125])
axs.set_ylim([-5, 125])
ar1 = ax.arrow(data_mu[1], data_mu[1], 50*eigVec_ata[1,0], 50*eigVec_ata[1,1], head_width=3, head_length=3, fc='r', ec='r', label="First Singular Vector")
ar2 = ax.arrow(data_mu[1], data_mu[1], 10*eigVec_ata[0,0], 10*eigVec_ata[0,1], head_width=3, head_length=3, fc='g', ec='g', label="Second Singular Vector")
plt.legend([ar1,ar2,], ["First Singular Vector", "Second Singular Vector"])
plt.show()


sortInd(eigV_ata, eigVec_ata)
print("Sorted Squares of Singular Values: ")
print(str(eigV_ata))
print("Singular Vectors: ")
print(str(eigVec_ata))


data_project = np.matmul(data, eigVec_ata[:,0])
data_projected = np.zeros(data.shape, dtype=float)
for i in range(data_projected.shape[0]):
    data_projected[i] = data_project[i] * np.transpose(eigVec_ata[:,0])
plt.scatter(data[:,0],data[:,1], facecolor='none', edgecolor='b')
plt.xlabel("X1 (First Coordinate)")
plt.ylabel("X2 (First Coordinate)")
plt.title("Dataset")
plt.scatter(data_projected[:,0], data_projected[:,1], facecolor='none', edgecolor='r')
plt.legend(["Data Points", "Projected Data Points"])
axs = plt.gca()
axs.set_xlim([-5, 125])
axs.set_ylim([-5, 125])
plt.show()





total_error = 0
for i in range(data.shape[0]):
    total_error += error(data[i,:],data_projected[i,:])

print("Average Error by Dimetionality Reduction: " + str(total_error/data.shape[0]))
print("Percentage Information Lost: " + str((math.sqrt(eigV_ata[1])*100)/(math.sqrt(eigV_ata[1])+math.sqrt(eigV_ata[0]))))





X = data[:,0]
Y = data[:,1]
const = np.matmul(np.transpose(X),Y)/(np.matmul(np.transpose(X),X))
print("Least Squared Weight: ")
print(const)
least_squared_solution = np.zeros(data.shape, dtype=float)
least_squared_solution[:,0] = data[:,0]
least_squared_solution[:,1] = const*data[:,0]
dp = plt.scatter(data[:,0],data[:,1], facecolor='none', edgecolor='black')
plt.xlabel("X1 (First Coordinate)")
plt.ylabel("X2 (First Coordinate)")
plt.title("Dataset")
ax = plt.axes()
axs = plt.gca()
axs.set_xlim([-5, 125])
axs.set_ylim([-5, 125])
print("Data Mean: ")
print(data_mu)
lss = plt.scatter(X, const*X, facecolor='none', edgecolor='green')
ar1 = ax.arrow(data_mu[1], data_mu[1], 60*eigVec_ata[1,0], 60*eigVec_ata[1,1], head_width=5, head_length=3, fc='r', ec='r', label="First Singular Vector")
#ar2 = ax.arrow(data_mu[1], data_mu[1], 20*eigVec_ata[0,0], 20*eigVec_ata[0,1], head_width=3, head_length=3, fc='g', ec='g', label="Second Singular Vector")
plt.legend([dp,lss,ar1,], ["Data Points", "Least Squared Solution", "First Singular Vector",])#plt.scatter(data_projected[:,0], data_projected[:,1], facecolor='none', edgecolor='r')
plt.show()



total_error_least_squared = 0
for i in range(data.shape[0]):
    total_error_least_squared += error(data[i,:],least_squared_solution[i,:])
print("Average Error by Least Squared: " + str(total_error_least_squared/data.shape[0]))


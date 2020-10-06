import numpy as np
import matplotlib.pyplot as plt
import math

n = 1000

def generateBernoulli(n, mu):
	samples = np.zeros([n, 1], dtype=int)

	for i in range(n):
		tmp = np.random.rand()
		if(tmp < (1 - mu)):
			tmp = 0
		else:
			tmp = 1
		samples[i] = tmp

	return samples

def generatePoisson(n, lam):
	samples = np.zeros([n, 1], dtype=float)

	for i in range(n):
		tmpj = np.random.rand()
		#print(tmpj)
		ex = 0
		tmp = math.exp(-lam)
		j = 0
		while ex < tmpj:
			j = j + 1
			if(j >= 9):
				j = int(round(np.random.rand()*8))
				break
			tmp = (tmp * (float(lam)/j))
			#print(str(j) + " " + str(ex) + " " + str(tmp))
			ex = ex + tmp

		samples[i] = j

	return samples

def generateBinomial(n, p):
	samples = np.zeros([n, 1], dtype=float)

	for i in range(n):
		tmpj = np.random.rand()
		#print(tmpj)
		ex = (1 - p) ** n
		tmp = (1 - p) ** n
		j = 0
		while j < n:
			if(ex >= tmpj):
				break
			j = j + 1
			tmp = (tmp * (p/(1 - p)) * ((n - j + 1) / j))
			#print(str(j) + " " + str(ex) + " " + str(tmp))
			ex = ex + tmp

		samples[i] = j

	return samples

def generateUniform(n, lower, upper):
	samples = np.zeros([n, 1], dtype=int)

	for i in range(n):
		tmp = int(round(np.random.rand()*(upper - lower) + lower))
		samples[i] = tmp

	return samples

def findConfidenceInterval(samples, mu, n):
	ci = 0
	count = 0
	total = samples.shape[0]
	while(count < (0.95 * total)):
		ci += 1/n
		count = np.count_nonzero(np.logical_and(samples <= (mu + ci), samples >= (mu - ci)))

	return ci, count

z = 1.960

accuracy = 1000

count_mean_in_point_zero_one = 0
count_mean_in_point_one = 0
count_mean_in_confidence_interval = 0

truemean = 0.5
var = 0.25
std_dev = math.sqrt(var)
diff_around_mean_for_confidience_interval = z * std_dev / math.sqrt(n)

samplemeans = np.zeros([10000,1], dtype=float)

for i in range(10000):
	samples_bernoulli = generateBernoulli(n, 0.5)
	samplemean = np.sum(samples_bernoulli)/n
	samplemeans[i] = samplemean
	if((samplemean >= truemean - 0.01) and (samplemean <= truemean + 0.01)):
		count_mean_in_point_zero_one += 1
	if((samplemean >= truemean - 0.1) and (samplemean <= truemean + 0.1)):
		count_mean_in_point_one += 1
	if((samplemean >= truemean - diff_around_mean_for_confidience_interval) and (samplemean <= truemean + diff_around_mean_for_confidience_interval)):
		count_mean_in_confidence_interval += 1

print(diff_around_mean_for_confidience_interval)
print(count_mean_in_confidence_interval)

print("Bernoulli Distribution: ")
diff_around_mean_for_confidience_interval, count_mean_in_confidence_interval = findConfidenceInterval(samplemeans, truemean, accuracy)
print("Confidence Interval is: [" + str(truemean) + " + " + str(diff_around_mean_for_confidience_interval) + ", " + str(truemean) + " - " + str(diff_around_mean_for_confidience_interval) + "]")

print("Number of Times Sample Mean was in [mu - 0.01, mu + 0.01]: " + str(count_mean_in_point_zero_one))
print("Number of Times Sample Mean was in [mu - 0.1, mu + 0.1]: " + str(count_mean_in_point_one))
print("Number of Times Sample Mean was in Confidence Interval: " + str(count_mean_in_confidence_interval))


plt.hist(samplemeans, 400)
plt.title("Histogram of Sample Means for Bernoulli Distribution")
plt.xlabel("Values of Sample Means")
plt.ylabel("Frequency")
plt.show()



count_mean_in_point_zero_one = 0
count_mean_in_point_one = 0
count_mean_in_confidence_interval = 0

truemean = 5
var = 5
std_dev = math.sqrt(var)
diff_around_mean_for_confidience_interval = z * std_dev / math.sqrt(n)

for i in range(10000):
	samples_poisson = np.random.poisson(5, (n, 1))
	#plt.hist(samples_poisson)
	#plt.show()
	samplemean = np.sum(samples_poisson)/n
	samplemeans[i] = samplemean
	if((samplemean >= truemean - 0.01) and (samplemean <= truemean + 0.01)):
		count_mean_in_point_zero_one += 1
	if((samplemean >= truemean - 0.1) and (samplemean <= truemean + 0.1)):
		count_mean_in_point_one += 1
	if((samplemean >= truemean - diff_around_mean_for_confidience_interval) and (samplemean <= truemean + diff_around_mean_for_confidience_interval)):
		count_mean_in_confidence_interval += 1
	#print(samplemean)

print(diff_around_mean_for_confidience_interval)
print(count_mean_in_confidence_interval)

print("Poisson Distribution: ")
diff_around_mean_for_confidience_interval, count_mean_in_confidence_interval = findConfidenceInterval(samplemeans, truemean, accuracy)
print("Confidence Interval is: [" + str(truemean) + " + " + str(diff_around_mean_for_confidience_interval) + ", " + str(truemean) + " - " + str(diff_around_mean_for_confidience_interval) + "]")

print("Number of Times Sample Mean was in [mu - 0.01, mu + 0.01]: " + str(count_mean_in_point_zero_one))
print("Number of Times Sample Mean was in [mu - 0.1, mu + 0.1]: " + str(count_mean_in_point_one))
print("Number of Times Sample Mean was in Confidence Interval: " + str(count_mean_in_confidence_interval))

plt.hist(samplemeans, 400)
plt.title("Histogram of Sample Means for Poisson Distribution")
plt.xlabel("Values of Sample Means")
plt.ylabel("Frequency")
plt.show()



count_mean_in_point_zero_one = 0
count_mean_in_point_one = 0
count_mean_in_confidence_interval = 0

truemean = 5
var = 100.0/12
std_dev = math.sqrt(var)
diff_around_mean_for_confidience_interval = z * std_dev / math.sqrt(n)

for i in range(10000):
	samples_uniform = generateUniform(n, 0, 10)
	samplemean = np.sum(samples_uniform)/n
	samplemeans[i] = samplemean
	if((samplemean >= truemean - 0.01) and (samplemean <= truemean + 0.01)):
		count_mean_in_point_zero_one += 1
	if((samplemean >= truemean - 0.1) and (samplemean <= truemean + 0.1)):
		count_mean_in_point_one += 1
	if((samplemean >= truemean - diff_around_mean_for_confidience_interval) and (samplemean <= truemean + diff_around_mean_for_confidience_interval)):
		count_mean_in_confidence_interval += 1

print(diff_around_mean_for_confidience_interval)
print(count_mean_in_confidence_interval)

print("Uniform Distribution: ")
diff_around_mean_for_confidience_interval, count_mean_in_confidence_interval = findConfidenceInterval(samplemeans, truemean, accuracy)
print("Confidence Interval is: [" + str(truemean) + " + " + str(diff_around_mean_for_confidience_interval) + ", " + str(truemean) + " - " + str(diff_around_mean_for_confidience_interval) + "]")

print("Number of Times Sample Mean was in [mu - 0.01, mu + 0.01]: " + str(count_mean_in_point_zero_one))
print("Number of Times Sample Mean was in [mu - 0.1, mu + 0.1]: " + str(count_mean_in_point_one))
print("Number of Times Sample Mean was in Confidence Interval: " + str(count_mean_in_confidence_interval))

plt.hist(samplemeans, 400)
plt.title("Histogram of Sample Means for Uniform Distribution")
plt.xlabel("Values of Sample Means")
plt.ylabel("Frequency")
plt.show()

count_mean_in_point_zero_one = 0
count_mean_in_point_one = 0
count_mean_in_confidence_interval = 0

truemean = 5
var = 5 * (1 - 5.0/n)
std_dev = math.sqrt(var)
diff_around_mean_for_confidience_interval = z * std_dev / math.sqrt(n)

for i in range(10000):
	samples_uniform = np.random.binomial(n, 5.0/n, (n, 1))
	samplemean = np.sum(samples_uniform)/n
	samplemeans[i] = samplemean
	if((samplemean >= truemean - 0.01) and (samplemean <= truemean + 0.01)):
		count_mean_in_point_zero_one += 1
	if((samplemean >= truemean - 0.1) and (samplemean <= truemean + 0.1)):
		count_mean_in_point_one += 1
	if((samplemean >= truemean - diff_around_mean_for_confidience_interval) and (samplemean <= truemean + diff_around_mean_for_confidience_interval)):
		count_mean_in_confidence_interval += 1

print(diff_around_mean_for_confidience_interval)
print(count_mean_in_confidence_interval)

print("Binomial Distribution: ")
diff_around_mean_for_confidience_interval, count_mean_in_confidence_interval = findConfidenceInterval(samplemeans, truemean, accuracy)
print("Confidence Interval is: [" + str(truemean) + " + " + str(diff_around_mean_for_confidience_interval) + ", " + str(truemean) + " - " + str(diff_around_mean_for_confidience_interval) + "]")

print("Number of Times Sample Mean was in [mu - 0.01, mu + 0.01]: " + str(count_mean_in_point_zero_one))
print("Number of Times Sample Mean was in [mu - 0.1, mu + 0.1]: " + str(count_mean_in_point_one))
print("Number of Times Sample Mean was in Confidence Interval: " + str(count_mean_in_confidence_interval))

plt.hist(samplemeans, 400)
plt.title("Histogram of Sample Means for Binomial Distribution")
plt.xlabel("Values of Sample Means")
plt.ylabel("Frequency")
plt.show()
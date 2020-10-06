import numpy as np
import matplotlib.pyplot as plt
import math

n = 10


def generateSamples(n):
	samples = np.zeros([n, 1], dtype=int)

	A = 3.0/(math.pi ** 2)
	for i in range(n):
		tmp = np.random.rand()/2.0
		tmp1 = 0
		tp = 0
		k = 0
		if(i%2 == 0):
			tmp1 = 1
		else:
			tmp1 = -1
		while(tp < tmp):
			k += 1
			tp += A/(k ** 2)
		samples[i] = int(round(tmp1 * k))

	return samples


def findConfidenceInterval(samples, mu, n):
	ci = 0
	count = 0
	total = samples.shape[0]
	while(count < (0.95 * total)):
		ci += 1/n
		count = np.count_nonzero(np.logical_and(samples <= (mu + ci), samples >= (mu - ci)))

	return ci, count



samplemeans = np.zeros([10000,1], dtype=float)
for i in range(10000):
	samples_bernoulli = generateSamples(n)
	samplemean = np.sum(samples_bernoulli)/n
	samplemeans[i] = samplemean

minSample = np.min(samplemeans)
maxSample = np.max(samplemeans)

truemean = 0
accuracy = 1000
diff_around_mean_for_confidience_interval, count_mean_in_confidence_interval = findConfidenceInterval(samplemeans, truemean, accuracy)
print("Confidence Interval is: [" + str(truemean) + " + " + str(diff_around_mean_for_confidience_interval) + ", " + str(truemean) + " - " + str(diff_around_mean_for_confidience_interval) + "]")

print("Number of Times Sample Mean was in Confidence Interval: " + str(count_mean_in_confidence_interval))

bins = [i for i in range(-100, 100)]

plt.hist(samplemeans, bins)
plt.xlim((-20, 20))
plt.title("Histogram of Sample Means for Probability Mass Distribution")
plt.xlabel("Values of Sample Means")
plt.ylabel("Frequency")
plt.show()

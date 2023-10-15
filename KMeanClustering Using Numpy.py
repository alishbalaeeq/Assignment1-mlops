import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy


class KMeanClustering:
    def __init__(self, k=3):
        self.meanPoints = None
        self.k = k
        return

    def train(self, X):
        rnd = np.random.choice(X.shape[0], self.k, replace=False)
        self.meanPoints = X[rnd, :]
        previousMean = np.zeros((self.k, X.shape[1]))
        if not np.array_equal(previousMean, self.meanPoints):
            check = True
        while check:
            dist = np.zeros((self.k, X.shape[0]))
            for idx, val in enumerate(self.meanPoints):
                dist[idx] = np.sum((X - val) ** 2, axis=1)
            Y = np.argmin(dist, axis=0)
            previousMean = deepcopy(self.meanPoints)
            for i in range(self.k):
                self.meanPoints[i, :] = np.mean(X[Y == i], axis=0)
            if not np.array_equal(previousMean, self.meanPoints):
                check = True
        return

    def predict(self, X):
        dist = np.zeros((self.k, X.shape[0]))
        for idx, val in enumerate(self.meanPoints):
            dist[idx] = np.sum((X - val) ** 2, axis=1)
        return np.argmin(dist, axis=0)


if __name__ == "__main__":
    data = pd.read_csv('iris.data', names=['PL', 'PW', 'SL', 'SW', 'C'])
    X = np.asarray(data.drop(['C'], axis=1))
    Y = np.asarray(data['C'])
    Y[Y == 'Iris-setosa'] = 0
    Y[Y == 'Iris-virginica'] = 1
    Y[Y == 'Iris-versicolor'] = 2
    data_S = train_test_split(X, Y, test_size=0.33, random_state=42)
    Xtr, xts, ytr, yts = data_S
    kmean = KMeanClustering(k=3)
    kmean.train(Xtr)

    pred = kmean.predict(xts)
    acc = np.sum(pred == yts) / float(yts.shape[0])
    print("Accuracy on testing data:", acc)

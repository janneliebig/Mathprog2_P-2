import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Kmeans:
    def __init__(self, n_clusters=3, max_iterations=300, epsilon=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.epsilon = epsilon # "Konvergenz"-Tolerenz

    def fit(self, X):

        # Zufällige Wahl der n_clusters Mittelpunkte
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iterations):
            # Labels setzen
            self.labels = self.set_labels(X)
            # Neue Mittelpunkte bestimmen
            new_centers = self.compute_centers(X)

            # epsilon 
            if np.all(np.linalg.norm(new_centers - self.centers, axis=1) < self.epsilon):
                break
            self.centers = new_centers

    def set_labels(self, X):
        # eukl. Norm für die Abstände
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        # kleinsten Abstand zurückgeben
        return np.argmin(distances, axis=1)

    def compute_centers(self, X):
        # Leeres Array für Mittelpunkte (t+1)
        centers = np.zeros((self.n_clusters, X.shape[1]))

        # Mittelwert der Summe der Vektoren
        for k in range(self.n_clusters):
            centers[k] = X[self.labels == k].mean(axis=0)
            
        return centers

    def predict(self, X):
        # Clusterlabel für jeden Datenpunkt
        return self.set_labels(X)

# kmeans auf Datensatz anweden
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans = Kmeans(n_clusters=4)
kmeans.fit(X)
y = kmeans.predict(X)
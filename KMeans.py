import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score


def initialize_centroids(features, k):
    return features[np.random.choice(features.shape[0], k), :]

def assign_cluster(features, centroids):
    return np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in features])

def kmeans(features, k, max_iter=10):
    np.random.seed(1)
    centroids = initialize_centroids(features, k)
    centroids_unchanged = True
    for i in range(0, max_iter):
        C = assign_cluster(features, centroids)
        old_centroids = centroids
        centroids = [features[C == k_].mean(axis = 0) for k_ in range(k)]
        centroids_unchanged = np.array_equal(old_centroids, centroids)
        if centroids_unchanged:
            print("Cluster already converged")
            break
    return np.array(C), np.array(centroids)

def calculate_accuracy(y_truth, y_predicted):
    labels = np.zeros_like(y_predicted)
    for i in range(3):
        mask = (y_predicted == i)
        labels[mask] = mode(y_truth[mask])[0]
    return accuracy_score(y_truth, labels)

def main() :
    iris_df = pd.read_csv('./dataset/iris.data', names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'label'], index_col=False)
    iris_df.head()

    le = LabelEncoder()
    iris_df['label'] = le.fit_transform(iris_df['label'])

    C = initialize_centroids(iris_df.drop(['label'], axis=1).values, 3)

    clustered = kmeans(iris_df.drop(['label'], axis=1).values, 3, 300)
    clustered[0]

    kmeans_model = KMeans(3, random_state=1)
    kmeans_model.fit(iris_df.drop(['label'], axis=1))
    kmeans_model.labels_
    kmeans_model.cluster_centers_

    y = iris_df['label'].values

    print("accuracy Kmeans from model sklearn : ",end="")
    print(calculate_accuracy(y, kmeans_model.labels_))

    print("accuracy Kmeans : ", end="")
    print(calculate_accuracy(y, clustered[0]))



if __name__ == "__main__":
    main()
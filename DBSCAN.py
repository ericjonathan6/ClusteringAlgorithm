import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score


def find_neighbors(data, core, eps):
    neighbors = []
    
    for i in range(len(data)):
        if np.linalg.norm(core - data[i]) < eps:
           neighbors.append(i)
            
    return neighbors

def iterate_neighbors(data, eps, min_pts, cluster, labels, core, neighbors):
    i = 0
    while i < len(neighbors):      
        neighbor = neighbors[i]
        if labels[neighbor] == -1:
           labels[neighbor] = cluster
        elif labels[neighbor] == -2:
            labels[neighbor] = cluster
            
            new_neighbors = find_neighbors(data, data[neighbor], eps)

            if len(new_neighbors) >= min_pts:
                neighbors = neighbors + new_neighbors
        i += 1

def dbscan(data, eps, min_pts):
    labels = [-2 for i in range(len(data))]
    
    cluster = 0
    
    for i in range(len(data)):
        if (labels[i] == -2):
            neighbors = find_neighbors(data, data[i], eps)
            
            if len(neighbors) < min_pts:
                labels[i] = -1   
            else:
                iterate_neighbors(data, eps, min_pts, cluster, labels, data[i], neighbors)
                cluster+=1
    
    return np.array(labels)

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

    clustered = dbscan(iris_df.drop(['label'], axis=1).values, 0.5, 5)

    features = iris_df.drop(['label'], axis=1).values
    dbscan_model = DBSCAN(min_samples=5, eps=0.5)
    dbscan_model.fit(features)

    y = iris_df['label'].values

    print("accuracy DBSCAN from model sklearn : ",end="")
    print(calculate_accuracy(y, dbscan_model.labels_))

    print("accuracy DBSCAN : ",end="")
    print(calculate_accuracy(y, clustered))

if __name__ == "__main__":
    main()
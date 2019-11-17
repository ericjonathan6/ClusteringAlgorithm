import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score


def single_link(features, clusters, p_a):
    new_distance = []
    list_f = features.values.tolist()
    for cluster in clusters :
        init_a = list_f[cluster[0]]
        init_b = list_f[clusters[p_a][0]]
        min_distance = np.linalg.norm(np.array(init_a)-np.array(init_b))
        for a in cluster :
            for b in clusters[p_a] :
                new_dist = np.linalg.norm(np.array(list_f[a])-np.array(list_f[b]))
                min_distance = new_dist if new_dist < min_distance else min_distance 
        new_distance.append(min_distance)
    return new_distance

def complete_link(features, clusters, p_a):
    new_distance = []
    list_f = features.values.tolist()
    for cluster in clusters :
        init_a = list_f[cluster[0]]
        init_b = list_f[clusters[p_a][0]]
        max_distance = np.linalg.norm(np.array(init_a)-np.array(init_b))
        for a in cluster :
            for b in clusters[p_a] :
                new_dist = np.linalg.norm(np.array(list_f[a])-np.array(list_f[b]))
                max_distance = new_dist if new_dist > max_distance else max_distance 
        new_distance.append(max_distance)
    return new_distance

def avg_list(p):
    sum_feature = [0 for i in range(len(p[0]))]
    for i in range(len(p[0])):
        for j in range(len(p)):
            sum_feature[i] = sum_feature[i] + p[j][i]
    sum_feature[:] = [x / len(p) for x in sum_feature]
    
    return sum_feature
    
def create_feature_list(features, cluster):
    feature_list = []
    list_f = features.values.tolist()

    for p in cluster:
        feature_list.append(list_f[p])
    
    return feature_list
    
def average_link(features, clusters, p_a):
    new_distance = []
    feature_list_a = create_feature_list(features, clusters[p_a])

    for cluster in clusters:
        feature_list_b = create_feature_list(features, cluster)
        distance = np.linalg.norm( np.array(avg_list(feature_list_a)) - np.array(avg_list(feature_list_b)) )
        new_distance.append(distance)

    return new_distance

def average_complete_link(features, clusters, p_a) :
    new_distance = []
    feature_list_a = create_feature_list(features, clusters[p_a])

    for cluster in clusters:
        feature_list_b = create_feature_list(features, cluster)
        distance = 0
        for i in range(len(feature_list_a)):
            for j in range(len(feature_list_b)):
                distance += np.linalg.norm( np.array(feature_list_a[i]) - np.array(feature_list_b[j]) )
                
        new_distance.append(distance / (len(feature_list_a)*len(feature_list_b)))

    return new_distance

def find_adj_matrix(features) :
    
    adj_matrix = [[0 for x in range(features.shape[0])] for y in range(features.shape[0])]
    
    for a in range(features.shape[0]) :
        for b in range(features.shape[0]) :
            if (b <= a) :
                adj_matrix[a][b] = np.linalg.norm(features.iloc[a] - features.iloc[b])
                adj_matrix[b][a] = adj_matrix[a][b]
            else :
                break
                
    return adj_matrix

def find_cluster_pair(adj_matrix) :
    idx_min_a, idx_min_b = 1,0
    closest_distance = adj_matrix [1][0]
    for i in range(len(adj_matrix)) :
        for j in range(len(adj_matrix)) :
            if (not i == j) and (j <= i) :
                if (adj_matrix[i][j] < closest_distance) :
                    closest_distance = adj_matrix[i][j]
                    idx_min_a, idx_min_b = i,j
    
    return (idx_min_a,idx_min_b, closest_distance)

def update_cluster(p_1, p_2, cluster) :
    new_cluster = cluster[p_2][:]
    new_cluster.extend(cluster[p_1])
    
    cluster.pop(p_1)
    cluster.pop(p_2)
    cluster.insert(p_2, new_cluster)
    
    return cluster

def update_adj_matrix(features, p_1, p_2, cluster, cluster_matrix, linkage) :
    cluster_matrix.pop(p_1)
    cluster_matrix.pop(p_2)

    new_distance = []
    if (linkage == 1):
        new_distance = single_link(features, cluster, p_2)
    elif (linkage == 2):
        new_distance = complete_link(features, cluster, p_2)
    elif (linkage == 3):
        new_distance = average_link(features, cluster, p_2)
    elif (linkage == 4) :
        new_distance = average_complete_link(features, cluster, p_2)
        
    cluster_matrix.insert(p_2,new_distance)
    for i in range(len(cluster_matrix)) :
        cluster_matrix[i][p_2] = new_distance[i]
    return cluster_matrix

'''
N_cluster = number of cluster

linkage (1) = Single
linkage (2) = Complete
linkage (3) = Average
linkage (4) = Average-Group

'''

def agglomerative(features, n_cluster, linkage) :

    cluster = [[i] for i in range(features.shape[0])]
    adj_matrix = find_adj_matrix(features)
    cluster_matrix = adj_matrix[:][:]
    number_cluster = len(cluster)
    while number_cluster > n_cluster :
        p_1, p_2, distance = find_cluster_pair(cluster_matrix)
        cluster = update_cluster(p_1, p_2, cluster)
        cluster_matrix = update_adj_matrix(features, p_1, p_2, cluster, cluster_matrix, linkage)
        number_cluster = len(cluster)
    return cluster

def set_label(cluster, features) :
    predicted_label = [0 for i in range(features.shape[0])]

    for i in range(len(cluster)) :
        for x in cluster[i] :
            predicted_label[x] = i
    predicted_label = np.array(predicted_label)        
    return predicted_label

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

    agg = AgglomerativeClustering(3,linkage="single")
    features = iris_df.drop(['label'], axis=1)
    agg.fit(features)

    linkage = 1
    n_cluster = 3
    cluster = agglomerative(features, n_cluster, linkage)

    predicted_label = set_label(cluster, features)

    y = iris_df['label'].values

    print("accuracy Agglomerative with Single Link from model sklearn : ",end="")
    print(calculate_accuracy(y, agg.labels_))

    print("accuracy Agglomerative with Single Link : ",end="")
    print(calculate_accuracy(y, predicted_label),end="\n\n")
    
    agg = AgglomerativeClustering(3,linkage="complete")
    agg.fit(features)

    print("accuracy Agglomerative with Complete Link from model sklearn : ",end="")
    print(calculate_accuracy(y, agg.labels_))

    linkage = 2
    cluster = agglomerative(features, n_cluster, linkage)

    predicted_label = set_label(cluster, features)

    print("accuracy Agglomerative with Complete Link : ",end="")
    print(calculate_accuracy(y, predicted_label),end="\n\n")

    agg = AgglomerativeClustering(3,linkage="average")
    agg.fit(features)

    print("accuracy Agglomerative with Average Link from model sklearn : ",end="")
    print(calculate_accuracy(y, agg.labels_))

    linkage = 3
    cluster = agglomerative(features, n_cluster, linkage)

    predicted_label = set_label(cluster, features)

    print("accuracy Agglomerative with Average Link : ",end="")
    print(calculate_accuracy(y, predicted_label),end="\n\n")

    linkage = 4
    cluster = agglomerative(features, n_cluster, linkage)

    predicted_label = set_label(cluster, features)

    print("accuracy Agglomerative with Average Complete Link : ",end="")
    print(calculate_accuracy(y, predicted_label))

if __name__ == "__main__":
    main()
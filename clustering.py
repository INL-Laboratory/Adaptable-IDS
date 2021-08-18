import numpy as np
distance_treshold = 1

assignments = []
centroids = []
cluster_label_counts = []
label_mapping = {}

def reset_clustering():
    global assignments
    global centroids
    global cluster_label_counts
    global label_mapping

    assignments = []
    centroids = []
    cluster_label_counts = []
    label_mapping = {}

def distance(x, y):
    return np.linalg.norm(x-y)

def min_distance_from_centroids(x):
    global centroids
    distances = []
    for i, center in enumerate(centroids):
        distances.append(distance(x, center))
    return np.argmin(distances), np.min(distances)

def cluster(input_data):
    global assignments
    global centroids
    res = []
    for x in input_data:
        if len(centroids) == 0:
            assignments.append(len(np.unique(assignments)))
            centroids.append(x)
        else:
            index, min_distance = min_distance_from_centroids(x)
            if min_distance < distance_treshold:
                assignments.append(index)
                centroids[index] = (centroids[index] + x) / 2
            else:
                assignments.append(len(np.unique(assignments)))
                centroids.append(x)
        res.append(assignments[-1])
    return res

def get_nearest_same_label_centeroid(labels):
    global centroids
    global label_mapping
    res = []
    indexes = []
    for label in labels:
        cluster_id = label_mapping[label][0]
        res.append(centroids[cluster_id])
    return res

def create_centroids_label_mapping(data_mapping, dataController):
    global centroids
    global cluster_label_counts
    #Assign a label to each cluster
    print("Assign labels to clusters")
    for cluster_label, center in enumerate(centroids):
        cluster_label_data = {k: v for k, v in data_mapping.items() if v==cluster_label}
        label_counts = {}
        for name, cluster in cluster_label_data.items():
            label = dataController.get_label_from_name(name)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        cluster_label_counts.append(label_counts)
        # print("Data points in cluster number", cluster_label, label_counts)

def create_label_mapping():
    global cluster_label_counts
    global label_mapping
    for cluster_id, label_counts in enumerate(cluster_label_counts):
        for label, count in label_counts.items():
            if label in label_mapping:
                if label_mapping[label][1] < count:
                    label_mapping[label] = (cluster_id, count)    
            else:
                label_mapping[label] = (cluster_id, count)

def run_clustering(model, dataController, validation_mode):
    global cluster_label_counts
    reset_clustering()
    data_mapping = model.create_mapping(dataController, validation_mode)
    create_centroids_label_mapping(data_mapping, dataController)
    # print(cluster_label_counts)
    create_label_mapping()
    return label_mapping, cluster_label_counts
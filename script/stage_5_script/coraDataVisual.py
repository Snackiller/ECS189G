import pickle
import numpy as np
import torch

link_data_path = "../../data/stage_5_data/cora/link"
node_data_path = "../../data/stage_5_data/cora/node"

id_to_index = {}
label_to_index = {}

features = []
labels = []
edge = []
with open(node_data_path,'r') as node_file:
    nodes = node_file.readlines()
    for node in nodes:
        node = node.split()
        id_to_index[node[0]] = len(id_to_index)
        features.append([int(i) for i in node[1:-1]])
        if node[-1] not in label_to_index:
            label_to_index[node[-1]] = len(label_to_index)
        labels.append(label_to_index[node[-1]])

with open(link_data_path,'r') as link_file:
    links = link_file.readlines()
    for link in links:
        origin, destination = link.split()
        edge.append([id_to_index[origin], id_to_index[destination]])
        edge.append([id_to_index[destination], id_to_index[origin]])

labels = np.array(labels)
features = np.array(features)
print(len(features),len(labels))
train_labels = []
train_features = []
test_labels = []
test_features = []
vail_labels = []
vail_features = []
for i in range(7):
    for j in np.where(labels == i)[0][:20]:
        train_labels.append(labels[j])
        train_features.append(features[j])
    for j in np.where(labels == i)[0][20:20+150]:
        test_labels.append(labels[j])
        test_features.append(features[j])
    for j in np.where(labels == i)[0][20+150:]:
        vail_labels.append(labels[j])
        vail_features.append(features[j])

edge = np.array(edge)
oriedge = edge
edge = edge[edge[:,1].argsort()]
edge = edge[edge[:,0].argsort(kind='mergesort')]
print(edge)
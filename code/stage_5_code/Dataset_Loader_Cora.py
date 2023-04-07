'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import random


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_link_file_name = None
    dataset_source_node_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        id_to_index = {}
        label_to_index = {}

        features = []
        labels = []
        edge = []
        with open(self.dataset_source_folder_path + self.dataset_source_node_file_name, 'r') as node_file:
            nodes = node_file.readlines()
            for node in nodes:
                node = node.split()
                id_to_index[node[0]] = len(id_to_index)
                features.append([int(i) for i in node[1:-1]])
                if node[-1] not in label_to_index:
                    label_to_index[node[-1]] = len(label_to_index)
                labels.append(label_to_index[node[-1]])

        with open(self.dataset_source_folder_path + self.dataset_source_link_file_name, 'r') as link_file:
            links = link_file.readlines()
            for link in links:
                origin, destination = link.split()
                edge.append([id_to_index[origin], id_to_index[destination]])
                edge.append([id_to_index[destination], id_to_index[origin]])

        labels = np.array(labels)
        features = np.array(features)
        train_mask = []

        test_mask = []

        valid_mask = []


        edge = np.array(edge)
        edge = edge[edge[:, 1].argsort()]
        edge = edge[edge[:, 0].argsort(kind='mergesort')]
        for i in range(7):
            for j in np.where(labels == i)[1][:20]:
                train_mask.append(labels[j])
            for j in np.where(labels == i)[1][20:20 + 150]:
                test_mask.append(labels[j])
            for j in np.where(labels == i)[1][20 + 150:]:
                valid_mask.append(features[j])
        return {'edge': edge,
                'train_mask': train_mask,
                'test_mask': test_mask,
                'valid_mask': valid_mask,
                }

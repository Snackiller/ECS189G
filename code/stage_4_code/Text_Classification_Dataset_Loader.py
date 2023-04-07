'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import os

from code.base_class.dataset import dataset


class Text_Classification_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        f2label = {}
        train_dir = os.path.join(self.dataset_source_folder_path, 'train')
        for label, f in enumerate(os.listdir(train_dir)):
            f2label[f] = label
            self._load(X_train, y_train, os.path.join(train_dir, f), label)
        test_dir = os.path.join(self.dataset_source_folder_path, 'test')
        for f in os.listdir(test_dir):
            self._load(X_test, y_test, os.path.join(train_dir, f), f2label[f])

        X = X_train + X_test
        y = y_train + y_test

        return {'X': X, 'y': y, 'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    def _load(self, X, y, data_dir, label):
        for file in os.listdir(data_dir):
            file = os.path.join(data_dir, file)
            with open(file) as fin:
                for line in fin:
                    X.append(line.strip())
                    y.append(label)
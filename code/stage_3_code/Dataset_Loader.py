'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt


class Dataset_Loader(dataset):
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
        if 1:
            print("@@@@@",self.dataset_source_file_name)
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
            data = pickle.load(f)

        for train_data in data['train']:
            X_train.append(train_data['image'])
            y_train.append(train_data['label'])


        for test_data in data['test']:
            X_test.append(test_data['image'])
            y_test.append(test_data['label'])


        f.close()
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

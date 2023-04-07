'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import os

from code.base_class.dataset import dataset


class Text_Generation_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X, y = [], []
        file = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        with open(file) as fin:
            for i, line in enumerate(fin):
                if i == 0:
                    continue
                x = ','.join(line.split(',')[1:])[1:-1]
                X.append(x)
                y.append(x)

        return {'X': X, 'y': y}
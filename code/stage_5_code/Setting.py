'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Test_Data(setting):

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        # train_mask = np.array(loaded_data['train_mask'])
        # test_mask = np.array(loaded_data['test_mask'])
        # valid_mask = np.array(loaded_data['valid_mask'])

        self.method.data = loaded_data
        # run MethodModule
        # self.method.data = {'train_mask': train_mask,
        #                     'test_mask': test_mask,
        #                     'valid_mask': valid_mask,
        #                     'edge': loaded_data['edge']}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None


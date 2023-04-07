'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class Setting_Train_Test_ORL(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        X_train, X_test = loaded_data['X_train'], loaded_data['X_test']
        y_train, y_test = loaded_data['y_train'], loaded_data['y_test']

        X_train = torch.from_numpy(np.array(X_train))
        X_train = torch.permute(X_train, (0, 3, 1, 2))
        X_test = torch.from_numpy(np.array(X_test))
        X_test = torch.permute(X_test, (0, 3, 1, 2))


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        
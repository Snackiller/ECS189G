'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import *
import numpy as np
import torch

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        print("Accuracy-Score:", accuracy_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu()))
        print("Precision-Score:", precision_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu(), average='weighted'))
        print("Recall-Score:", recall_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu(), average='weighted'))
        print("F1-Score:", f1_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu(), average='weighted'))
        return accuracy_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu())

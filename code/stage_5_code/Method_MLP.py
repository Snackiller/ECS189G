'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
import torch
from torch import nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def calculate_acc(y_pred, y_true):
    count = 0
    for i, j in zip(y_pred, y_true):
        if i == j:
            count += 1
    return count / len(y_pred)

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the 2_GRU_model
    max_epoch = 200
    writer = SummaryWriter('runs/' + datetime.now().strftime("%b%d_%H-%M-%S"))
    # it defines the learning rate for gradient descent based optimizer for 2_GRU_model learning
    learning_rate = 1e-2
    weight_decay = 5e-4
    # it defines the the MLP 2_GRU_model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the 2_GRU_model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, n_features, n_classes):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(n_features, 256).cuda()
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, n_classes).cuda()
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        train_mask = self.data['train_test_val']['idx_train']
        test_mask = self.data['train_test_val']['idx_test']
        valid_mask = self.data['train_test_val']['idx_val']
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate
                                     , weight_decay=self.weight_decay
                                     )  # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(X)
            train_loss = loss_function(y_pred[train_mask], y[train_mask])
            train_loss.backward()
            optimizer.step()
            # Draw image
            train_acc = calculate_acc(y_pred[train_mask].max(1)[1], y[train_mask])
            valid_loss = loss_function(y_pred[valid_mask], y[valid_mask])
            valid_acc = calculate_acc(y_pred[valid_mask].max(1)[1], y[valid_mask])
            test_loss = loss_function(y_pred[test_mask], y[test_mask])
            test_acc = calculate_acc(y_pred[test_mask].max(1)[1], y[test_mask])
            self.writer.add_scalar('Loss/train', train_loss.item(), epoch)
            self.writer.add_scalar('Loss/test', test_loss.item(), epoch)
            self.writer.add_scalar('Loss/valid', valid_loss.item(), epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            self.writer.add_scalar('Accuracy/valid', valid_acc, epoch)
            if epoch % 5 == 0:
                print(
                    "Epoch:{} | Train Loss:{:.5f} Train Acc:{:.2f} | Valid Loss:{:.5f} Valid Acc:{:.2f} | Test Loss:{:.5f} Test Acc:{:.2f}".format(
                        epoch,
                        train_loss.item(),
                        train_acc,
                        valid_loss.item(),
                        valid_acc,
                        test_loss.item(),
                        test_acc
                    ))
            # if valid_loss.item() > self.last_valid_loss:
            #     print('Stop')
            #     break
            # else:
            #     self.last_valid_loss = valid_loss.item()

    def test(self, X):
        test_mask = self.data['train_test_val']['idx_test']
        y_pred = self.forward(X)
        return y_pred[test_mask].max(1)[1]

    def train_score(self, X, y):
        train_mask = self.data['train_test_val']['idx_train']
        y_pred = self.forward(X)
        print("Accuracy-Score:", accuracy_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu()))
        print("Precision-Score:",
              precision_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu(), average='weighted'))
        print("Recall-Score:",
              recall_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu(), average='weighted'))
        print("F1-Score:", f1_score(y_pred[train_mask].max(1)[1].cpu(), y[train_mask].cpu(), average='weighted'))
        f = open('../../result/stage_5_result/GCN_Pubmed_' + 'prediction_result' + '_Train', 'wb')
        pickle.dump({'pred_y': y_pred[train_mask].max(1)[1], 'true_y': y[train_mask]}, f)
        f.close()
        return 0

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        self.train_score(self.data['graph']['X'], self.data['graph']['y'])
        pred_y = self.test(self.data['graph']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}
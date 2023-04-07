'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from datetime import datetime
import pickle


def calculate_acc(y_pred, y_true):
    count = 0
    for i, j in zip(y_pred, y_true):
        if i == j:
            count += 1
    return count / len(y_pred)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = output + self.bias
        return output


class Method_GCN_Citeseer(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 0.01
    weight_decay = 5e-4
    writer = SummaryWriter('runs/' + datetime.now().strftime("%b%d_%H-%M-%S"))
    np.random.seed(10)
    torch.manual_seed(10)

    def __init__(self, mName, mDescription, features_n, class_n):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.conv1 = GraphConvolution(features_n, 128).cuda()
        self.conv2 = GraphConvolution(128, 128).cuda()
        self.conv3 = GraphConvolution(128, 128).cuda()
        self.conv4 = GraphConvolution(128, class_n).cuda()

    def forward(self, x, train=False):
        adj = self.data['graph']['utility']['A'].cuda()
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=train)
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=train)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=train)
        x = self.conv4(x, adj)

        return x

    def train(self, x, y):
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
            y_pred = self.forward(x, True)
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
        f = open('../../result/stage_5_result/GCN_Citeseer_' + 'prediction_result' + '_Train', 'wb')
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

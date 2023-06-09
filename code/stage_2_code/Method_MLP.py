'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_Precision import Evaluate_Precision
from code.stage_2_code.Evaluate_Recall import Evaluate_Recall
from code.stage_2_code.Evaluate_F1 import Evaluate_F1
from matplotlib import pyplot
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    #max_epoch = 880
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    #learning_rate = 0.1

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 400)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(400, 400)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(400, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_3 = nn.Softmax(dim=1)

        ######change in layer dimension
        # self.fc_layer_1 = nn.Linear(784, 500)
        # # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # self.activation_func_1 = nn.ReLU()
        #
        # self.fc_layer_2 = nn.Linear(500, 250)
        # self.activation_func_2 = nn.ReLU()
        #
        # self.fc_layer_3 = nn.Linear(250, 10)
        # # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # self.activation_func_3 = nn.Softmax(dim=1)

        ######change in hiddden layer
        # self.fc_layer_1 = nn.Linear(784, 500)
        # # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # self.activation_func_1 = nn.ReLU()
        #
        # self.fc_layer_2 = nn.Linear(500, 400)
        # self.activation_func_2 = nn.ReLU()
        #
        # self.fc_layer_3 = nn.Linear(400, 400)
        # self.activation_func_3 = nn.ReLU()
        #
        # self.fc_layer_4 = nn.Linear(400, 200)
        # self.activation_func_4 = nn.ReLU()
        #
        # self.fc_layer_5 = nn.Linear(200, 10)
        # # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # self.activation_func_5 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.rmsprop(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        #loss_function = nn.L1Loss()
        #loss_function = nn.MSELoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator_accuracy', '')
        precision_evaluator = Evaluate_Precision('training evaluator_precision', '')
        recall_evaluator = Evaluate_Recall('training evaluator_recall', '')
        f1_evaluator = Evaluate_F1('training evaluator_f1', '')

        loss = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                precision_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                recall_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                f1_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch,
                      'Accuracy:', accuracy_evaluator.evaluate(),
                      'Precision:', precision_evaluator.evaluate(),
                      'Recall:', recall_evaluator.evaluate(),
                      'F1:', f1_evaluator.evaluate(),
                      'Loss:', train_loss.item())
            loss.append(train_loss.item())
        pyplot.title('Epoch and Loss learning curve')
        pyplot.plot(loss)
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss')
        pyplot.savefig('../../result/stage_2_result/loss_plot_mlp.png')
        pyplot.clf()


    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}




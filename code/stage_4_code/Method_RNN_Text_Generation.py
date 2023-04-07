'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import gc
from collections import Counter
from code.base_class.method import method
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


class Method_RNN_Text_Generation(method):
    data = None
    vocab = None
    vocab_r = None

    use_cuda = True
    rnn_model = None
    epochs = None
    batch_size = None

    eval_mode = False

    def train(self, X, y):
        torch.manual_seed(0)

        input_seq_len = 4
        dataset = TextGenerationDataset(X + self.data['test']['X'], input_seq_len=input_seq_len)
        self.vocab = dataset.word_to_index
        self.vocab_r = dataset.index_to_word
        model = RNN_model(len(dataset.uniq_words), self.rnn_model)
        print(model)

        model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        if self.use_cuda:
            torch.backends.cudnn.enabled = True
            cudnn.benchmark = True
            model.cuda()
            criterion = criterion.cuda()

        for epoch in range(1, self.epochs+1):
            losses = AverageMeter()
            #state_h, state_c = model.init_state(input_seq_len)
            state = model.init_state(input_seq_len)
            for i, (X_batch, y_batch) in enumerate(dataloader):
                if self.use_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    if isinstance(state, tuple):
                        state_h, state_c = state
                        state = (state_h.cuda(), state_c.cuda())
                    else:
                        state = state.cuda()
                optimizer.zero_grad()
                y_pred, state = model(X_batch, state)
                loss = criterion(y_pred.transpose(1, 2), y_batch)
                losses.update(loss.data, X_batch.size(0))
                if isinstance(state, tuple):
                    state_h, state_c = state
                    state = (state_h.detach(), state_c.detach())
                else:
                    state = state.detach()
                loss.backward()
                optimizer.step()
            print('Epoch[{0}] Batch[{1}/{2}] Loss[{loss.val:.4f}] Loss-Avg[{loss.avg:.4f}]'.format(
                           epoch, i, len(dataloader), loss=losses))
            #pred_y = self.test(model, self.data['test']['X'])
            #for i in range(3):
            #    print('Training data: ' + self.data['test']['X'][i])
            #    print('Generated text: ' + pred_y[i])

        return model

    def test(self, model, X):
        pred_y = []
        for x in X:
            ws = x.split()
            text = ' '.join(ws[:3])
            pred_y.append(self.predict(model, text=text, next_words=len(ws)-3, eval_mode=self.eval_mode))
        return pred_y

    def predict(self, model, text, next_words=50, eval_mode=True):
        words = text.split(' ')
        if eval_mode:
            model.eval()

        #state_h, state_c = model.init_state(len(words))
        state = model.init_state(len(words))

        for i in range(0, next_words):
            x = torch.tensor([[self.vocab[w] for w in words[i:]]])
            if self.use_cuda:
                x = x.cuda()
                if isinstance(state, tuple):
                    state_h, state_c = state
                    state = (state_h.cuda(), state_c.cuda())
                else:
                    state = state.cuda()
            y_pred, state = model(x, state)

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.vocab_r[word_index])

        return ' '.join(words)

    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        self.eval_mode = True
        pred_y = self.test(model, self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RNN_model(nn.Module):

    def __init__(self, n_vocab, rnn_model):
        super(RNN_model, self).__init__()
        self.rnn_model = rnn_model
        self.size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        if self.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.size, hidden_size=self.size, num_layers=self.num_layers, dropout=0.2)
        elif self.rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=self.size, hidden_size=self.size, num_layers=self.num_layers, dropout=0.2)
        elif self.rnn_model == 'RNN':
            self.rnn = nn.RNN(input_size=self.size, hidden_size=self.size, num_layers=self.num_layers, dropout=0.2)
        else:
            raise
        self.fc = nn.Linear(self.size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, input_seq_len):
        if self.rnn_model == 'LSTM':
            return (torch.zeros(self.num_layers, input_seq_len, self.size),
                    torch.zeros(self.num_layers, input_seq_len, self.size))
        else:
            return torch.zeros(self.num_layers, input_seq_len, self.size)


class TextGenerationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X,
        input_seq_len,
    ):
        self.X = X
        self.input_seq_len = input_seq_len

        self.words = []
        for x in self.X:
            for w in x.split():
                self.words.append(w)
        word_counts = Counter(self.words)
        self.uniq_words = sorted(word_counts, key=word_counts.get, reverse=True)
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def __len__(self):
        return len(self.words_indexes) - self.input_seq_len

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.input_seq_len]),
            torch.tensor(self.words_indexes[index+1:index+self.input_seq_len+1]),
        )

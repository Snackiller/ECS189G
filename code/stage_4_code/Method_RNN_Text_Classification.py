'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import gc
from code.base_class.method import method
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


class Method_RNN_Text_Classification(method):
    data = None
    vocab = None

    use_cuda = True
    rnn_model = None
    epochs = None
    batch_size = None

    def train(self, X, y):
        np.random.seed(0)
        torch.manual_seed(0)

        # create vocab
        min_sample = 5
        word_count = {}
        for x in X:
            for w in x.split():
                word_count[w.lower()] = word_count.get(w.lower(), 0) + 1
        word_count_filtered = dict(filter(lambda x:  min_sample <= x[1], word_count.items()))
        self.vocab = {w: i for i, w in enumerate(['<PADDING>', '<UNK>'] + sorted(list(word_count_filtered.keys())))}

        train_loader = BatchDataLoader(X, y, self.vocab, batch_size=self.batch_size)

        model = RNN_model(vocab_size=len(self.vocab), num_output=len(set(y)), rnn_model=self.rnn_model)
        print(model)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        if self.use_cuda:
            torch.backends.cudnn.enabled = True
            cudnn.benchmark = True
            model.cuda()
            criterion = criterion.cuda()

        for epoch in range(1, self.epochs + 1):
            losses = AverageMeter()
            acc = AverageMeter()
            model.train()
            for i, (X_batch, y_batch, input_seq_len) in enumerate(train_loader):
                if self.use_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                output = model(X_batch, input_seq_len)
                loss = criterion(output, y_batch)
                losses.update(loss.data, X_batch.size(0))
                acc.update(accuracy(output.data, y_batch), X_batch.size(0))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                if i > 0 and i % 10 == 0:
                    print('Epoch[{0}] Batch[{1}/{2}] Loss[{loss.val:.4f}] Loss-Avg[{loss.avg:.4f}] '
                          'Acc[{acc.val:.3f}] Acc-Avg[{acc.avg:.3f}]'.format(
                           epoch, i, len(train_loader), loss=losses, acc=acc))
                    gc.collect()
            # Call test to evaluate after each epoch
            self.test(model, self.data['test']['X'])
        return model

    def test(self, model, X):
        val_loader = BatchDataLoader(X, self.data['test']['y'], self.vocab, batch_size=1000, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        if self.use_cuda:
            criterion = criterion.cuda()
        acc = AverageMeter()
        model.eval()
        pred_y = []
        for i, (X_batch, y_batch, input_seq_len) in enumerate(val_loader):
            if self.use_cuda:
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
            output = model(X_batch, input_seq_len)
            loss = criterion(output, y_batch)
            acc.update(accuracy(output.data, y_batch), X_batch.size(0))
            _, pred = output.data.topk(1, 1, True, True)
            pred_y.extend(list(np.array(pred.cpu())))
        print('Test Acc[{acc.avg:.3f}]'.format(acc=acc))
        return pred_y

    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(model, self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

def accuracy(output, target):
    pred = output.topk(1, 1, True, True)[1].t()
    return pred.eq(target.view(1, -1).expand_as(pred))[:1].view(-1).float().sum(0, keepdim=True).mul_(100.0 / target.size(0))[0]


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

    def __init__(self, vocab_size, num_output, rnn_model='LSTM'):
        super(RNN_model, self).__init__()
        embed_size = 50
        hidden_size = 128
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_dropout = nn.Dropout(p=0.6)
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=2, dropout=0.5,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=2, dropout=0.5,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'RNN':
            self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=2, dropout=0.5,
                                batch_first=True, bidirectional=True)
        else:
            raise
        self.batchnorm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_output)

    def forward(self, x, input_seq_len):
        embed = self.encoder_dropout(self.encoder(x))
        rnn_input = pack_padded_sequence(embed, input_seq_len.cpu().numpy(), batch_first=True)
        rnn_output, _ = pad_packed_sequence(self.rnn(rnn_input, None)[0], batch_first=True)
        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = input_seq_len - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()
        return self.fc(self.batchnorm(rnn_output[row_indices, col_indices, :]))


class BatchDataLoader(object):

    def __init__(self, X, y, vocab, batch_size, shuffle=True):
        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = []
        for i, x in enumerate(X):
            label = y[i]
            x = [self.vocab.get(w.lower(), self.vocab['<UNK>']) for w in x.split()]
            self.data.append([int(label), x])
        self.n_batches = int(len(self.data) / self.batch_size)
        self.indices = list(range(len(self.data)))
        self._shuffle()

    def _shuffle(self):
        if self.shuffle:
            self.indices = np.random.permutation(len(self.data))
        self.index = 0
        self.batch_index = 0

    def _batch(self):
        batch = []
        for n in range(0, self.batch_size):
            batch.append(self.data[self.indices[self.index]])
            self.index += 1
        self.batch_index += 1
        y, x = tuple(zip(*batch))
        input_seq_len = torch.LongTensor(list(map(len, x)))
        seq_tensor = torch.zeros((len(x), input_seq_len.max())).long()
        for i, (s, l) in enumerate(zip(x, input_seq_len)):
            seq_tensor[i, :l] = torch.LongTensor(s)
        input_seq_len, perm_idx = input_seq_len.sort(0, descending=True)
        return seq_tensor[perm_idx], torch.LongTensor(y)[perm_idx], input_seq_len

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._shuffle()
        for i in range(self.n_batches):
            yield self._batch()

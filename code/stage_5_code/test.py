import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the Citeseer dataset
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

# Randomly sample 120 nodes (20 per class) for training and 1200 nodes (200 per class) for testing
train_mask = torch.zeros(len(dataset), dtype=torch.bool)
test_mask = torch.zeros(len(dataset), dtype=torch.bool)
print(dataset)
classes = dataset.y.max().item() + 1
for i in range(classes):
    idx = (dataset.y == i).nonzero(as_tuple=False).view(-1)
    idx_train = idx[:20]
    idx_test = idx[20:200]
    train_mask[idx_train] = True
    test_mask[idx_test] = True

# Create PyTorch data loaders for the training and testing sets
train_data = dataset[torch.where(train_mask)[0]]
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = dataset[torch.where(test_mask)[0]]
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the model, optimizer, and loss function
model = GCN(dataset.num_node_features, 16, dataset.num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Train the model
model.train()
for epoch in range(200):
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_nodes
    print('Epoch {}, Training Loss {:.4f}'.format(epoch, loss_all / len(train_data)))

# Evaluate the model on the testing set
model.eval()
correct = 0
for data in test_loader:
    output = model(data.x, data.edge_index)
    pred = output.argmax(dim=1)
    correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
acc = correct / int(test_mask.sum())
print('Test Accuracy: {:.4f}'.format(acc))

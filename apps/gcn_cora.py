import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True, device=None, dtype="float32"):
        super(GCN, self).__init__()

        # self.gcn_layer = nn.GCLayer(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.gcn_layer_1 = nn.GCLayer(in_features, hidden_features, bias=bias, device=device, dtype=dtype)
        self.gcn_layer_2 = nn.GCLayer(hidden_features, out_features, bias=bias, device=device, dtype=dtype)
        self.relu_layer = nn.ReLU()

    def forward(self, x, adj):
        # out = self.gcn_layer(x, adj)
        out = self.gcn_layer_1(x, adj)
        out = self.relu_layer(out)
        out = self.gcn_layer_2(out, adj)
        return out


def train_cora_one_epoch(x, adj, labels, model, loss_fn, opt, device, dtype):
    np.random.seed(4)

    if opt is None:
        model.eval()
    else:
        model.train()

    output_prob = model(x, adj)
    acc = np.mean(np.argmax(output_prob.numpy(), axis=1) == labels.numpy())
    loss = loss_fn(output_prob, labels)
    if opt is not None:
        loss.backward()
        opt.step()

    loss_numpy = loss.numpy()[0]

    del x
    del adj
    del loss
    del output_prob

    return acc, loss_numpy


def train_cora(model, x, adj, labels, lr=0.3, n_epochs=100, optimizer=ndl.optim.SGD, loss_fn=nn.SoftmaxLoss(), device=None, dtype="float32"):
    np.random.seed(4)
    
    opt = None
    if optimizer is not None:
        opt = optimizer(model.parameters(), lr=lr)

    accuracies = []
    losses = []

    print("Training:")
    for epoch in range(n_epochs):
        acc, loss = train_cora_one_epoch(x, adj, labels, model, loss_fn, opt, device, dtype)
        print('epoch: {}, accuracy: {}, loss: {}'.format(epoch+1, acc, loss))
        accuracies.append(acc)
        losses.append(loss)

    avg_acc = np.mean(accuracies)
    avg_loss = np.mean(losses)

    return avg_acc, avg_loss


if __name__ == '__main__':
    print('Using', sys.argv[1], 'backends')
    if sys.argv[1] == "cpu":
        device = ndl.cpu()
    elif sys.argv[1] == "cuda":
        device = ndl.cuda()
    feats, labels, _, adj = ndl.data.load_cora('./data/cora', device=device)
    model = GCN(feats.shape[1], 10, 7, device=device)
    avg_acc, avg_loss = train_cora(model, feats, adj, labels)

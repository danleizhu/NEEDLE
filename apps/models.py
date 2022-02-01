import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


def ConvBN(a, b, k, s, device):
    conv2d = nn.Conv(a, b, k, s, device=device)
    bn = nn.BatchNorm(b, device=device)
    relu = nn.ReLU()
    return nn.Sequential(conv2d, bn, relu)

def ResConvBN(a, b, k, s, device):
    conv1 = ConvBN(a, b, k, s, device)
    conv2 = ConvBN(a, b, k, s, device)
    return nn.Residual(nn.Sequential(conv1, conv2))

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.conv1 = ConvBN(3, 16, 7, 4, self.device)
        self.conv2 = ConvBN(16, 32, 3, 2, self.device)
        self.res1 = ResConvBN(32, 32, 3, 1, self.device)
        self.conv3 = ConvBN(32, 64, 3, 2, self.device)
        self.conv4 = ConvBN(64, 128, 3, 2, self.device)
        self.res2 = ResConvBN(128, 128, 3, 1, self.device)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=self.device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=self.device)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.res1(res)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.res2(res)
        res = self.flatten(res)
        res = self.linear1(res)
        res = self.relu(res)
        res = self.linear2(res)
        return res
        

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.

        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        
        self.embedding_layer = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == "rnn":
            self.sequence_layer = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            self.sequence_layer = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear_layer = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

        self.hidden_size = hidden_size

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).

        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)

        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        seq_len, bs = x.shape
        embedding = self.embedding_layer(x)
        output, h_out = self.sequence_layer(embedding, h)
        output_reshape = output.reshape((seq_len * bs, self.hidden_size))
        output_prob = self.linear_layer(output_reshape)
        return output_prob, h_out


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)

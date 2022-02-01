"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle as ndl
import needle.init as init
import needle.autograd as autograd
import numpy as np
from needle import backend_ndarray as nd


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    res = []
    mods = []
    if 'modules' in value:
        mods =  mods + list(value['modules'])
    if 'stuff' in value:
        for _, v in value['stuff'].items():
            if isinstance(v, Module):
                mods.append(v)
            elif isinstance(v, List):
                mods = mods + v
    if 'fn' in value:
        v = value['fn']
        if isinstance(v, Module):
            mods.append(v)
        elif isinstance(v, List):
            mods = mods + v
            
    if mods:
        res = res + mods
    for mod in mods:
        child_res = _child_modules(mod.__dict__)
        if child_res:
            res = res + child_res
    return res


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# class Flatten(Module):
#     """
#     Flattens the dimensions of a Tensor after the first into one dimension.

#     Input shape: (bs, s_1, ..., s_n)
#     Output shape: (bs, s_1*...*s_n)
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shape = (in_features, out_features)
        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.weight = Parameter(nd.array(np.ones(self.shape), device=nd_device), dtype=dtype, device=device)
        bound = np.sqrt(1 / in_features)
        init.uniform(self.weight, -bound, bound)
        self.has_bias = False
        if bias:
            self.has_bias = True
            self.bias = Parameter(nd.array(np.ones((out_features)), device=nd_device), dtype=dtype, device=device)
            init.uniform(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        reshape_shape = [1] * len(x.shape)
        reshape_shape[-1] = self.out_features
        reshape_shape = tuple(reshape_shape)
        broadcast_shape = list(x.shape)
        broadcast_shape[-1] = self.out_features
        broadcast_shape = tuple(broadcast_shape)
        if self.has_bias:
            return ops.matmul(x, self.weight) + ops.broadcast_to(ops.reshape(self.bias, reshape_shape), broadcast_shape)
        else:
            return ops.matmul(x, self.weight) 


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        res = None
        for module in self.modules:
            if res is None:
                res = module(x)
            else:
                res = module(res)
        return res


class SoftmaxLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        bs = x.shape[0]
        classes = x.shape[-1]
        logsoftmax = ops.logsoftmax(x)
        logsum = (x - logsoftmax) / classes
        summation = ops.summation(logsum, axes=(1,))
        one_hot = x * ops.one_hot(y, num_classes=x.shape[1], device=x.device)
        one_hot_sum = ops.summation(one_hot, axes=(1,))
        res = summation - one_hot_sum
        mean = ops.summation(res) / bs
        return mean


class BatchNorm(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.device = device
        self.weight = Parameter(nd.array(np.ones(dim), device=nd_device), dtype=dtype, device=device)
        self.bias = Parameter(nd.array(np.zeros(dim), device=nd_device), dtype=dtype, device=device)
        self.running_mean = Tensor(nd.array(np.zeros(dim), device=nd_device), dtype=dtype, requires_grad=False, device=device)
        self.running_var = Tensor(nd.array(np.ones(dim), device=nd_device), dtype=dtype, requires_grad=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        axis = tuple([i for i in range(len(x.shape)) if i != 1])
        reshape = tuple([1 if i != 1 else v for i, v in enumerate(x.shape)])
        cnt = 1.0
        for xs in axis:
            cnt *= x.shape[xs]
        if self.training:
            mean = x.sum(axis)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            mean_reshape = mean.reshape(reshape)
            mean_broadcast = mean_reshape.broadcast_to(x.shape)
            diff = x - mean_broadcast
            var = (diff ** 2 / cnt).sum(axis)
            var_unbiased = (diff.data ** 2 / (cnt - 1)).sum(axis)
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var_unbiased.data
            var_reshape = var.reshape(reshape)
            var_broadcast = var_reshape.broadcast_to(x.shape)
            eps_broadcast = ops.full(x.shape, self.eps, device=self.device)

            weight_reshape = self.weight.reshape(reshape)
            bias_reshape = self.bias.reshape(reshape)
            weight_broadcast = weight_reshape.broadcast_to(x.shape)
            bias_broadcast = bias_reshape.broadcast_to(x.shape)

            res = weight_broadcast * diff / (var_broadcast + eps_broadcast) ** 0.5 + bias_broadcast
            return res
        else:
            mean_reshape = self.running_mean.reshape(reshape)
            mean_broadcast = mean_reshape.broadcast_to(x.shape)
            diff = x - mean_broadcast
            var_reshape = self.running_var.reshape(reshape)
            var_broadcast = var_reshape.broadcast_to(x.shape)
            eps_broadcast = ops.full(x.shape, self.eps, device=self.device)

            weight_reshape = self.weight.reshape(reshape)
            bias_reshape = self.bias.reshape(reshape)
            weight_broadcast = weight_reshape.broadcast_to(x.shape)
            bias_broadcast = bias_reshape.broadcast_to(x.shape)

            res = weight_broadcast * diff / (var_broadcast + eps_broadcast) ** 0.5 + bias_broadcast
            return res


class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.weight = Parameter(np.ones(dims), dtype=dtype)
        self.bias = Parameter(np.zeros(dims), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        shape_diff = len(x.shape) - len(self.dims)
        axis = tuple(range(shape_diff, len(x.shape)))
        reshape = list(x.shape)
        for xs in axis:
            reshape[xs] = 1
        reshape = tuple(reshape)
        cnt = 1.0
        for xs in x.shape[shape_diff:]:
            cnt *= xs
        mean = ops.summation(x, axes=axis) / cnt
        mean_reshape = ops.reshape(mean, shape=reshape)
        mean_broadcast = ops.broadcast_to(mean_reshape, x.shape)
        diff = x - mean_broadcast
        var = ops.summation(ops.power_scalar(diff, 2) / cnt, axes=axis)
        var_reshape = ops.reshape(var, shape=reshape)
        var_broadcast = ops.broadcast_to(var_reshape, x.shape)
        eps_broadcast = ops.full(x.shape, self.eps)

        z = diff / ops.power_scalar(var_broadcast + eps_broadcast, 0.5)
        leading_dims = tuple(range(shape_diff))
        expand_dim_shape = list(x.shape)
        for xs in leading_dims:
            expand_dim_shape[xs] = 1
        weight_reshape = ops.reshape(self.weight, shape=expand_dim_shape)
        bias_reshape = ops.reshape(self.bias, shape=expand_dim_shape)
        weight_broadcast = ops.broadcast_to(weight_reshape, shape=x.shape)
        bias_broadcast = ops.broadcast_to(bias_reshape, shape=x.shape)
        res = ops.add(ops.multiply(weight_broadcast, z), bias_broadcast)
        return res


class Dropout(Module):
    def __init__(self, drop_prob=0.3):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            size = 1
            for xs in x.shape:
                size *= xs
            prob = [np.random.binomial(1, 1 - self.p) for i in range(size)]
            prob_adjust = [i if i == 0 else 1 / (1 - self.p) for i in prob]
            prob_resize = np.reshape(np.array(prob_adjust), x.shape)
            prob_tensor = Tensor(prob_resize)
            res = ops.multiply(x, prob_tensor)
            return res
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return ops.add(self.fn(x), x)


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Flatten(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format

    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()

        self.weight = Parameter(nd.array(np.ones((kernel_size, kernel_size, in_channels, out_channels)), device=nd_device), dtype=dtype, device=device)
        init.kaiming_uniform(self.weight)
        if bias:
            self.bias = Parameter(nd.array(np.ones((out_channels,)), device=nd_device), dtype=dtype, device=device)
            bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            init.uniform(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        _x = x.transpose((1, 2)).transpose((2, 3))
        _, H, W, _ = _x.shape
        if self.stride == 1:
            padding = self.kernel_size // 2
        else:
            padding = (self.kernel_size - self.stride + 1) // 2
        conv = ops.conv(_x, self.weight, stride=self.stride, padding=padding)
        if self.bias is None:
            return conv.transpose((2, 3)).transpose((1, 2))
        else:
            bias_reshape = self.bias.reshape((1, 1, 1, self.out_channels))
            bias_broadcast = bias_reshape.broadcast_to(conv.shape)
            res = conv + bias_broadcast
            return res.transpose((2, 3)).transpose((1, 2))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.nd_device = nd_device
        self.device = device
        self.dtype = dtype

        bound = (1 / hidden_size) ** 0.5

        self.W_ih = Parameter(nd.empty((input_size, hidden_size), device=nd_device), device=device, dtype=dtype)
        init.uniform(self.W_ih, -bound, bound)
        self.W_hh = Parameter(nd.empty((hidden_size, hidden_size), device=nd_device), device=device, dtype=dtype)
        init.uniform(self.W_hh, -bound, bound)

        self.bias_ih = None
        self.bias_hh = None
        self.bias = bias

        self.hidden_size = hidden_size
        
        self.activation_layer = None
        if nonlinearity == "tanh":
            self.activation_layer = Tanh()
        else:
            self.activation_layer = ReLU()

        if bias:
            self.bias_ih = Parameter(nd.empty((hidden_size,), device=nd_device), device=device, dtype=dtype)
            init.uniform(self.bias_ih, -bound, bound)
            self.bias_hh = Parameter(nd.empty((hidden_size,), device=nd_device), device=device, dtype=dtype)
            init.uniform(self.bias_hh, -bound, bound)
        

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            empty = nd.empty((X.shape[0], self.hidden_size), device=self.nd_device)
            empty.fill(0.0)
            h = Tensor(empty, device=self.device, dtype=self.dtype)
        linear = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            bias_ih = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(linear.shape)
            bias_hh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(linear.shape)
            linear = linear + bias_ih + bias_hh
        activation = self.activation_layer(linear)
        return activation
        


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.rnn_cells = []

        self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
        for i in range(num_layers - 1):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))

        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.nd_device = nd_device
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        hidden_size = self.hidden_size
        output = [None] * seq_len
        h_n = [None] * self.num_layers
        if h0 is None:
            empty = nd.empty((self.num_layers, X.shape[1], hidden_size), device=self.nd_device)
            empty.fill(0.0)
            h0 = Tensor(empty, device=self.device, dtype=self.dtype)

        for seq in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0 and seq == 0:
                    h_n[layer] = self.rnn_cells[layer](X[seq, :, :], h0[layer, :, :])
                elif layer == 0 and seq != 0:
                    h_n[layer] = self.rnn_cells[layer](X[seq, :, :], h_n[layer])
                elif layer != 0 and seq == 0:
                    h_n[layer] = self.rnn_cells[layer](output[seq], h0[layer, :, :])
                else:
                    h_n[layer] = self.rnn_cells[layer](output[seq], h_n[layer])
                output[seq] = h_n[layer]
        
        outputs = ops.stack(output, axis=0)
        h_ns = ops.stack(h_n, axis=0)
        return outputs, h_ns


                    
        


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.nd_device = nd_device
        self.device = device
        self.dtype = dtype

        bound = (1 / hidden_size) ** 0.5

        self.W_ih = Parameter(nd.empty((input_size, 4 * hidden_size), device=nd_device), device=device, dtype=dtype)
        init.uniform(self.W_ih, -bound, bound)
        self.W_hh = Parameter(nd.empty((hidden_size, 4 * hidden_size), device=nd_device), device=device, dtype=dtype)
        init.uniform(self.W_hh, -bound, bound)

        self.bias_ih = None
        self.bias_hh = None
        self.bias = bias

        self.hidden_size = hidden_size
        
        self.sigmoid_layer = Sigmoid()
        self.tanh_layer = Tanh()

        if bias:
            self.bias_ih = Parameter(nd.empty((4 * hidden_size,), device=nd_device), device=device, dtype=dtype)
            init.uniform(self.bias_ih, -bound, bound)
            self.bias_hh = Parameter(nd.empty((4 * hidden_size,), device=nd_device), device=device, dtype=dtype)
            init.uniform(self.bias_hh, -bound, bound)


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        if h is None:
            empty_h = nd.empty((X.shape[0], self.hidden_size), device=self.nd_device)
            empty_h.fill(0.0)
            h0 = Tensor(empty_h, device=self.device, dtype=self.dtype)
            empty_c = nd.empty((X.shape[0], self.hidden_size), device=self.nd_device)
            empty_c.fill(0.0)
            c0 = Tensor(empty_c, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        linear = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            bias_ih = self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(linear.shape)
            bias_hh = self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(linear.shape)
            linear = linear + bias_ih + bias_hh
        i = linear[:, :self.hidden_size]
        f = linear[:, self.hidden_size: 2 * self.hidden_size]
        g = linear[:, 2 * self.hidden_size: 3 * self.hidden_size]
        o = linear[:, 3 * self.hidden_size: 4 * self.hidden_size]

        desired_shape = (X.shape[0], self.hidden_size)

        if i.shape != desired_shape:
            i = i.reshape(desired_shape)
        if f.shape != desired_shape:
            f = f.reshape(desired_shape)
        if g.shape != desired_shape:
            g = g.reshape(desired_shape)
        if o.shape != desired_shape:
            o = o.reshape(desired_shape)    

        i = self.sigmoid_layer(i)
        f = self.sigmoid_layer(f)
        g = self.tanh_layer(g)
        o = self.sigmoid_layer(o)

        c = f * c0 + i * g
        h = o * self.tanh_layer(c)
        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        super().__init__()
        self.lstm_cells = []

        self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
        for i in range(num_layers - 1):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))

        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.nd_device = nd_device
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        hidden_size = self.hidden_size
        output = [None] * seq_len
        h_n = [None] * self.num_layers
        c_n = [None] * self.num_layers
        if h is None:
            empty_h = nd.empty((self.num_layers, X.shape[1], hidden_size), device=self.nd_device)
            empty_h.fill(0.0)
            h0 = Tensor(empty_h, device=self.device, dtype=self.dtype)
            empty_c = nd.empty((self.num_layers, X.shape[1], hidden_size), device=self.nd_device)
            empty_c.fill(0.0)
            c0 = Tensor(empty_c, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        for seq in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0 and seq == 0:
                    h_n[layer], c_n[layer] = self.lstm_cells[layer](X[seq, :, :], (h0[layer, :, :], c0[layer, :, :]))
                elif layer == 0 and seq != 0:
                    h_n[layer], c_n[layer] = self.lstm_cells[layer](X[seq, :, :], (h_n[layer], c_n[layer]))
                elif layer != 0 and seq == 0:
                    h_n[layer], c_n[layer] = self.lstm_cells[layer](output[seq], (h0[layer, :, :], c0[layer, :, :]))
                else:
                    h_n[layer], c_n[layer] = self.lstm_cells[layer](output[seq], (h_n[layer], c_n[layer]))
                output[seq] = h_n[layer]
        
        outputs = ops.stack(output, axis=0)
        h_ns = ops.stack(h_n, axis=0)
        c_ns = ops.stack(c_n, axis=0)
        return outputs, (h_ns, c_ns)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.nd_device = nd_device
        self.device = device
        self.dtype = dtype

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(nd.empty((num_embeddings, embedding_dim), device=nd_device), device=device, dtype=dtype)
        init.normal(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        empty = nd.empty((seq_len, bs, self.num_embeddings), device=self.nd_device)
        empty.fill(0.0)
        x_one_hot = Tensor(empty, device=self.device, dtype=self.dtype)
        for seq in range(seq_len):
            for b in range(bs):
                x_one_hot[seq, b, int(x[seq, b][0].numpy())] = Tensor(nd.array(1.0, device=self.nd_device, dtype=self.dtype), device=self.device, dtype=self.dtype)
        # x_one_hot = ops.one_hot(x, num_classes=self.num_embeddings, dtype=self.dtype, device=self.device)
        x_one_hot_reshape = x_one_hot.reshape((seq_len * bs, self.num_embeddings))
        out = x_one_hot_reshape @ self.weight
        out_reshape = out.reshape((seq_len, bs, self.embedding_dim))
        return out_reshape


class GCLayer(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        nd_device = None
        if isinstance(device, ndl.nd_backend.CUDADevice):
            nd_device = nd.cuda()
        elif isinstance(device, ndl.nd_backend.CPUDevice):
            nd_device = nd.cpu()
        self.nd_device = nd_device
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(nd.empty((in_features, out_features), device=nd_device), device=device, dtype=dtype)
        init.kaiming_uniform(self.weight)
        # init.uniform(self.weight)

        self.bias = None
        if bias:
            self.bias = Parameter(nd.empty((out_features,), device=nd_device), device=device, dtype=dtype)
            init.uniform(self.bias)

        self.out_features = out_features

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        linear = x @ self.weight
        out = adj @ linear
        if self.bias is not None:
            bias_reshape = self.bias.reshape((1, self.out_features))
            bias_broadcast = bias_reshape.broadcast_to(out.shape)
            return out + bias_broadcast
        else:
            return out
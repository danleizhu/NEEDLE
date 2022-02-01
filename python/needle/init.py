import math
import needle as ndl
from needle import backend_ndarray as nd
import numpy as np

def uniform(x, low=0.0, high=1.0):
    nd_device = None
    if isinstance(x.device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(x.device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()
    else:
        nd_device = nd.numpy_device()
    x.cached_data = nd.array(np.random.uniform(low, high, x.shape).astype("float32"), device=nd_device)


def normal(x, mean=0.0, std=1.0):
    nd_device = None
    if isinstance(x.device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(x.device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()
    else:
        nd_device = nd.numpy_device()
    x.cached_data = nd.array(np.random.normal(mean, std, x.shape).astype("float32"), device=nd_device)


def constant(x, c=0.0):
    nd_device = None
    if isinstance(x.device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(x.device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()
    else:
        nd_device = nd.numpy_device()
    x.cached_data = nd.array(np.full(x.shape, c), device=nd_device)


def ones(x):
    nd_device = None
    if isinstance(x.device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(x.device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()
    else:
        nd_device = nd.numpy_device()
    x.cached_data = nd.array(np.ones(shape=x.shape, dtype="float32"), device=nd_device)

def zeros(x):
    nd_device = None
    if isinstance(x.device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(x.device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()
    else:
        nd_device = nd.numpy_device()
    x.cached_data = nd.array(np.zeros(x.shape, dtype="float32"), device=nd_device)


def _calculate_fans(x):
    if len(x.shape) == 4:
        receptive_field = x.shape[0] * x.shape[1]
        return x.shape[2] * receptive_field, x.shape[3] * receptive_field
    else:
        return x.shape[0], x.shape[1]


def xavier_uniform(x, gain=1.0):
    fan_in, fan_out = _calculate_fans(x)
    fan = fan_in + fan_out
    a = gain * np.sqrt(6 / (fan))
    uniform(x, -a, a)


def xavier_normal(x, gain=1.0):
    fan_in, fan_out = _calculate_fans(x)
    fan = fan_in + fan_out
    std = gain * np.sqrt(2 / fan)
    normal(x, 0, std)


def kaiming_uniform(x, mode="fan_in", nonlinearity="relu"):
    fan_in, fan_out = _calculate_fans(x)
    gain = np.sqrt(2)
    if mode == 'fan_in':
        bound = gain * np.sqrt(3 / fan_in)
    else:
        bound = gain * np.sqrt(3 / fan_out)
    uniform(x, -bound, bound)    


def kaiming_normal(x, mode="fan_in", nonlinearity="relu"):
    fan_in, fan_out = _calculate_fans(x)
    gain = np.sqrt(2)
    if mode == 'fan_in':
        std = gain / np.sqrt(fan_in)
    else:
        std = gain / np.sqrt(fan_out)
    normal(x, 0, std)

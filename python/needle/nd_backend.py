"""NDDArray backed computation backend.

This backend uses cuda backend_ndarray for cached data and computation.
"""
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr
import needle.device
import numpy as np


class NDDevice(Device):
    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, device=self.nd_device)

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, device=self.nd_device)

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        array.fill(fill_value)
        return array

    def randn(self, shape, dtype, mean=0.0, std=1.0):
        return nd.array(np.random.normal(loc=mean, scale=std, size=shape).astype(dtype), device=self.nd_device)

    def randb(self, shape, dtype, ntrials=1, p=0.5):
        return nd.array(np.random.binomial(ntrials, p, size=shape).astype(dtype), device=self.nd_device)

    def randu(self, shape, dtype, low=0, high=0):
        return nd.array(np.random.uniform(low=low, high=high, size=shape).astype(dtype), device=self.nd_device)

    def one_hot(self, y, num_classes=10):
        #TODO fix this
        y_one_hot = []
        if len(y.shape) == 1:
            for i in range(y.shape[0]):
                y_one_hot.append(np.eye(num_classes)[int(y[i])])
        elif len(y.shape) == 2:
            for i in range(y.shape[0]):
                y_one_hot.append([])
                for j in range(y.shape[1]):
                    y_one_hot[-1].append(np.eye(num_classes)[int(y[i][j])])
        y_one_hot = np.array(y_one_hot)
        return nd.array(y_one_hot, device=self.nd_device)
                


    def enabled(self):
        return self.nd_device.enabled()

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.nd_compute(inputs, attrs)



class CUDADevice(NDDevice):
    def __init__(self, device_id: int = 0):
        assert device_id == 0
        self.nd_device = nd.cuda()
        self.device_id = device_id

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def __str__(self):
        return self.__repr__()


class CPUDevice(NDDevice):
    def __init__(self, device_id: int = 0):
        self.nd_device = nd.cpu()
        self.device_id = device_id

    def __repr__(self):
        return "cpu(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CPU, self.device_id)

    def __str__(self):
        return self.__repr__()



def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)


def cpu() -> CPUDevice:
    return CPUDevice()

# set default device to be cpu device.
needle.device._DEFAULT_DEVICE = CPUDevice

def register_nd_compute(name, value=None):
    """Register the compute property based on backend_ndarray
    nd computation can be shared across multiple backends.
    """
    return register_op_attr(name, "nd_compute", value)


# device specific computations
@register_nd_compute("EWiseAdd")
def add(inputs, attrs):
    return inputs[0] + inputs[1]


@register_nd_compute("AddScalar")
def add_scalar(inputs, attrs):
    return inputs[0] + attrs["scalar"]

@register_nd_compute("AddSparseScalar")
def add_sparse_scalar(inputs, attrs):
    return inputs[0].add_sparse_scalar(attrs["scalar"], attrs["indices"], attrs["shape"])

@register_nd_compute("AddSparseDense")
def add_sparse_dense(inputs, attrs):
    return inputs[0].add_sparse_dense(inputs[1], attrs["indices"], attrs["shape"])

@register_nd_compute("AddSparseSparse")
def add_sparse_sparse(inputs, attrs):
    return attrs["values"]

@register_nd_compute("EWiseMul")
def mul(inputs, attrs):
    return inputs[0] * inputs[1]


@register_nd_compute("MulScalar")
def mul(inputs, attrs):
    return inputs[0] * attrs["scalar"]


@register_nd_compute("MulSparseScalar")
def mul_sparse_scalar(inputs, attrs):
    # return inputs[0].mul_sparse_scalar(attrs["scalar"], attrs["indices"], attrs["shape"])
    return inputs[0] * attrs["scalar"]

@register_nd_compute("EwiseMulSparseDense")
def ewise_mul_sparse_dense(inputs, attrs):
    return inputs[0].ewise_mul_sparse_dense(inputs[1], attrs["indices"], attrs["shape"])

@register_nd_compute("EWiseDiv")
def divide(inputs, attrs):
    return inputs[0] / inputs[1]


@register_nd_compute("DivScalar")
def divide_scalar(inputs, attrs):
    return inputs[0] / attrs["scalar"]

@register_nd_compute("DivSparseScalar")
def div_sparse_scalar(inputs, attrs):
    return inputs[0] / attrs["scalar"]

@register_nd_compute("EwiseDivSparseDense")
def ewise_div_sparse_dense(inputs, attrs):
    return inputs[0].ewise_div_sparse_dense(inputs[1], attrs["indices"], attrs["shape"])


@register_nd_compute("PowerScalar")
def power_scalar(inputs, attrs):
    return inputs[0] ** attrs["scalar"]


@register_nd_compute("MatMul")
def matmul(inputs, attrs):
    return inputs[0] @ inputs[1]

@register_nd_compute("SparseDenseMatMul")
def sparse_dense_matmul(inputs, attrs):
    """
    returns dense matrix
    """
    return inputs[0].matmul_sparse_dense(inputs[1], attrs["indices"], attrs["shape"])

@register_nd_compute("Summation")
def summation(inputs, attrs):
    """
    Parameters:
    axes - int or tuple of ints or None

    If axes is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis.
    If axes is None, sum over all of the axes.

    Returns an array with the same shape, except with the specified axes removed.
    """
    return inputs[0].sum(attrs["axes"])


@register_nd_compute("BroadcastTo")
def broadcast_to(inputs, attrs):
    return inputs[0].broadcast_to(attrs["shape"]).compact()


@register_nd_compute("Reshape")
def reshape(inputs, attrs):
    return inputs[0].reshape(attrs["shape"]).compact()


@register_nd_compute("Negate")
def negate(inputs, attrs):
    return - inputs[0]

@register_nd_compute("NegateSparse")
def negate_sparse(inputs, attrs):
    return - inputs[0]


@register_nd_compute("Transpose")
def transpose(inputs, attrs):
    """
    Parameters:
    axes - tuple of ints or None

    If axes is a tuple of ints, permute those two axes.
    If axes is None, permutes the last two axes.
    """
    # if attrs["axes"] is None:
    #     axis = list(range(len(inputs[0].shape)))
    #     temp = axis[-1]
    #     axis[-1] = axis[-2]
    #     axis[-2] = temp
    #     axis = tuple(axis)
    # else:
    #     axis = list(range(len(inputs[0].shape)))
    #     axis[attrs["axes"][0]] = attrs["axes"][1]
    #     axis[attrs["axes"][1]] = attrs["axes"][0]
    #     axis = tuple(axis)

    # return inputs[0].permute(axis)
    new_axes = []
    dim = len(inputs[0].shape)
    for i in range(dim):
        new_axes.append(i)
    if attrs["axes"] == None:
        axis1 = dim - 2
        axis2 = dim - 1
        new_axes[axis1] = axis2
        new_axes[axis2] = axis1
    else:
        axis = attrs["axes"]
        axis1 = axis[0]
        axis2 = axis[1]
        new_axes[axis1] = axis2
        new_axes[axis2] = axis1
    new_axes = tuple(new_axes)
    return inputs[0].permute(new_axes)

@register_nd_compute("TransposeSparse")
def transpose_sparse(inputs, attrs):
    return inputs[0]


@register_nd_compute("Log")
def log(inputs, attrs):
    return inputs[0].log()


@register_nd_compute("Exp")
def exp(inputs, attrs):
    return inputs[0].exp()


@register_nd_compute("ReLU")
def relu(inputs, attrs):
    return inputs[0].relu()

@register_nd_compute("Sigmoid")
def sigmoid(inputs, attrs):
    return inputs[0].sigmoid()


@register_nd_compute("LogSoftmax")
def logsoftmax(inputs, attrs):
    """
    Computes log softmax along the last dimension of the array.
    """
    axis = len(inputs[0].shape) - 1
    shape = inputs[0].shape
    row_wise_max = inputs[0].max(axis).reshape(shape[:-1] + (1,)).broadcast_to(shape)
    norm = inputs[0] - row_wise_max
    exp = norm.exp()
    expsum = exp.sum(axis).reshape(shape[:-1] + (1,)).broadcast_to(shape)
    return norm - expsum.log()


@register_nd_compute("Tanh")
def tanh(inputs, attrs):
    return inputs[0].tanh()


@register_nd_compute("GetItem")
def get_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Returns array indexed by idxs i.e. if array A has shape (5, 3, 2),
    then the shape of the A[0, :, :] would be (3, 2).
    """
    idxs = attrs["idxs"]
    ele = inputs[0].__getitem__(idxs).compact()
    if len(ele.shape) == 1:
        return ele
    new_shape = []
    def none_slice(s):
        if isinstance(s, int):
            return False
        if s.start is None and s.stop is None and s.step is None:
            return True
        else:
            return False
    for i, dim in enumerate(ele.shape):
        if dim != 1 or none_slice(idxs[i]):
            new_shape.append(dim)
    if not new_shape:
        new_shape = [1]
    new_shape = tuple(new_shape)
    if new_shape != ele.shape:
        return ele.reshape(new_shape)
    else:
        return ele
    


@register_nd_compute("SetItem")
def set_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Sets array A at idxs with array B and returns the array.
    """
    inputs[0][attrs["idxs"]] = inputs[1]
    return inputs[0]


@register_nd_compute("Stack")
def stack(inputs, attrs):
    """
    Concatenates a sequence of arrays along a new dimension.

    Parameters:
    axis - dimension to concatenate along

    All arrays need to be of the same size.
    """
    shape = inputs[0].shape
    axis = attrs["axis"]
    new_shape = list(shape)
    new_shape.insert(axis, len(inputs))
    stacked = nd.empty(tuple(new_shape), device=inputs[0].device)
    slices = [slice(None, None, None)] * len(new_shape)
    for i, mat in enumerate(inputs):
        slices[axis] = i
        stacked[tuple(slices)] = mat
    return stacked


@register_nd_compute("Flip")
def flip(inputs, attrs):
    """
    Flips the input along specified axes.

    Parameters:
    axes - Axes to flip.
    """
    return inputs[0].flip(attrs["axes"])


@register_nd_compute("Dilate")
def dilate(inputs, attrs):
    """
    Dilates the input by a dilation factor on specified axes.
    (i.e., inserts 0s between elements)

    Parameters:
    dilation - Dilation amount (number of 0s to insert)
    axes - Axes to dilate by this amount
    """
    dilation = attrs["dilation"]
    axes = attrs["axes"]
    new_shape = list(inputs[0].shape)
    slices = [slice(None, None, None) for i in range(len(new_shape))]
    if isinstance(axes, int):
        axes = (axes,)
    for a in axes:
        if a >= len(new_shape):
            continue
        new_shape[a] *= 1 + dilation
        slices[a] = slice(None, None, 1 + dilation)
    out = nd.empty(tuple(new_shape), device=inputs[0].device)
    out.fill(0.0)
    out[tuple(slices)] = inputs[0]
    return out.compact()


@register_nd_compute("Conv")
def conv(inputs, attrs):
    """
    Multi-channel 2D convolution of two inputs (called input and weight, respectively).
    inputs[0]: "input", NHWC
    inputs[1]: "weight", (kernel_size, kernel_size, c_in, c_out)

    Parameters:
    padding - (int) Pad the HW axes of the input by this amount
    stride - (int) Stride of the convolution
    """
    padding = attrs["padding"]
    Z = inputs[0].compact().pad(((0, 0), (padding, padding), (padding, padding), (0, 0)))
    weight = inputs[1]
    stride = attrs["stride"]

    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    Ns, Hs, Ws, Cs = Z.strides
    
    mat_dim = N * ((H-K)//stride+1) * ((W-K)//stride+1)
    inner_dim = K * K * C_in

    new_shape = (N, (H-K)//stride+1, (W-K)//stride+1, K, K, C_in)
    new_strides = (Ns, Hs*stride, Ws*stride, Hs, Ws, Cs)

    A = nd.NDArray.make(new_shape, new_strides, Z.device, Z._handle, Z._offset)
    A = A.compact()
    out = A.reshape((mat_dim, inner_dim)) @ weight.reshape((inner_dim, C_out))
    out = out.reshape((N, (H-K)//stride+1, (W-K)//stride+1, C_out))
    return out

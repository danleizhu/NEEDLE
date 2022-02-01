"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy as np
from .autograd import Op, Tensor, Value, Tuple, sparse_coo_tensor
from .device import default_device

OP_TABLE = {}


def register_op(name: str, op: Op) -> Op:
    """Register an operator to the op table.

    Parameters
    ----------
    name : str
        The name of the op.

    Returns
    -------
    op : Op
        The registered op.
    """
    if name in OP_TABLE:
        raise ValueError("Op %s is already registered")
    OP_TABLE[name] = op
    return op


def register_op_attr(op_name, attr_name, attr_value=None):
    """Register additional attributes to an existing op by name.


    Parameters
    ----------
    op_name : str
        The name of the op

    attr_name : str
        The name of the attribute

    attr_value :
        The attribute value to be set.

    Returns
    -------
    The attr_value if attr_value is not None.
    Otherwise returns a decorator function.


    Note
    ----
    This function can be used to register additional attributes
    to an Op used by a specific backend.
    """

    def _register(value):
        if op_name not in OP_TABLE:
            raise ValueError("Op %s does not exist")
        op = OP_TABLE[op_name]
        setattr(op, attr_name, value)
        return op

    if attr_value is None:
        return _register
    return _register(attr_value)


class MakeTupleOp(Op):
    def __call__(self, *args: List[Value]) -> Tuple:
        return Tuple.make_from_op(self, list(args))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, Tuple)
        return [out_grad[i] for i in range(len(out_grad))]


make_tuple = register_op("MakeTuple", MakeTupleOp())


class TupleGetItemOp(Op):
    def __call__(self, a: Tuple, index: int, *, fold_const=True) -> Tensor:
        assert isinstance(a, Tuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTupleOp):
            return a.inputs[index]
        return Tensor.make_from_op(self, [a], attrs={"index": index})

    def gradient(self, out_grad, node):
        index = node.attrs["index"]
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return [make_tuple(*in_grad)]


tuple_get_item = register_op("TupleGetItem", TupleGetItemOp())


class FusedAddScalarsOp(Op):
    def __call__(self, a: Tensor, c0: float, c1: float) -> Tuple:
        return Tuple.make_from_op(self, [a], attrs={"c0": c0, "c1": c1})

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


fused_add_scalars = register_op("FusedAddScalars", FusedAddScalarsOp())


class EWiseAddOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add = register_op("EWiseAdd", EWiseAddOp())


class AddScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad]


add_scalar = register_op("AddScalar", AddScalarOp())

class AddSparseScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar, "indices": a.indices, "shape": a.shape})

    def gradient(self, out_grad, node):
        return [out_grad]


add_sparse_scalar = register_op("AddSparseScalar", AddSparseScalarOp())


class AddSparseDenseOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"indices": a.indices, "shape": a.shape})

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add_sparse_dense = register_op("AddSparseDense", AddSparseDenseOp())

class AddSparseSparseOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        indices_a = a.indices.numpy()
        indices_b = b.indices.numpy()
        values_a = a.cached_data.numpy()
        values_b = b.cached_data.numpy()
        idx_val_mapping = {}
        for cnt in range(indices_a.shape[1]):
            idx = (indices_a[0][cnt], indices_a[1][cnt])
            val = values_a[cnt]
            if idx not in idx_val_mapping:
                idx_val_mapping[idx] = 0.0
            idx_val_mapping[idx] += val
        for cnt in range(indices_b.shape[1]):
            idx = (indices_b[0][cnt], indices_b[1][cnt])
            val = values_b[cnt]
            if idx not in idx_val_mapping:
                idx_val_mapping[idx] = 0.0
            idx_val_mapping[idx] += val
        rows = []
        cols = []
        values = []
        for (row, col), val in idx_val_mapping.items():
            rows.append(row)
            cols.append(col)
            values.append(val)
        indices_numpy = np.array((rows, cols), dtype=np.float32)
        values_numpy = np.array(values, dtype=np.float32)
        indices = a.device.array(indices_numpy, dtype="float32")
        values = a.device.array(values_numpy, dtype="float32")

        return Tensor.make_from_op(self, [a, b], attrs={"values": values}, indices=indices, data_shape=a.data_shape)

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]

add_sparse_sparse = register_op("AddSparseSparse", AddSparseSparseOp())


class EWiseMulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


multiply = register_op("EWiseMul", EWiseMulOp())

class EwiseMulSparseDenseOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"indices": a.indices, "shape": a.shape})
    
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lgrad = (out_grad * rhs).to_sparse(lhs.indices, lhs.data_shape)
        rgrad = lhs * out_grad
        return (lgrad, rgrad)

ewise_mul_sparse_dense = register_op("EwiseMulSparseDense", EwiseMulSparseDenseOp())


class MulScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]


multiply_scalar = register_op("MulScalar", MulScalarOp())

class MulSparseScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar}, indices=a.indices, data_shape=a.data_shape)
        # return sparse_coo_tensor(indices=a.indices, values=a.cached_data * scalar, data_shape=a.shape)
    
    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]

mul_sparse_scalar = register_op("MulSparseScalar", MulSparseScalarOp())


class PowerScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        power = node.attrs["scalar"]
        return [out_grad * power * np.power(mat, power - 1)]


power_scalar = register_op("PowerScalar", PowerScalarOp())


class EWiseDivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        divident, divisor = node.inputs
        return [out_grad / divisor, - out_grad * divident / divisor / divisor]


divide = register_op("EWiseDiv", EWiseDivOp())

class EwiseDivSparseDenseOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"indices": a.indices, "shape": a.shape})
    
    def gradient(self, out_grad, node):
        divident, divisor = node.inputs
        lgrad = (out_grad / divisor).to_sparse(divident.indices, divident.data_shape)
        rgrad = - out_grad / divisor / divisor
        rgrad = divident * rgrad
        return [lgrad, rgrad]

ewise_div_sparse_dense = register_op("EwiseDivSparseDense", EwiseDivSparseDenseOp())


class DivScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad / node.attrs["scalar"]]


divide_scalar = register_op("DivScalar", DivScalarOp())

class DivSparseScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar}, indices=a.indices, data_shape=a.data_shape)
    
    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]

div_sparse_scalar = register_op("DivSparseScalar", DivSparseScalarOp())


class MatMulOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        # lmat, rmat = node.inputs
        # lsize = len(lmat.shape)
        # laxes = (lsize - 2, lsize - 1)
        # rsize = len(rmat.shape)
        # raxes = (rsize - 2, rsize - 1)
        # lgrad = out_grad.matmul(rmat.transpose(raxes))
        # rgrad = lmat.transpose(laxes).matmul(out_grad)
        # if lsize > rsize:
        #     axes = tuple(range(lsize - rsize))
        #     rgrad = rgrad.sum(axes)
        # if lsize < rsize:
        #     axes = tuple(range(rsize - lsize))
        #     lgrad = lgrad.sum(axes)
        # return (lgrad, rgrad)
        lhs, rhs = node.inputs
        lhs_T = lhs.transpose(None)
        rhs_T = rhs.transpose(None)
        lhs_g = out_grad.matmul(rhs_T)
        rhs_g = lhs_T.matmul(out_grad)
        if len(lhs_g.shape) != len(lhs.shape):
            size_diff = len(lhs_g.shape) - len(lhs.shape)
            diff_list = []
            for i in range(size_diff):
                diff_list.append(i)
            axes = tuple(diff_list)
            lhs_g = lhs_g.sum(axes)
        if len(rhs_g.shape) != len(rhs.shape):
            size_diff = len(rhs_g.shape) - len(rhs.shape)
            diff_list = []
            for i in range(size_diff):
                diff_list.append(i)
            axes = tuple(diff_list)
            rhs_g = rhs_g.sum(axes)
        return [lhs_g, rhs_g]


matmul = register_op("MatMul", MatMulOp())

class MatMulSparseDenseOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"indices": a.indices, "shape": a.shape})
    
    def gradient(self, out_grad, node):
        lmat, rmat = node.inputs
        lgrad = matmul(out_grad, transpose(rmat)).to_sparse(lmat.indices, lmat.data_shape) 
        rgrad = matmul_sparse_dense(transpose_sparse(lmat), out_grad)
        # out_grad.T @ lmat
        return (lgrad, rgrad)

matmul_sparse_dense = register_op("SparseDenseMatMul", MatMulSparseDenseOp())


class SummationOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        new_shape = list(mat.shape)
        axes = node.attrs["axes"]
        if axes is None:
            axes = tuple(range(len(mat.shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        for axis in axes:
            new_shape[axis] = 1
        return [out_grad.reshape(new_shape).broadcast_to(mat.shape)]


summation = register_op("Summation", SummationOp())


class BroadcastToOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        shape = mat.shape
        axes = []
        for i, dim in enumerate(node.attrs["shape"]):
            if i >= len(shape) or dim != shape[i]:
                axes.append(i)
        return [out_grad.sum(tuple(axes)).reshape(shape)]


broadcast_to = register_op("BroadcastTo", BroadcastToOp())


class ReshapeOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        return [out_grad.reshape(mat.shape)]


reshape = register_op("Reshape", ReshapeOp())


class NegateOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [-out_grad]


negate = register_op("Negate", NegateOp())

class NegateSparseOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a], indices=a.indices, data_shape=a.data_shape)

    def gradient(self, out_grad, node):
        return [-out_grad]


negate_sparse = register_op("NegateSparse", NegateSparseOp())


class TransposeOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        return [out_grad.transpose(node.attrs["axes"])]


transpose = register_op("Transpose", TransposeOp())

class TransposeSparseOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes}, indices=a.indices.swap(), data_shape=a.data_shape[::-1])

    def gradient(self, out_grad, node):
        return [out_grad.transpose(node.attrs["axes"])]


transpose_sparse = register_op("TransposeSparse", TransposeSparseOp())


class LogOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        return [out_grad / mat]


log = register_op("Log", LogOp())


class ExpOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [exp(node.inputs[0]) * out_grad]


exp = register_op("Exp", ExpOp())


class ReLUOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        relu_recompute = relu(mat)
        return [relu_recompute / mat * out_grad]


relu = register_op("ReLU", ReLUOp())


class SigmoidOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        sigmoid_recompute = sigmoid(mat)
        sigmoid_negate = negate(sigmoid_recompute)
        return [sigmoid_recompute * (1 + sigmoid_negate) * out_grad]


sigmoid = register_op("Sigmoid", SigmoidOp())



class LogSoftmaxOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        out_grad_sum_np = np.sum(out_grad.numpy(), axis=1, keepdims=True)
        out_grad_sum = out_grad.sum(1).reshape((mat.shape[0],) + (1,)).broadcast_to(mat.shape)
        return [out_grad - exp(logsoftmax(mat)) * out_grad_sum]


logsoftmax = register_op("LogSoftmax", LogSoftmaxOp())


class TanhOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        t = tanh(mat)
        sq = power_scalar(t, 2)
        neg = negate(sq)
        return [out_grad * add_scalar(neg, 1)]


tanh = register_op("Tanh", TanhOp())


class GetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        [mat] = node.inputs
        idxs = node.attrs["idxs"]
        grad = zeros_like(mat)
        grad[idxs] = out_grad
        return [grad]


get_item = register_op("GetItem", GetItemOp())


class SetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        raise NotImplementedError()

set_item = register_op("SetItem", SetItemOp())


class StackOp(Op):
    def __call__(self, args: List[Value], axis: int) -> Tensor:
        return Tensor.make_from_op(self, args, attrs={'axis': axis})

    def gradient(self, out_grad, node):
        axis = node.attrs["axis"]
        shape = out_grad.shape[:axis] + out_grad.shape[axis + 1:]
        slices = [slice(None, None, None) for i in range(len(out_grad.shape))]
        grads = []
        for s in range(out_grad.shape[axis]):
            slices[axis] = slice(s, s + 1, None)
            grad = out_grad[tuple(slices)]
            grads.append(grad.reshape(shape))
        return grads


stack = register_op("Stack", StackOp())


class ConvOp(Op):
    def __call__(self, a: Tensor, b: Tensor, stride: Optional[int] = 1, padding: Optional[int] = 0) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={'stride': stride, 'padding': padding})

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        weight = node.inputs[1]
        padding = node.attrs["padding"]
        stride = node.attrs["stride"]

        N, H, W, C_in = Z.shape
        K, _, _, C_out = weight.shape
        weight_flip = flip(weight, (0, 1))
        weight_permute = weight_flip.transpose((2, 3))
        if stride > 1:
            out_grad = dilate(out_grad, stride-1, (1, 2))
        _, H_out, W_out, _ = out_grad.shape
        Z_grad_padding = (((H - 1) + K) - H_out) // 2
        Z_grad = conv(out_grad, weight_permute, padding=Z_grad_padding, stride=1)

        Z_permute = Z.transpose((0, 3))
        out_grad_permute = out_grad.transpose((0, 1)).transpose((1, 2))
        weight_grad_padding = ((K - 1) + H_out - H) // 2
        weight_grad = conv(Z_permute, out_grad_permute, padding=weight_grad_padding, stride=1)
        weight_grad_permute = weight_grad.transpose((0, 1)).transpose((1, 2))       

        return [Z_grad, weight_grad_permute]

conv = register_op("Conv", ConvOp())


class FlipOp(Op):
    def __call__(self, a: Tensor, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'axes': axes})

    def gradient(self, out_grad, node):
        return [flip(out_grad, node.attrs["axes"])]

flip = register_op("Flip", FlipOp())


class DilateOp(Op):
    def __call__(self, a: Tensor, dilation: int, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'dilation': dilation, 'axes': axes})

    def gradient(self, out_grad, node):
        axes = node.attrs["axes"]
        dilation = node.attrs["dilation"]
        slices = [slice(None, None, None) for i in range(len(out_grad.shape))]
        if isinstance(axes, int):
            axes = (axes,)
        for a in axes:
            if a >= len(slices):
                continue
            slices[a] = slice(None, None, 1 + dilation)
        out = out_grad.__getitem__(tuple(slices))
        return [out]

dilate = register_op("Dilate", DilateOp())


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    device = device if device else default_device()

    if not rand or "dist" not in rand:
        arr = device.empty(shape, dtype)
        device.fill(arr, fill_value)
    else:
        if rand["dist"] == "normal":
            arr = device.randn(shape, dtype, mean=rand["mean"], std=rand["std"])
        if rand["dist"] == "binomial":
            arr = device.randb(shape, dtype, ntrials=rand["trials"], p=rand["prob"])
        if rand["dist"] == "uniform":
            arr = device.randu(shape, dtype, low=rand["low"], high=rand["high"])

    return Tensor.make_const(arr, device, requires_grad=requires_grad)


def one_hot(labels: Tensor, *, num_classes=10, dtype="float32", device=None):
    device = device if device else default_device()
    arr = device.one_hot(labels.numpy(), num_classes=num_classes)
    return Tensor.make_const(arr, device, requires_grad=False)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )

"""Core data structures."""
import needle
import numpy as np
from needle import backend_ndarray as nd

from typing import List, Optional, NamedTuple
from collections import namedtuple
from .device import default_device, Device, CachedData

LAZY_MODE = False
TENSOR_COUNTER = 0


class Op:
    """Operator definition."""

    def gradient(self, out_grad: "Value", node: "Value") -> List["Value"]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: List[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    attrs: object
    # The following fields are cached fields for
    # dynamic computation
    cached_data: CachedData
    cached_device: Device
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.cached_device.compute(
            self.op, [x.realize_cached_data() for x in self.inputs], self.attrs
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def is_sparse(self):
        return self.is_sparse

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        attrs: object = None,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        cached_device: Device = None,
        requires_grad: Optional[bool] = None,
        is_sparse = False,
        indices = None,
        data_shape = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        # deduce the device of the computation
        if cached_device is None:
            if not inputs:
                raise ValueError(
                    "Requires cached device to be available for tensor with no inputs"
                )
            cached_device = inputs[0].cached_device
            for x in inputs:
                if cached_device != x.cached_device:
                    raise ValueError(
                        "Requires all input devices to be the same to automatically"
                        "deduce device"
                    )
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.attrs = attrs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.cached_device = cached_device
        self.requires_grad = requires_grad
        self.is_sparse = is_sparse
        self.indices = indices
        if indices is not None:
            self.is_sparse = True
        self.data_shape = data_shape

    @property
    def device(self):
        return self.cached_device

    @classmethod
    def make_const(cls, data, device, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            cached_device=device,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(
        cls, op: Op, inputs: List["Value"], *, attrs=None, indices=None, data_shape=None, cached_device=None
    ):
        value = cls.__new__(cls)
        value._init(op, inputs, attrs=attrs, indices=indices, data_shape=data_shape, cached_device=cached_device)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class Tuple(Value):
    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.Tuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, Tuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data(), self.device)


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self, array, *, device: Optional[Device] = None, dtype=None, requires_grad=True, is_sparse = False, indices = None, data_shape = None
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = device.array(array.numpy(), dtype=dtype)
        else:
            device = device if device else default_device()
            cached_data = device.array(array, dtype=dtype)
            if indices is not None:
                indices = device.array(indices, dtype="float32")

        self._init(
            None,
            [],
            cached_device=device,
            cached_data=cached_data,
            requires_grad=requires_grad,
            is_sparse=is_sparse,
            indices=indices,
            data_shape = data_shape
        )

    def to_sparse(self, indices=None, data_shape=None):
        if not self.is_sparse:
            if indices is not None:
                return Tensor(self.cached_data.to_sparse_by_indices(indices), device=self.device, dtype=self.dtype, is_sparse=True, indices=indices, data_shape=data_shape)
            else:
                values = []
                indices = []
                shape = self.cached_data.shape
                ndims = len(shape)
                start = [0 for i in range(ndims)]
                prev_index = start
                if self.cached_data.numpy().item(tuple(start)) > np.finfo(np.float32).eps:
                    indices.append(start)
                    values.append(self.cached_data.numpy().item(tuple(start)))
                num_loops = 1
                data_device = self.cached_data.device
                for i in range(ndims):
                    num_loops *= shape[i]
                for n in range(1, num_loops):
                    cur_index = []
                    carry = 1
                    for i in range(ndims-1, -1, -1):
                        cur_index.append((carry+prev_index[i])%shape[i])
                        carry = (carry+prev_index[i])//shape[i]
                    cur_index.reverse()
                    if self.cached_data.numpy().item(tuple(cur_index)) > np.finfo(np.float32).eps:
                        indices.append(cur_index)
                        values.append(self.cached_data.numpy().item(tuple(cur_index)))
                    prev_index = cur_index
                indices = self.device.array(np.array(indices).transpose(), dtype=self.dtype)
                value = self.device.array(np.array(values), dtype=self.dtype)
                return Tensor(values, device=self.device, dtype=self.dtype, is_sparse=True, indices=indices, data_shape=shape)
        else:
            return self


    def to_dense(self):
        if self.is_sparse:
            data = nd.array(np.zeros(self.data_shape), device=self.cached_data.device)
            indices_list = []
            index_shape = self.indices.shape
            for i in range(index_shape[1]):
                index = []
                for j in range(index_shape[0]):
                    index.append(int(self.indices[j, i].numpy()))
                index = tuple(index)
                data[index] = self.cached_data[(i,)]
            return Tensor(data, device=self.device, dtype=self.dtype)
        else:
            return self



    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"], *, attrs=None, indices=None, data_shape=None, cached_device=None):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs, attrs=attrs, indices=indices, data_shape=data_shape, cached_device=cached_device)
        if not LAZY_MODE:
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, device, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            cached_device=device,
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.device == self.device and value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data(), self.device)

    @property
    def shape(self):
        if self.is_sparse:
            return self.data_shape
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else needle.ops.ones_like(self)
        compute_gradient_of_variables(self, out_grad)

    def __getitem__(self, idxs):
        return needle.ops.get_item(self, idxs)

    def __setitem__(self, idxs, other):
        return needle.ops.set_item(self, idxs, other)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        if self.is_sparse:
            return self.to_dense().realize_cached_data().__str__()
        return self.realize_cached_data().__str__()

    def numpy(self):
        if self.is_sparse:
            return self.device.to_numpy(self.to_dense().realize_cached_data())
        return self.device.to_numpy(self.realize_cached_data())

    def __add__(self, other):
        if isinstance(other, Tensor):
            if self.is_sparse:
                if other.is_sparse:
                    return needle.ops.add_sparse_sparse(self, other)
                else:
                    assert self.shape == other.shape
                    return needle.ops.add_sparse_dense(self, other)
            else:
                return needle.ops.add(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            if self.is_sparse:
                return needle.ops.add_sparse_scalar(self, other)
            else:
                return needle.ops.add_scalar(self, other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if self.is_sparse:
                if other.is_sparse:
                    raise NotImplementedError("sparse-sparse multiply not implemented")
                else:
                    assert self.shape == other.shape
                    return needle.ops.ewise_mul_sparse_dense(self, other)
            else:
                return needle.ops.multiply(self, other)
        else:
            if self.is_sparse:
                return needle.ops.mul_sparse_scalar(self, other)
            else:
                return needle.ops.multiply_scalar(self, other)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return needle.ops.power_scalar(self, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.add(self, needle.ops.negate(other))
        else:
            return needle.ops.add_scalar(self, -other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            if self.is_sparse:
                if other.is_sparse:
                    raise NotImplementedError("sparse-sparse division not implemented")
                else:
                    assert self.shape == other.shape
                    return needle.ops.ewise_div_sparse_dense(self, other)
            else:
                return needle.ops.divide(self, other)
        else:
            if self.is_sparse:
                return needle.ops.div_sparse_scalar(self, other)
            else:
                return needle.ops.divide_scalar(self, other)

    def __matmul__(self, other):
        if self.is_sparse:
            assert other.is_sparse != True
            assert self.shape[1] == other.shape[0]
            return needle.ops.matmul_sparse_dense(self, other)
        else:
            return needle.ops.matmul(self, other)
    
    def matmul(self, other):
        return needle.ops.matmul(self, other)

    def sum(self, axes=None):
        return needle.ops.summation(self, axes)

    def broadcast_to(self, shape):
        return needle.ops.broadcast_to(self, shape)

    def reshape(self, shape):
        return needle.ops.reshape(self, shape)

    def __neg__(self):
        if self.is_sparse:
            return needle.ops.negate_sparse(self)
        else:
            return needle.ops.negate(self)

    def transpose(self, axes=None):
        if self.is_sparse:
            return needle.ops.transpose_sparse(self, axes)
        else:
            return needle.ops.transpose(self, axes)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

def sparse_coo_tensor(indices, values, data_shape, device=None, dtype=None):
      values = nd.array(values)
      indices = nd.array(indices)
      return Tensor(array=values, device=device, dtype=dtype, indices=indices, data_shape=data_shape, is_sparse=True)

def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        node.grad = grad
        if node.op is not None:
            input_grad = node.op.gradient(grad, node)
            for input_node, grad in zip(node.inputs, input_grad):
                if input_node not in node_to_output_grads_list:
                    node_to_output_grads_list[input_node] = [grad]
                else:
                    node_to_output_grads_list[input_node].append(grad)


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    topo_order_list = []
    for node in node_list:
        visited = []
        topo_order = []
        topo_sort_dfs(node, visited, topo_order)
        topo_order_list.extend(topo_order)
    return topo_order_list


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.append(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)

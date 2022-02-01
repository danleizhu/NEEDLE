import sys
sys.path.append('./python')
import torch
import numpy as np
import needle as ndl
import pytest


i = [[0, 1, 1, 2],
      [2, 0, 2, 3]]
v =  [3.0, 4.0, 5.0, 6.0]
shape = (3, 4)

def test_sparse_basics_and_sparse_dense_conversion():
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape)
    assert a.shape == shape
    a_ = torch.sparse_coo_tensor(i, v, shape)
    np.testing.assert_allclose(a.indices.numpy(), a_.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), a_.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)
    assert a.is_sparse == True

    b = a.to_dense()
    b_ = a_.to_dense()
    np.testing.assert_allclose(b.numpy(), b_.numpy(), atol=1e-5, rtol=1e-5)
    assert b.indices == None
    assert b.is_sparse == False

    a = b.to_sparse()
    np.testing.assert_allclose(a.indices.numpy(), a_._indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), a_._values().numpy(), atol=1e-5, rtol=1e-5)
    assert a.is_sparse == True

def test_higher_dimension_sparse_basics_and_sparse_dense_conversion():
    i = [[0, 1, 0, 1],
        [2, 1, 0, 3],
        [3, 2, 1, 4],
        [0, 0, 1, 0]]
    v = [3.0, 4.0, 5.0, 6.0]
    shape = (2, 4, 5, 3)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape)
    assert a.shape == shape
    a_ = torch.sparse_coo_tensor(i, v, shape)
    np.testing.assert_allclose(a.indices.numpy(), a_._indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), a_._values().numpy(), atol=1e-5, rtol=1e-5)
    assert a.is_sparse == True

    b = a.to_dense()
    b_ = a_.to_dense()
    np.testing.assert_allclose(b.numpy(), b_.numpy(), atol=1e-5, rtol=1e-5)
    assert b.indices == None
    assert b.is_sparse == False

    a = b.to_sparse()
    np.testing.assert_allclose(a.indices.numpy(), a_.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), a_.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)
    assert a.is_sparse == True

# sparse_scalar_add_cpu
def test_sparse_scalar_add_cpu():
    scalar = np.random.rand(1)[0]
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    a = a + scalar
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a.to_dense() + scalar
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

# sparse_dense_add_cpu
def test_sparse_dense_add_cpu():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    a = a + d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _d + _a
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

# sparse_scalar_mul_cpu
def test_sparse_scalar_mul_cpu():
    scalar = np.random.rand(1)[0]
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    a = a * scalar
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a * scalar
    np.testing.assert_allclose(a.indices.numpy(), _a.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), _a.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_mul_cpu
def test_sparse_dense_mul_cpu():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    a = a * d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _a.to_dense() * _d
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_mul_backward_cpu
def test_sparse_dense_mul_backward_cpu():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    y = a * d
    y.sum().backward()
    _a = torch.sparse_coo_tensor(i, v, shape).requires_grad_(True)
    _d = torch.from_numpy(dense).to(torch.float32).requires_grad_(True)
    _y = _a.to_dense() * _d
    _y.sum().backward()
    np.testing.assert_allclose(a.grad.to_dense().numpy(), _a.grad.to_dense().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(d.grad.numpy(), _d.grad.numpy(), atol=1e-5, rtol=1e-5)

# sparse_scalar_div_cpu
def test_sparse_scalar_div_cpu():
    scalar = np.random.rand(1)[0]
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    a = a / scalar
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a / scalar
    np.testing.assert_allclose(a.indices.numpy(), _a.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), _a.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)


#sparse_dense_div_cpu
def test_sparse_dense_div_cpu():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    a = a / d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _a.to_dense() / _d
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_div_backward_cpu
def test_sparse_dense_div_backward_cpu():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    y = a / d
    y.sum().backward()
    _a = torch.sparse_coo_tensor(i, v, shape).requires_grad_(True)
    _d = torch.from_numpy(dense).to(torch.float32).requires_grad_(True)
    _y = _a.to_dense() / _d
    _y.sum().backward()
    np.testing.assert_allclose(a.grad.to_dense().numpy(), _a.grad.to_dense().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(d.grad.numpy(), _d.grad.numpy(), atol=1e-5, rtol=1e-5)

#sparse_negate_cpu
def test_sparse_negate_cpu():
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    a = - a
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = - _a
    np.testing.assert_allclose(a.indices.numpy(), _a.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), _a.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)

#sparse_transpose_cpu
def test_sparse_transpose_cpu():
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    a = a.transpose()
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a.transpose(0, 1)
    # pytorch randomly permute the sequence of indices after transpose, so we do not compare the indices and values separately
    np.testing.assert_allclose(a.to_dense().numpy(), _a.to_dense().numpy(), atol=1e-5, rtol=1e-5)


#sparse_dense_matmul_cpu
def test_sparse_dense_matmul_cpu():
    dense = np.random.rand(*shape[::-1])
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    a = a @ d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _a @ _d
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)


#sparse_dense_matmul_backward_cpu
def test_sparse_dense_matmul_backward_cpu():
    dense = np.random.rand(*shape[::-1])
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cpu())
    d = ndl.Tensor(dense, device=ndl.cpu())
    y = a @ d
    y.sum().backward()
    _a = torch.sparse_coo_tensor(i, v, shape).requires_grad_(True)
    _d = torch.from_numpy(dense).to(torch.float32).requires_grad_(True)
    _y = torch.sparse.mm(_a, _d)
    _y.sum().backward()
    np.testing.assert_allclose(d.grad.numpy(), _d.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.grad.to_dense().numpy(), _a.grad.to_dense().numpy(), atol=1e-5, rtol=1e-5)

# sparse_scalar_add_cuda
def test_sparse_scalar_add_cuda():
    scalar = np.random.rand(1)[0]
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    a = a + scalar
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a.to_dense() + scalar
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

# sparse_dense_add_cuda
def test_sparse_dense_add_cuda():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    a = a + d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _d + _a
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

# sparse_scalar_mul_cuda
def test_sparse_scalar_mul_cuda():
    scalar = np.random.rand(1)[0]
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    a = a * scalar
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a * scalar
    np.testing.assert_allclose(a.indices.numpy(), _a.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), _a.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_mul_cuda
def test_sparse_dense_mul_cuda():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    a = a * d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _a.to_dense() * _d
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_mul_backward_cuda
def test_sparse_dense_mul_backward_cuda():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    y = a * d
    y.sum().backward()
    _a = torch.sparse_coo_tensor(i, v, shape).requires_grad_(True)
    _d = torch.from_numpy(dense).to(torch.float32).requires_grad_(True)
    _y = _a.to_dense() * _d
    _y.sum().backward()
    np.testing.assert_allclose(a.grad.to_dense().numpy(), _a.grad.to_dense().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(d.grad.numpy(), _d.grad.numpy(), atol=1e-5, rtol=1e-5)

# sparse_scalar_div_cuda
def test_sparse_scalar_div_cuda():
    scalar = np.random.rand(1)[0]
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    a = a / scalar
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a / scalar
    np.testing.assert_allclose(a.indices.numpy(), _a.coalesce().indices().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.cached_data.numpy(), _a.coalesce().values().numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_div_cuda
def test_sparse_dense_div_cuda():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    a = a / d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _a.to_dense() / _d
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_div_backward_cuda
def test_sparse_dense_div_backward_cuda():
    dense = np.random.rand(*shape)
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    y = a / d
    y.sum().backward()
    _a = torch.sparse_coo_tensor(i, v, shape).requires_grad_(True)
    _d = torch.from_numpy(dense).to(torch.float32).requires_grad_(True)
    _y = _a.to_dense() / _d
    _y.sum().backward()
    np.testing.assert_allclose(a.grad.to_dense().numpy(), _a.grad.to_dense().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(d.grad.numpy(), _d.grad.numpy(), atol=1e-5, rtol=1e-5)

#sparse_transpose
def test_sparse_transpose_cuda():
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    a = a.transpose()
    _a = torch.sparse_coo_tensor(i, v, shape)
    _a = _a.transpose(0, 1)
    # pytorch randomly permute the sequence of indices after transpose, so we do not compare the indices and values separately
    np.testing.assert_allclose(a.to_dense().numpy(), _a.to_dense().numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_matmul_cuda
def test_sparse_dense_matmul_cuda():
    dense = np.random.rand(*shape[::-1])
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    a = a @ d
    _a = torch.sparse_coo_tensor(i, v, shape)
    _d = torch.from_numpy(dense).to(torch.float32)
    _a = _a @ _d
    np.testing.assert_allclose(a.numpy(), _a.numpy(), atol=1e-5, rtol=1e-5)

#sparse_dense_matmul_backward_cuda
def test_sparse_dense_matmul_backward_cuda():
    # dense = np.random.rand(*shape[::-1])
    dense = np.ones(shape[::-1])
    a = ndl.sparse_coo_tensor(indices=i, values=v, data_shape=shape, device=ndl.cuda())
    d = ndl.Tensor(dense, device=ndl.cuda())
    y = a @ d
    y.sum().backward()
    _a = torch.sparse_coo_tensor(i, v, shape).requires_grad_(True)
    _d = torch.from_numpy(dense).to(torch.float32).requires_grad_(True)
    _y = torch.sparse.mm(_a, _d)
    _y.sum().backward()
    np.testing.assert_allclose(y.numpy(), _y.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(a.grad.to_dense().numpy(), _a.grad.to_dense().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(d.grad.numpy(), _d.grad.numpy(), atol=1e-5, rtol=1e-5)

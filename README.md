# Necessary Element for Deep Learning Library


## Support for Sparse Matrix Computation
In addition we added the support for sparse matrices to our `needle` library. Specifically, the following features were implemented:

1) sparse matrix construction and dense conversion

2) sparse matrix operations

3) auto-differentiation 

4) support for CPU, CUDA backends

Finally, we used our `needle` library to construct a Graph Convolutional Network and trained it on a document classification task using Cora dataset.


## Introduction to `Sparse Matrix`

The basic `needle ` library stores all elements of a multidimensional array contiguously in memory. This implementation is efficient for general matrix operations by allowing the fast access to array elements. However, it is not optimal for sparse matrices, which have a majority of zero elements. The space complexity can be improved a lot if only the non-zero elements are stored in the memory. The time complexity would be also improved because we could just handle the non-zero elements in some matrix operations such as matrix multiplication. Multiple formats to store sparse matrices have been proposed, such as COO, CSR, LIL and etc. We implemented the COO format in our extended `needle` library.

## Part 1: Sparse Matrix Construction and Dense Conversion

In the COO format, a tensor is stored as an array of non-zero values, an array of their corresponding indices, and the shape of its dense counterpart. In our implementation, the original `cached_data` attribute is now used to store the 1D array of all the non-zero elements in a sparse matrix while another NDArray called `indices` is added as an attribute of the `Tensor` class to store the indices of the elements. `data_shape` and `is_sparse` attributes were also added to the `Tensor` class in `autograd.py` for sparse matrices specifically.

Function `sparse_coo_tensor(indices, values, data_shape, device=None, dtype=None)` defined in `autograd.py` is used to construct a sparse tensor with the following fields specified:

- `indices`: an array (list or numpy array) of indices of non-zero elements with shape `(ndim, nnze)`,

- `values`: an array (list or numpy array) of non-zero elements with shape `(nnze, )`,

- `data_shape`: a tuple denoting the shape of the dense version of the sparse tensor,

where `ndim` is the dimensionality of the tensor and `nnze` is the number of non-zero elements.

Sparse-to-dense conversion and dense-to-sparse conversion were implemented in `to_dense()` and `to_sparse()`. 

`to_dense()` was straightforward to implement. We created a zero array with size of `data_shape`. Then, we looped through `indices` to fill the non-zero elements in the array and returned the tensor containing the dense array as data.

`to_sparse()` was general for matrices with all kinds of shapes. Simply looping through all dimensions would not work since the number of loops would be different for tensors with different shapes. Thus, we used the trick we had used in HW3 to generate a list of all indices. Then, we iterated through all indices and store the non-zero elements plus their corresponding indices in `values` and `indices` ndarrays. A sparse tensor with `array = values, indices = indices, shape = self.cached_data.shape` would be returned.

Note that since the `cached_data` attribute was used to store only non-zero elements for sparse matrices, we also modified the functions `numpy()`, and `__str__()` so that the dense counterpart is returned.  

## Part 2: Sparse Matrix Operations and Auto-differentiation

Based on the proposal feedback, we only focused on the sparse-dense matrix operations. We modified the `ops.py`, `nd_backend.py`, and `ndarray.py` to add the following sparse matrix operations with their corresponding gradients:

- `AddSparseScalarOp`

- `AddSparseDenseOp`

- `AddSparseSparseOp`

- `EwiseMulSparseDenseOp`

- `MulSparseScalarOp`

- `EwiseDivSparseDenseOp`

- `DivSparseScalarOp`

- `NegateSparseOp`

- `TransposeSparseOp`

- `MatMulSparseDenseOp`

The implementation of the first eight operations is similar to that of the corresponding dense matrix operations. Note that the output of the operation between a sparse matrix and a dense matrix should be in dense form. The gradient of a sparse matrix is always sparse. 

We only support the sparse transpose operation for 2 dimensional tensors. Therefore, we just need to modify the `indices` and `data_shape` attribute of the tensor by swapping the its first and second axis. 

For example, the tranpose of a sparse matrix with `values = [3, 4, 5], indices = [[0, 1, 1], [2, 0, 2]], data_shape = (2, 3)` is a sparse matrix with `values = [3, 4, 5], indices = [[2, 0, 2], [0, 1, 1]], data_shape = (3, 2)`.

The implementation of sparse matrix multiplication is also similar to that of the dense version, except that we only iterate through the position with non-zero elements by looping through the `indices` instead of iterating through the whole rows and columns. This would improve the time complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n^2)$.

## Part 3: CPU and CUDA Backends

Our general idea and design choices follow the PyTorch sparse APIs for both testing convenience and practical reasons. However, we might have implemented some operations that are not supported (to our best knowledge) by PyTorch, like element-wise multiplication between a sparse matrix and a dense matrix. Notice that we only support 2D sparse matrices at this point.

### CPU Backends:
#### Forward:
- `AddSparseScalarOp`: The addition between a sparse matrix and a scalar value results in a dense matrix. In the CPU backend, we first fill the memory with the scalar value and then add back every element in the sparse matrix according to the `indices`.

- `AddSparseDenseOp`: The addition between a sparse matrix and a dense matrix also results in a dense matrix. Similarly, we first fill the memory with the values from the dense matrix before adding back every element in the sparse matrix by traversing the `indices`.

- `AddSparseSparseOp`: The addition between two sparse matrices gives a new sparse matrix. Unlike previous addition operations, here we do not want to re-construct the dense format of the matrices and iterate through every value. Instead, we find the union of their `indices` to get the indices of all non-zero elements. We then add up the values stored in the `cached_data` attribute according to the union of `indices` to get the non-zero elements in the resulting sparse matrix. This operation is important since it is also used in auto-differentiation to sum up sparse gradients in the computational graph.

- `EwiseMulSparseDenseOp`: The element-wise multiplication between a sparse matrix and a dense matrix results in a dense matrix. Similar to the element-wise addition operation, we first fill the memory with all values from the dense matrix and then multiply back every element in the sparse matrix according to the `indices`.

- `MulSparseScalarOp`: The element-wise multiplication between a sparse matrix and a scalar value gives a sparse matrix. For this operation, we multiply the non-zero elements stored in `cached_data` of the sparse matrix with the scalar by re-using the scalar multiplication operation for general matrices. We then pass the same `indices` and `data_shape` attributes of the sparse matrix to form the resulting sparse matrix.

- `EwiseDivSparseDenseOp`: The implementation of this operation is almost identical to `EwiseMulSparseDenseOp`.

- `DivSparseScalarOp`: The implementation of this operation is almost identical to `MulSparseScalarOp`.

- `NegateSparseOp`: Similarly, we re-use the negate operation of general matrices to negate the non-zero elements of the sparse matrix and keep the same `indices` and `data_shape`.

- `TransposeSparseOp`: For sparse matrices, transposing the matrix just requires swapping the row and column indices of the non-zero elements. In our implementation, we implement a `swap` a function to explicitly swap the values of the `indices` attribute. We leave the values stored in `cached_data` unchanged.

- `MatMulSparseDenseOp`: This operation gives dense matrix as result. While using CPU backend to compute the matrix multiplication, the sparse matrix remains in COO format. We iterate over every non-zero element in the sparse matrix. Let's say we have a non-zero element at `(i, j)` of the sparse matrix, with `i` and `j` fixed, we iterate over the columns of the dense matrix, denoted as `k`, to find the value at `(i, k)` of the resulting dense matrix

#### Backward:
- We only implemented and tested backward operations for sparse-dense element-wise multiplication, sparse-dense element-wise division and sparse-dense matrix multiplication. The first two are straightforward. But the last one requires some notice. The gradient of the sparse matrix is also sparse and only non-zero elements have gradients, however, we get a dense matrix by calculating the multiplication of the output gradient and the dense multiplier. Therefore, we need to explicitly convert the dense matrix to sparse COO format before return. We make some changes to the `Tensor.make_from_op()` call such that it accepts `indices` and `data_shape` as optional parameters to construct a sparse matrix from a dense matrix with non-zero elements' indices specified.

### CUDA Backends:
The only difference between CPU backend and CUDA backend is that, before any operation, we implicitly convert the sparse matrix from the COO format to CSR format using the `get_csr` function defined in `needle/backend_ndarray/ndarray.py`. CSR format allows us to quickly find the offset of a specific value in the underlying `cached_data` handle when iterating the matrix row-wise.


## Part 4: Application to Graph Convolution Network

One of the applications of the sparse matrices is on the graph neural networks. In general, only small portions of nodes in a graph are connected to each other, resulting in a sparse adjacency matrix. 

A graph convolution network (GCN) tries to learn a function of features on graph G = (V, E), which takes as input:

- a feature matrix `X` of size `(n, nfeat)`, of which each row represents the input feature of each node,

- an adjacency matrix `A` of size `(n, n)`, which represents the graph structure,

where `n` is the number of nodes in the graph and `nfeat` is the dimensionality of the feature of each node. The GCN produce a output matrix `Z` of size `(n, nout)`, where `nout` is the dimensionality of the output feature. 

In `nn.py`, we implemented the graph convolution layer (`GCLayer`), which propagates the neighbors' features based on the following rule:

\begin{align}
H^{l} = \sigma(AH^{l-1}W^{l}),
\end{align}

where $H^{l}$ is the feature matrix at the $l$th layer, $W^l$ is the weight matrix at the $l$th layer, and $\sigma$ is a non-linear activation function. Note that $H^{0} = X$ and $H^{m} = Z$, where $m$ is the number of graph convoluation layers in the model. 

In `data.py`, we used `load_cora` to load the cora dataset and generated four outputs, including `features`, `labels`, `labels_onehot` and `adjacency_matrix`.  Each node is a document with binary features denoting whether it contains a specific word or not. The adjacency matrix represents the citation relationship between documents. Both the feature matrix and the adjacency matrix are sparse. The label is the document category. Specifically, the number of nodes in the graph is 2708 and the input feature size is 1433, and the number of classes is 7. Therefore the dimensionalities of `features`, `labels`, and `adjacency_matrix` are `(2708, 1433)`, `(2708, 2708)`, and `(2708, 7)`.

In `apps/gcn_cora.py`, we constructed an two-layer GCN with `hidden_feature = 10` and `output_feature = 7`. We applied the `ReLU` activation function for the first layer. 

We used the softmax as the loss function and trained our model for 100 epochs using the SGD optimizer with the learning rate = 0.3. From the training logs, we could see that the model was learning and the accuracy converged to 0.71.



We supported both CPU and CUDA for training. Note that CUDA toke much longer time than CPU. The cause for this issue was the conversion from the COO format to the CSR format in the `MatMulSparseDenseOp`. The conversion required that the `indices` be sorted first. As we had not figured out how to sort an array using CUDA, we used python to sort the array, which was the bottleneck for a giant array. This could be our future work to speed up the sparse matrix operation. 


```python
!python3 apps/gcn_cora.py cpu
```

    Using cpu backends
    Loading cora dataset...
    Converting features and adjacency matrix to Needle sparse coo format...
    Training:
    epoch: 1, accuracy: 0.058714918759231904, loss: 2.643169641494751
    epoch: 2, accuracy: 0.16469719350073855, loss: 2.0339417457580566
    epoch: 3, accuracy: 0.39254062038404725, loss: 1.866609811782837
    epoch: 4, accuracy: 0.4154357459379616, loss: 1.6810632944107056
..................
    epoch: 98, accuracy: 0.7060561299852289, loss: 0.8860905766487122
    epoch: 99, accuracy: 0.7060561299852289, loss: 0.8847568035125732
    epoch: 100, accuracy: 0.7060561299852289, loss: 0.8834126591682434
    


```python
!python3 apps/gcn_cora.py cuda
```

    Using cuda backends
    Loading cora dataset...
    Converting features and adjacency matrix to Needle sparse coo format...
    Training:
    epoch: 1, accuracy: 0.08677991137370754, loss: 2.8672029972076416
    epoch: 2, accuracy: 0.15472673559822747, loss: 2.1328022480010986
    epoch: 3, accuracy: 0.3212703101920236, loss: 1.865748643875122
    epoch: 4, accuracy: 0.4401772525849335, loss: 1.7014124393463135
  ...................................................
    epoch: 99, accuracy: 0.7064254062038404, loss: 0.8809061050415039
    epoch: 100, accuracy: 0.7071639586410635, loss: 0.8796224594116211
    

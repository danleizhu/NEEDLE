import io
import os
import pickle

import numpy as np
from .autograd import Tensor, sparse_coo_tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import needle as ndl


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class FlipHorizontal(Transform):
    def __init__(self):
        pass

    def __call__(self, img):
        side = int(np.sqrt(len(img)))
        img = np.reshape(img, (side, side))
        flip = np.flip(img, axis=1)
        return flip.flatten()


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, _x):
        side = int(np.sqrt(len(_x)))
        img = np.reshape(_x, (side, side))        
        pad = np.pad(img, self.padding)
        rand_range = 2 * self.padding
        x = np.random.randint(0, rand_range)
        y = np.random.randint(0, rand_range)
        crop = pad[x: x + side, y: y + side]
        return crop.flatten()


class Sampler:
    """Base class for all Samplers.
    Every Sampler subclass has to provide an `__iter__` method, providing a
    way to iterate over indices of dataset elements, and a `__len__` method
    that returns the length of the returned iterators.
    """

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source
        self.size = len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.size))

    def __len__(self) -> int:
        return self.size


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify `num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.size = len(data_source)

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if self.num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if self.num_samples is None:
            self.num_samples = len(data_source)
        # if not isinstance(self.num_samples, int) or self.num_samples <= 0:
        #     raise ValueError("num_samples should be a positive integer "
        #                      "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            return iter(np.random.choice(self.size, self.num_samples))
        else:
            return iter(np.random.permutation(self.size))

    def __len__(self) -> int:
        if self.replacement:
            return self.num_samples
        else:
            return self.size


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        idx = []
        sub_idx = []
        for i in self.sampler:
            sub_idx.append(i)
            if len(sub_idx) == self.batch_size:
                idx.append(sub_idx)
                sub_idx = []
        if not self.drop_last and sub_idx:
            idx.append(sub_idx)
        return iter(idx)

    def __len__(self) -> int:
        total = len(self.sampler)
        batches = total // self.batch_size
        extra = total - batches * self.batch_size
        if not self.drop_last and extra:
            batches += 1
        return batches


def default_collate(batch, device, dtype):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    return collate_ndarray(batch, device, dtype)


def collate_ndarray(batch, device, dtype):
    """
    Returns Tensor batch with nd array backend
    """
    nd_device = None
    if isinstance(device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()
    else:
        nd_device = nd.numpy_device()
    if isinstance(batch, list):
        x = Tensor(nd.array([data[0] for data in batch], dtype=dtype, device=nd_device), device=device)
        y = Tensor(nd.array([data[1] for data in batch], dtype=dtype, device=nd_device), device=device)
        return x, y
    elif isinstance(batch, tuple):
        x, y = batch
        x = Tensor(nd.array(x, dtype=dtype, device=nd_device), device=device)
        y = Tensor(nd.array(y, dtype=dtype, device=nd_device), device=device)
        return x, y


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, p: Optional[int] = 0.5, transforms: Optional[List] = None):
        self.p = p
        self.transforms = transforms
        self.perform_transforms = transforms is not None

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError

    def apply_transforms(self, x):
        if self.perform_transforms:
            if np.random.rand() < self.p:
                # apply the transforms
                for tform in self.transforms:
                    x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    """
    dataset: Dataset
    batch_size: Optional[int]
    drop_last: bool
    sampler: Sampler
    _iterator: Optional["_BaseDataLoaderIter"]
    __initialized = False

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Union[Sampler, Iterable, None] = None,
        collate_fn: Optional = default_collate,
        drop_last: bool = False,
        device = None,
        dtype = None
    ):

        self.dataset = dataset
        self.collate_fn = collate_fn

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if sampler is None:  # give default samplers
            # We are only doing map style datasets
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.dtype = dtype
        self.sampler = sampler

        if batch_size > 1:
            self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

        self.collate_fn = collate_fn

        self.__initialized = True
        self._IterableDataset_len_called = None

        self._iterator = None

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        return _SingleProcessDataLoaderIter(self)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> "_BaseDataLoaderIter":
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        return self._get_iterator()

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self.batch_size > 1:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        length = self._IterableDataset_len_called = len(self.dataset)
        if self.batch_size is not None:
            from math import ceil

            if self.drop_last:
                length = length // self.batch_size
            else:
                length = ceil(length / self.batch_size)
        return length


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._dataset = loader.dataset
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
        self._collate_fn = loader.collate_fn
        self._device = loader.device
        self._dtype = loader.dtype
        self._base_seed = np.empty((), dtype=np.int64)
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(
            self.__class__.__name__
        )

    def __iter__(self) -> "_BaseDataLoaderIter":
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            self._reset(self.loader)
        data = self._next_data()
        self._num_yielded += 1
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)

        self._dataset_fetcher = _IterableDatasetFetcher(
            self._dataset, self._collate_fn, self._drop_last, self._device, self._dtype
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


class _BaseDatasetFetcher(object):
    def __init__(self, dataset, collate_fn, drop_last, device, dtype):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.device = device
        self.dtype = dtype

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, collate_fn, drop_last, device, dtype):
        super(_IterableDatasetFetcher, self).__init__(dataset, collate_fn, drop_last, device, dtype)
        self.dataset = dataset
        self.ended = False
        self.device = device
        self.dtype = dtype

    def fetch(self, possibly_batched_index):
        if isinstance(possibly_batched_index, list):
            data = [self.dataset[idx] for idx in possibly_batched_index]
            return self.collate_fn(data, self.device, self.dtype)
        elif isinstance(possibly_batched_index, int):
            data = self.dataset[possibly_batched_index]
            return self.collate_fn(data, self.device, self.dtype)


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.
    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format
    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.
            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filesname: str,
        label_filename: str,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset

        Divide pixel values by 255. so that images are in 0-1 range.

        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        if train:
            files = [os.path.join(base_folder, "data_batch_" + str(i)) for i in range(1, 6)]
        else:
            files = [os.path.join(base_folder, "test_batch")]
        self.X = None
        self.y = None
        for f in files:
            with open(f, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            if self.X is None:
                self.X = data_dict["data".encode()].reshape((-1, 3, 32, 32))
            else:
                self.X = np.vstack((self.X, data_dict["data".encode()].reshape((-1, 3, 32, 32))))
            if self.y is None:
                self.y = data_dict["labels".encode()]
            else:
                self.y.extend(data_dict["labels".encode()])
        self.size = len(self.y)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index

        Image should be of shape (3, 32, 32)
        """
        return self.X[index] / 255, self.y[index]
        

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.size


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.

    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str

        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.

        Returns the word's unique ID.
        """
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.word2idx)


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in

        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.

        Output:
        ids: List of ids
        """
        ids = []
        with open(path, "r") as f:
            for line in f.readlines()[:max_lines]:
                for word in line.split():
                    ids.append(self.dictionary.add_word(word))
                ids.append(self.dictionary.add_word("<eos>"))
        return ids



def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.

    If the data cannot be evenly divided by the batch size, trim off the remainder.

    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    size = len(data)
    nbatch = size // batch_size
    data_to_use = data[: nbatch * batch_size]
    return np.array(data_to_use, dtype=dtype).reshape((batch_size, nbatch)).transpose()


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.

    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length

    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    nd_device = None
    if isinstance(device, ndl.nd_backend.CUDADevice):
        nd_device = nd.cuda()
    elif isinstance(device, ndl.nd_backend.CPUDevice):
        nd_device = nd.cpu()

    data = batches[i: i+bptt, :]
    target = batches[i+1: i+bptt+1, :].reshape((bptt * batches.shape[1],))
    data_tensor = Tensor(nd.array(data, dtype=dtype, device=nd_device), dtype=dtype, device=device)
    target_tensor = Tensor(nd.array(target, dtype=dtype, device=nd_device), dtype=dtype, device=device)
    return data_tensor, target_tensor


# load cora dataset
def load_cora(cora_root, device=None):
    print('Loading cora dataset...')
    def onehot_encode(labels):
        classes = set(labels)
        class_mapping = {label: idx for idx, label in enumerate(classes)}
        mapper = lambda label: class_mapping[label]
        mapping_func = np.vectorize(mapper)
        label_targets = mapping_func(labels)
        onehot = np.eye(len(classes))[label_targets]
        return label_targets, onehot

    def get_adjacency_matrix(graph, ids):
        id_mapping = {id: idx for idx, id in enumerate(ids)}
        adjacency_matrix = np.zeros((len(ids), len(ids)), dtype=np.float32)
        for edge in graph:
            origin = id_mapping[edge[1]]
            dest = id_mapping[edge[0]]
            adjacency_matrix[origin][dest] = 1
        return adjacency_matrix, id_mapping

    def dense_to_sparse_tensor(mat):
        row_idx = []
        col_idx = []
        values = []
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                if mat[row][col] != 0.0:
                    row_idx.append(row)
                    col_idx.append(col)
                    values.append(1.0)
        indices = np.array((row_idx, col_idx))
        values = np.array(values, dtype=np.float32)
        return sparse_coo_tensor(indices=indices, values=values, data_shape=mat.shape, device=device)

    id_feat_label = np.loadtxt(os.path.join(cora_root, 'cora.content'), dtype=np.dtype(str))
    graph = np.loadtxt(os.path.join(cora_root, 'cora.cites'), dtype=np.int32)

    ids = id_feat_label[:, 0].astype(np.int32)
    feats = id_feat_label[:, 1:-1].astype(np.float32)
    labels, labels_onehot = onehot_encode(id_feat_label[:, -1])
    adjacency_matrix, id_mapping = get_adjacency_matrix(graph, ids)

    print('Converting features and adjacency matrix to Needle sparse coo format...')
    feats = dense_to_sparse_tensor(feats)
    labels = Tensor(labels, device=device)
    labels_onehot = Tensor(labels_onehot, device=device)
    adjacency_matrix = dense_to_sparse_tensor(adjacency_matrix)

    return feats, labels, labels_onehot, adjacency_matrix
import torch
import numpy as np


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, use_shared_mem=False):
    """Inner Implementation of the Pad batchify
    Parameters
    ----------
    arrs : list
    pad_axis : int
    pad_val : number
    use_shared_mem : bool, default False
    Returns
    -------
    ret : NDArray
    original_length : NDArray
    """
    if not isinstance(arrs[0], (torch.Tensor, np.ndarray)):
        arrs = [np.asarray(ele) for ele in arrs]
    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)
    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs),) + tuple(ret_shape)
    ret = torch.full(size=ret_shape, fill_value=pad_val, dtype=torch.float32)
    original_length = torch.tensor(original_length, dtype=torch.int32)
    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        else:
            if isinstance(arr, np.ndarray):
                slices = [slice(None) for _ in range(arr.ndim)]
            else:
                slices = [slice(None) for _ in range(arr.ndimension())]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            slices = [slice(i, i + 1)] + slices
            ret[tuple(slices)] = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    return ret, original_length


def _stack_arrs(arrs, use_shared_mem=False):
    """Internal imple for stacking arrays."""
    if isinstance(arrs[0], torch.Tensor):
        if use_shared_mem:
            out = torch.empty((len(arrs),) + arrs[0].shape, dtype=arrs[0].dtype)
            return torch.stack(arrs, out=out)
        else:
            return torch.stack(arrs)
    else:
        out = np.asarray(arrs)
        return torch.Tensor(out)


class Stack(object):
    r"""Stack the input data samples to construct the batch.
    The N input samples must have the same shape/length and will be stacked to construct a batch.
    Examples
    --------
    >>> from data import batchify
    >>> # Stack multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6, 8]
    >>> c = [8, 9, 1, 2]
    >>> batchify.Stack()([a, b, c])
    [[1. 2. 3. 4.]
     [4. 5. 6. 8.]
     [8. 9. 1. 2.]]
    >>> # Stack multiple numpy.ndarrays
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> batchify.Stack()([a, b])
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    >>> # Stack multiple NDArrays
    >>> import torch
    >>> a = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = torch.Tensor([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> batchify.Stack()([a, b])
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    """

    def __call__(self, data):
        """Batchify the input data
        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : NDArray
        """
        return _stack_arrs(data, True)


class Pad(object):
    """Pad the input ndarrays along the specific padding axis and stack them to get the output.
    Input of the function will be N samples. Each sample should contain a single element that
    can be 1) numpy.ndarray, 2) torch.Tensor, 3) list of numbers.
    You can set the `axis` and `pad_val` to determine the padding axis and
    value.
    The arrays will be padded to the largest dimension at `axis` and then
    stacked to form the final output. In addition, the function will output the original dimensions
    at the `axis` if ret_length is turned on.
    Parameters
    ----------
    axis : int, default 0
        The axis to pad the arrays. The arrays will be padded to the largest dimension at
        pad_axis. For example, assume the input arrays have shape
        (10, 8, 5), (6, 8, 5), (3, 8, 5) and the pad_axis is 0. Each input will be padded into
        (10, 8, 5) and then stacked to form the final output.
    pad_val : float or int, default 0
        The padding value.
    ret_length : bool, default False
        Whether to return the valid length in the output.
    Examples
    --------
    >>> from data import batchify
    >>> # Inputs are multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Pad()([a, b, c])
    [[ 1  2  3  4]
     [ 4  5  6  0]
     [ 8  2  0  0]]
    >>> # Also output the lengths
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Pad(ret_length=True)([a, b, c])
    (
     [[1 2 3 4]
      [4 5 6 0]
      [8 2 0 0]]

     [4 3 2])
    >>> # Inputs are multiple ndarrays
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 8], [1, 2]])
    >>> batchify.Pad(axis=1, pad_val=-1)([a, b])
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
     [[ 5  8 -1 -1]
      [ 1  2 -1 -1]]]
    >>> # Inputs are multiple NDArrays
    >>> import torch
    >>> a = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = torch.Tensor([[5, 8], [1, 2]])
    >>> batchify.Pad(axis=1, pad_val=-1)([a, b])
    [[[ 1.  2.  3.  4.]
      [ 5.  6.  7.  8.]]
     [[ 5.  8. -1. -1.]
      [ 1.  2. -1. -1.]]]
    """

    def __init__(self, axis=0, pad_val=0, ret_length=False):
        self._axis = axis
        assert isinstance(axis, int), 'axis must be an integer! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._pad_val = pad_val
        self._ret_length = ret_length

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            A list of N samples. Each sample can be 1) ndarray or
             2) a list/tuple of ndarrays
        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)
        valid_length: NDArray, optional
            The sequences' original lengths at the padded axis. Shape is (N,). This will only be
            returned in `ret_length` is True.
        """
        if isinstance(data[0], (torch.Tensor, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(data, self._axis,
                                                                  self._pad_val, True)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


class Tuple(object):
    """Wrap multiple batchify functions to form a function apply each input function on each
    input fields respectively.
    Each data sample should be a list or tuple containing multiple attributes. The `i`th batchify
    function stored in `Tuple` will be applied on the `i`th attribute. For example, each
    data sample is (nd_data, label). You can wrap two batchify functions using
    `Wrap(DataBatchify, LabelBatchify)` to batchify nd_data and label correspondingly.
    Parameters
    ----------
    fn : list or tuple or callable
        The batchify functions to wrap.
    *args : tuple of callable
        The additional batchify functions to wrap.
    Examples
    --------
    >>> from data import batchify
    >>> a = ([1, 2, 3, 4], 0)
    >>> b = ([5, 7], 1)
    >>> c = ([1, 2, 3, 4, 5, 6, 7], 0)
    >>> batchify.Tuple(batchify.Pad(), batchify.Stack())([a, b])
    (
     [[1 2 3 4]
      [5 7 0 0]],

     [0. 1.])
    >>> # Input can also be a list
    >>> batchify.Tuple([batchify.Pad(), batchify.Stack()])([a, b])
    (
     [[1 2 3 4]
      [5 7 0 0]],

     [0. 1.])
    >>> # Another example
    >>> a = ([1, 2, 3, 4], [5, 6], 1)
    >>> b = ([1, 2], [3, 4, 5, 6], 0)
    >>> c = ([1], [2, 3, 4, 5, 6], 0)
    >>> batchify.Tuple(batchify.Pad(), batchify.Pad(), batchify.Stack())([a, b, c])
    (
     [[1 2 3 4]
      [1 2 0 0]
      [1 0 0 0]],

     [[5 6 0 0 0]
      [3 4 5 6 0]
      [2 3 4 5 6]],

     [1. 0. 0.])
    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn,) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):
        """Batchify the input data.

        Parameters
        ----------
        data : list
            The samples to batchfy. Each sample should contain N attributes.
        Returns
        -------
        ret : tuple
            A tuple of length N. Contains the batchified result of each attribute in the input.
        """
        assert len(data[0]) == len(self._fn), \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(len(self._fn), len(data[0]))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)


class Empty(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, data):
        return data

# if __name__ == '__main__':
#     import torch
#
#     a = ([1, 2, 3, 4], 0)
#     b = ([5, 7], 1)
#     c = ([1, 2, 3, 4, 5, 6, 7], 0)
#     out = Tuple(Pad(), Stack())([a, b])
#     print(out)

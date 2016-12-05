# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN math functions."""

import tensorflow as tf
import math
import numpy as np
from libspn import utils

gather_columns_module = tf.load_op_library('/home/nash/Dropbox/KTH/Projects/tensorflow/tensorflow/core/user_ops/gather_columns.so')

class ValueType:

    """A class specifying various types of values that be passed to the SPN
    graph."""

    class RANDOM_UNIFORM:

        """A random value from a uniform distribution.

        Attributes:
            min_val: The lower bound of the range of random values.
            max_val: The upper bound of the range of random values.
        """

        def __init__(self, min_val=0, max_val=1):
            self.min_val = min_val
            self.max_val = max_val
            # TODO: Move to metaclass
            utils.register_serializable(type(self))

        def serialize(self):
            return {'min_val': self.min_val,
                    'max_val': self.max_val}

        def deserialize(self, data):
            self.min_val = data['min_val']
            self.max_val = data['max_val']


def gather_cols(params, indices, name=None, use_gather_nd=True):
    """Gather columns of a 2D tensor or values of a 1D tensor.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D integer array.
        name (str): A name for the operation (optional).
        use_gather_nd (bool): Use ``transpose`` and ``gather_nd`` instead of
                              ``gather`` when gathering multiple columns from
                              a 2D params tensor.

    Returns:
        Tensor: Has the same dtype and number of dimensions and type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols"):
        params = tf.convert_to_tensor(params, name="params")
        indices = np.asarray(indices)
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check indices
        if indices.ndim != 1:
            raise ValueError("'indices' must be 1D")
        if indices.size < 1:
            raise ValueError("'indices' cannot be empty")
        if not issubclass(indices.dtype.type, np.integer):
            raise ValueError("'indices' must be integer, not %s"
                             % indices.dtype)
        if np.any((indices < 0) | (indices >= param_size)):
            raise ValueError("'indices' must fit the the indexed dimension")
        # Define op
        if param_size == 1:
            # Single column tensor, indices must include it, just forward tensor
            return params
        elif indices.size == 1:
            index = indices[0]
            # Gathering a single column, just slice
            if param_dims == 1:
                return tf.slice(params, [index], [1])
            else:
                return tf.slice(params, [0, index], [-1, 1])
        else:
            # Gathering multiple columns from multi-column tensor
            if param_dims == 1:
                return tf.gather(params, indices)
            else:
                # Two possibilities to deal with 2d gathering
                if use_gather_nd:
                    return tf.transpose(tf.gather_nd(tf.transpose(params),
                                                     np.expand_dims(indices, 1)))
                else:
                    p_shape = tf.shape(params)
                    p_flat = tf.reshape(params, [-1])
                    i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                                   [-1, 1]) + indices, [-1])
                    return tf.reshape(tf.gather(p_flat, i_flat),
                                      [p_shape[0], -1])


def scatter_cols(params, indices, out_num_cols, name=None):
    """Scatter columns of a 2D tensor or values of a 1D tensor into a tensor
    with the same number of dimensions and ``out_num_cols`` columns or values.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D integer array indexing the columns in the
                              output array to which ``params`` is scattered.
        num_cols (int): The number of columns in the output tensor.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype and number of dimensions as ``params``.
    """
    with tf.op_scope([params, indices], name, "scatter_cols"):
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = np.asarray(indices)
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check num_cols
        if not isinstance(out_num_cols, int):
            raise ValueError("'out_num_cols' must be integer, not %s"
                             % type(out_num_cols))
        if out_num_cols < param_size:
            raise ValueError("'out_num_cols' must be larger than the size of "
                             "the indexed dimension of 'params'")
        # Check indices
        if indices.ndim != 1:
            raise ValueError("'indices' must be 1D")
        if indices.size != param_size:
            raise ValueError("Sizes of 'indices' and the indexed dimension of "
                             "'params' must be the same")
        if not issubclass(indices.dtype.type, np.integer):
            raise ValueError("'indices' must be integer, not %s"
                             % indices.dtype)
        if np.any((indices < 0) | (indices >= out_num_cols)):
            raise ValueError("'indices' must be smaller than 'out_num_cols'")
        if len(set(indices)) != len(indices):
            raise ValueError("'indices' cannot contain duplicates")
        # Define op
        if out_num_cols == 1:
            # Scatter to a single column tensor, it must be from 1 column
            # tensor and the indices must include it. Just forward the tensor.
            return params
        elif param_size == 1:
            # Scatter a single column tensor to a multi-column tensor
            # Just pad with zeros
            if param_dims == 1:
                return tf.pad(params, [[indices[0], out_num_cols - indices[0] - 1]])
            else:
                return tf.pad(params, [[0, 0],
                                       [indices[0], out_num_cols - indices[0] - 1]])
        else:
            # Scatte a multi-column tensor to a multi-column tensor
            if param_dims == 1:
                with_zeros = tf.concat(concat_dim=0, values=([0], params))
                gather_indices = np.zeros(out_num_cols, dtype=int)
                gather_indices[indices] = np.arange(indices.size) + 1
                return gather_cols(with_zeros, gather_indices)
            else:
                zero_col = tf.zeros((tf.shape(params)[0], 1), dtype=params.dtype)
                with_zeros = tf.concat(concat_dim=1, values=(zero_col, params))
                gather_indices = np.zeros(out_num_cols, dtype=int)
                gather_indices[indices] = np.arange(indices.size) + 1
                return gather_cols(with_zeros, gather_indices)


def broadcast_value(value, shape, dtype, name=None):
    """Broadcast the given value to the given shape and dtype. If ``value`` is
    one of the members of :class:`~libspn.ValueType`, the requested value will
    be generated and placed in every element of a tensor of the requested shape
    and dtype. If ``value`` is a 0-D tensor or a Python value, it will be
    broadcasted to the requested shape and converted to the requested dtype.
    Otherwise, the value is used as is.

    Args:
        value: The input value.
        shape: The shape of the output.
        dtype: The type of the output.

    Return:
        Tensor: A tensor containing the broadcasted and converted value.
    """
    with tf.op_scope([value], name, "broadcast_value"):
        # Recognize ValueTypes
        if isinstance(value, ValueType.RANDOM_UNIFORM):
            return tf.random_uniform(shape=shape,
                                     minval=value.min_val,
                                     maxval=value.max_val,
                                     dtype=dtype)

        # Broadcast tensors and scalars
        tensor = tf.convert_to_tensor(value, dtype=dtype)
        if tensor.get_shape() == tuple():
            return tf.fill(dims=shape, value=tensor)

        # Return original input if we cannot broadcast
        return tensor


def normalize_tensor(tensor, name=None):
    """Normalize the tensor so that all elements sum to 1.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    with tf.op_scope([tensor], name, "normalize_tensor"):
        tensor = tf.convert_to_tensor(tensor)
        s = tf.reduce_sum(tensor)
        return tf.truediv(tensor, s)


def reduce_log_sum(log_input, name=None):
    """Calculate log of a sum of elements of a tensor containing log values
    row-wise.

    Args:
        log_input (Tensor): Tensor containing log values.

    Returns:
        Tensor: The reduced tensor of shape ``(None, 1)``, where the first
        dimension corresponds to the first dimension of ``log_input``.
    """
    # WARNING: As described here:
    # http://stackoverflow.com/questions/39211546/bug-in-tensorflow-reduce-max-for-negative-infinity
    # there clearly is a problem with reduce_max which returns min float32
    # instead of -inf for negative infinity inputs. At the same time,
    # tf.maximum works as expected. The final result is still correct, and
    # actually will lead to a simpler code since the -inf detection is
    # not needed in such case. But it is unclear if this behavior of reduce_max is
    # stable or a bug that will be removed. For now, we include the -inf
    # detection in case this bug gets fixed one day.
    with tf.op_scope([log_input], name, "reduce_log_sum"):
        log_max = tf.reduce_max(log_input, 1, keep_dims=True)
        # Compute the value assuming at least one input is not -inf
        log_rebased = tf.sub(log_input, log_max)
        out_normal = log_max + tf.log(tf.reduce_sum(tf.exp(log_rebased),
                                                    1, keep_dims=True))
        # Check if all input values in a row are -inf (all non-log inputs are 0)
        # and produce output for that case
        all_zero = tf.equal(log_max,
                            tf.constant(-math.inf, dtype=log_input.dtype))
        out_zeros = tf.fill(tf.shape(out_normal),
                            tf.constant(-math.inf, dtype=log_input.dtype))
        # Choose the output for each row
        return tf.select(all_zero, out_zeros, out_normal)


def concat_maybe(concat_dim, values, name='concat'):
    """Concatenate values if there is more than one value. Oherwise, just
    forward value as is.

    Args:
        values (list of Tensor): Values to concatenate

    Returns:
        Tensor: Concatenated values.
    """
    if len(values) > 1:
        return tf.concat(concat_dim, values)
    else:
        return values[0]


def split(split_dim, split_sizes, value, name=None):
    """Split ``value`` into multiple tensors of sizes given by ``split_sizes``.
    ``split_sizes`` must sum to the size of ``split_dim``. If only one split_size
    is given, the function does nothing and just forwards the value as the only split.

    Args:
        split_dim (int): The dimensions along which to split.
        split_sizes (list of int): Sizes of each split.
        value (Tensor): The tensor to split.

    Returns:
        list of Tensor: List of resulting tensors.
    """
    # Check input
    if split_dim > 1:
        raise NotImplementedError("split only works for dimensions 0 and 1")
    split_sizes = np.asarray(split_sizes)
    # Anything to split?
    if split_sizes.size == 1:
        return [value]
    # Split
    with tf.op_scope([value], name, "split"):
        # Check input
        value = tf.convert_to_tensor(value, name="params")
        value_dims = value.get_shape().ndims
        if value_dims < 1 or value_dims > 2:
            raise NotImplementedError("split only works for 1D or 2D values")
        # Split
        slice_indices = np.cumsum(split_sizes)
        if split_dim == 0:
            if value_dims == 1:
                split = [value[start:stop] for start, stop in
                         zip(np.r_[0, slice_indices], slice_indices)]
            elif value_dims == 2:
                split = [value[start:stop, :] for start, stop in
                         zip(np.r_[0, slice_indices], slice_indices)]
        elif split_dim == 1:
            split = [value[:, start:stop] for start, stop in
                     zip(np.r_[0, slice_indices], slice_indices)]
    return split


def print_tensor(*tensors):
    return tf.Print(tensors[0], tensors)

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.graph.node import ParamNode
from libspn.graph.algorithms import traverse_graph
from libspn import conf
from libspn import utils


class Weights(ParamNode):
    """A node containing a vector of weights of a sum node.

    Args:
        init_value: Initial value of the weights. For possible values, see
                    :meth:`~libspn.utils.broadcast_value`.
        num_weights (int): Number of weights in the vector.
        num_sums (int): Number of sum nodes the weight vector/matrix represents.
        name (str): Name of the node.

    Attributes:
        trainable(bool): Should these weights be updated during training?
    """

    def __init__(self, init_value=1, num_weights=1,
                 num_sums=1, trainable=True, name="Weights"):
        if not isinstance(num_weights, int) or num_weights < 1:
            raise ValueError("num_weights must be a positive integer")

        if not isinstance(num_sums, int) or num_sums < 1:
            raise ValueError("num_sums must be a positive integer")

        self._init_value = init_value
        self._num_weights = num_weights
        self._num_sums = num_sums
        self._trainable = trainable
        super().__init__(name)

    def serialize(self):
        data = super().serialize()
        data['num_weights'] = self._num_weights
        data['num_sums'] = self._num_sums
        data['trainable'] = self._trainable
        # If we're in session and the variable is set, get its value,
        # otherwise get init value
        sess = tf.get_default_session()
        if (sess is not None and
                sess.run(tf.is_variable_initialized(self._variable))):
            data['value'] = sess.run(self._variable).tolist()
        else:
            data['value'] = self._init_value
        return data

    def deserialize(self, data):
        self._init_value = data['value']
        self._num_weights = data['num_weights']
        self._num_sums = data['num_sums']
        self._trainable = data['trainable']
        super().deserialize(data)

    @property
    def num_weights(self):
        """int: Number of weights in the vector."""
        return self._num_weights

    @property
    def num_sums(self):
        """int: Number of sum nodes the weights vector/matrix represents."""
        return self._num_sums

    @property
    def variable(self):
        """Variable: The TF variable of shape ``[num_sums, num_weights]`` holding the
        weights."""
        return self._variable

    def initialize(self):
        """Return a TF operation assigning the initial value to the weights.

        Returns:
            Tensor: The initialization assignment operation.
        """
        return self._variable.initializer

    def assign(self, value):
        """Return a TF operation assigning values to the weights.

        Args:
            value: The value to assign to the weights. For possible values, see
                   :meth:`~libspn.utils.broadcast_value`.

        Returns:
            Tensor: The assignment operation.
        """
        value = utils.broadcast_value(value, (self._num_weights,),
                                      dtype=conf.dtype)
        value = utils.normalize_tensor(value)
        return tf.assign(self._variable, value)

    def _create(self):
        """Creates a TF variable holding the vector of the SPN weights.

        Returns:
            Variable: A TF variable of shape ``[num_weights]``.
        """
        init_val = utils.broadcast_value(self._init_value,
                                         (self._num_weights,),
                                         dtype=conf.dtype)
        init_val = utils.normalize_tensor_2D(init_val, self._num_weights, self._num_sums)
        self._variable = tf.Variable(init_val, dtype=conf.dtype,
                                     collections=['spn_weights'])

    def _compute_out_size(self):
        return self._num_weights

    def _compute_value(self):
        return self._variable

    def _compute_hard_em_update(self, counts):
        return tf.reduce_sum(counts, 0)


def assign_weights(root, value):
    """Generate an assign operation assigning a value to all the weights in
    the SPN graph rooted in ``root``.

    Args:
        root (Node): The root node of the SPN graph.
        value: The value to assign to the weights. For possible values, see
               :meth:`~libspn.utils.broadcast_value`.
    """
    assign_ops = []

    def assign(node):
        if isinstance(node, Weights):
            assign_ops.append(node.assign(value))

    # Get all assignment operations
    traverse_graph(root, fun=assign, skip_params=False)

    # Return a collective operation
    return tf.group(*assign_ops)


def initialize_weights(root):
    """Generate an assign operation initializing all the sum weights in the SPN
    graph rooted in ``root``.

    Args:
        root (Node): The root node of the SPN graph.
    """
    assign_ops = []

    def assign(node):
        if isinstance(node, Weights):
            assign_ops.append(node.initialize())

    # Get all assignment operations
    traverse_graph(root, fun=assign, skip_params=False)

    # Return collective operation
    return tf.group(*assign_ops, name="initialize_weights")

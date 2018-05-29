# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain, combinations, product
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn.utils.serialization import register_serializable

import numpy as np


@register_serializable
class PermProducts(OpNode):
    """A node representing multiple products, permuted over the input space, in
       an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values. The only
            criterion for the input is that all inputs, in the list, should
            have the same dimention.
        name (str): Name of the node.
    """

    logger = get_logger()
    info = logger.info

    def __init__(self, *values, name="PermProducts"):
        self._values = []
        super().__init__(InferenceType.MARGINAL, name)
        self.set_values(*values)

        self.create_products()

    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])
        self.create_products()

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return self._values

    @property
    def num_prods(self):
        """int: Number of Product ops modelled by this node."""
        return self._num_prods

    @property
    def values(self):
        """list of Input: List of value inputs."""
        return self._values

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._parse_inputs(*values)

    def create_products(self):
        """Based on the number and size of inputs connected to this node, model
        products by permuting over the inputs.
        """
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        self._input_sizes = list(self.get_input_sizes())
        self._num_inputs = len(self._input_sizes)

        # Calculate number of products this node would model.
        if self._num_inputs == 1:
            self._num_prods = 1
        else:
            self._num_prods = int(np.prod(self._input_sizes))

        # Create indices by permuting over the input space, such that inputs
        # for the products can be generated by gathering from concatenated
        # input values.
        self._permuted_indices = self.permute_indices(self._input_sizes)

    def permute_indices(self, input_sizes):
        """Create indices by permuting over the inputs, such that inputs for each
        product modeled by this node can be generated by gathering from concatenated
        values of the node.

        Args:
            inputs_sizes (list): List of input sizes.

        Return:
            permuted_indices (list): List of indices for gathring inputs of all
                                     the product nodes modeled by this Op.
        """
        ind_range = np.cumsum([0] + input_sizes)
        ind_list = list(product(*[range(start, stop)
                                  for start, stop in zip(ind_range,
                                                         ind_range[1:])]))

        return list(chain(*ind_list))

    def add_values(self, *values):
        """Add more inputs providing input values to this node. Then remodel the
        products based on the newly added inputs.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._values + self._parse_inputs(*values)
        self.create_products()

    @property
    def _const_out_size(self):
        return True

    def _compute_out_size(self, *input_out_sizes):
        return self._num_prods

    def _compute_scope(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes = self._gather_input_scopes(*value_scopes)
        value_scopes_list = [Scope.merge_scopes(pvs) for pvs in product(*[vs for
                             vs in value_scopes])]
        return [value_scopes_list[0]] if self._num_prods == 1 else value_scopes_list

    def _compute_valid(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes_ = self._gather_input_scopes(*value_scopes)
        # If already invalid, return None
        if any(s is None for s in value_scopes_):
            return None
        # Check product decomposability
        permuted_value_scopes = list(product(*value_scopes_))
        for perm_val_scope in permuted_value_scopes:
            for s1, s2 in combinations(perm_val_scope, 2):
                if s1 & s2:
                    PermProducts.info("%s is not decomposable with input value "
                                      "scopes %s", self, value_scopes_)
                    return None
        return self._compute_scope(*value_scopes)

    @utils.lru_cache
    def _compute_value_common(self, *value_tensors):
        """Common actions when computing value."""
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        # Prepare values
        value_tensors = self._gather_input_tensors(*value_tensors)
        if len(value_tensors) > 1:
            values = tf.concat(values=value_tensors, axis=1)
        else:
            values = value_tensors[0]
        if self._num_prods > 1:
            # Gather values based on permuted_indices
            permuted_values = utils.gather_cols(values, self._permuted_indices)

            # Shape of values tensor = [Batch, (num_prods * num_vals)]
            # First, split the values tensor into 'num_prods' smaller tensors.
            # Then pack the split tensors together such that the new shape
            # of values tensor = [Batch, num_prods, num_vals]
            reshape = (-1, self._num_prods, int(permuted_values.shape[1].value /
                                                self._num_prods))
            reshaped_values = tf.reshape(permuted_values, shape=reshape)
            return reshaped_values
        else:
            return values

    @utils.lru_cache
    def _compute_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_prod(values, axis=-1,
                              keep_dims=(False if self._num_prods > 1 else True))

    @utils.lru_cache
    def _compute_log_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        @tf.custom_gradient
        def value_gradient(*value_tensors):
            def gradient(gradients):
                scattered_grads = self._compute_mpe_path(gradients, *value_tensors)
                return [sg for sg in scattered_grads if sg is not None]
            return tf.reduce_sum(values, axis=-1, keep_dims=(False if self._num_prods > 1
                                                             else True)), gradient
        return value_gradient(*value_tensors)

    def _compute_mpe_value(self, *value_tensors):
        return self._compute_value(*value_tensors)

    def _compute_log_mpe_value(self, *value_tensors):
        return self._compute_log_value(*value_tensors)

    @utils.lru_cache
    def _compute_mpe_path(self, counts, *value_values, add_random=False,
                          use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        # Path per product node is calculated by permuting backwards to the
        # input nodes, then adding the appropriate counts per input, and then
        # scattering the summed counts to value inputs

        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        def permute_counts(input_sizes):
            # Function that permutes count values, backward to inputs.
            counts_indices_list = []

            def range_with_blocksize(start, stop, block_size, step):
                # A function that produces an arithmetic progression (Similar to
                # Python's range() function), but for a given block-size of
                # consecutive numbers.
                # E.g: range_with_blocksize(start=0, stop=20, block_size=3, step=5)
                # = [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]
                counts_indices = []
                it = 0
                low = start
                high = low + block_size
                while low < stop:
                    counts_indices = counts_indices + list(range(low, high))
                    it += 1
                    low = start + (it * step)
                    high = low + block_size

                return counts_indices

            for inp, inp_size in enumerate(input_sizes):
                block_size = int(self._num_prods / np.prod(input_sizes[:inp+1]))
                step = int(np.prod(input_sizes[inp:]))
                for i in range(inp_size):
                    start = i * block_size
                    stop = self._num_prods - (block_size * (inp_size-i-1))
                    counts_indices_list.append(range_with_blocksize(start, stop,
                                                                    block_size,
                                                                    step))

            return counts_indices_list

        if(len(self._input_sizes) > 1):
            permuted_indices = permute_counts(self._input_sizes)
            summed_counts = tf.reduce_sum(utils.gather_cols_3d(counts, permuted_indices),
                                          axis=-1)
            processed_counts_list = tf.split(summed_counts, self._input_sizes, axis=-1)
        else:  # For single input case, i.e, when _num_prods = 1
            summed_counts = self._input_sizes[0] * [counts]
            processed_counts_list = [tf.concat(values=summed_counts, axis=-1)]

        # Zip lists of processed counts and value_values together for scattering
        value_counts = zip(processed_counts_list, value_values)

        return self._scatter_to_input_tensors(*value_counts)

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False,
                              use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path(counts, *value_values)

    def _compute_log_gradient(self, gradients, *value_values, with_ivs=False):
        return self._compute_mpe_path(gradients, *value_values)

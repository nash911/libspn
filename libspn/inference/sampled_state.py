# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.inference.sampled_path import SampledPath


class SampledState():
    """Assembles TF operations computing Sampled state for an SPN.

    Args:
        sampled_path (SampledPath): Pre-computed Sampled_path.
        value (Value or LogValue): Pre-computed SPN values.  Ignored if
            ``sampled_path`` is given.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``sampled_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
            if ``sampled_path`` is given.
    """

    def __init__(self, sampled_path=None, value=None, log=True, value_inference_type=None):
        # Create internal Sampled path generator
        if sampled_path is None:
            self._sampled_path = SampledPath(log=log, value=value,
                                             value_inference_type=value_inference_type)
        else:
            self._sampled_path = sampled_path

    @property
    def sampled_path(self):
        """SampledPath: Computed Sampled path."""
        return self._sampled_path

    def get_state(self, root, *var_nodes):
        """Assemble TF operations computing the Sampled state of the given SPN
        variables for the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
            *var_nodes (VarNode): Variable nodes for which the state should
                                  be computed.

        Returns:
            list of Tensor: A list of tensors containing the Sampled state for the
            variable nodes.
        """
        # Generate path if not yet generated
        if not self._sampled_path.counts:
            self._sampled_path.get_probable_path(root)

        with tf.name_scope("SampledState"):
            return tuple(var_node._compute_probable_state(
                self._sampled_path.counts[var_node])
                for var_node in var_nodes)

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.graph.algorithms import traverse_graph
from libspn.graph.sum import Sum
from libspn.graph.product import Product
from libspn.log import get_logger
from libspn.exceptions import StructureError
from collections import defaultdict

logger = get_logger()


class Pruning():
    """Performs graph pruning.

    Args:
        root (Node): The root of the SPN graph.
        threshold: A value between [0, 1) that will be used for thresholding weights.
    """

    def __init__(self, root, threshold=None):
        self._root = root
        self.set_threshold(threshold)

        # TODO: Is this needed?
        # # Create a name scope
        # with tf.name_scope("Pruning") as self._name_scope:
        #     pass

    def set_threshold(self, threshold=None):
        """Set threshold value.

        Args:
            threshold: A value between [0, 1) that will be used for thresholding
                       weights.
        """
        if threshold is not None and (threshold < 0 or threshold >= 1.0):
            raise ValueError("'threshold' must be in the range [0, 1)")
        self._threshold = threshold

    def save_param_values(self, sess=None):
        """Run sessions to pull and save the latest values of all ParamNodes in
            the SPN graph, from TF graph.

        Args:
            sess (Session): Optional. Session used to assign parameter values.
                            If ``None``, the default session is used.
        """
        # Check session
        if sess is None:
            sess = tf.get_default_session()
        if sess is None:
            logger.debug1("No valid session found, parameter values will not "
                          "be saved!")

        def sess_fun(node):
            """Run a session and pull the latest values of a ParamNode. Save the
                pulled values in the respective node.

            Args:
                node (Node): Node for which its values needs to be pulled from
                             TF graph.
            """
            if node.is_param:
                param_vals = sess.run(node.variable)
                node.set_current_value(param_vals.tolist())

        # Traverse the graph breadth-first, and pull values of all ParamNodes
        # from TF graph
        traverse_graph(self._root, fun=sess_fun, skip_params=False)

    def prune(self, threshold=None):
        """Perform pruning on the SPN.

        Args:
            threshold: A value between [0, 1) that will be used for thresholding
                weights.
        """

        if threshold is not None:
            self.set_threshold(threshold)

        if self._threshold is None:
            raise StructureError("%s is missing threshold value" % self)

        # Parents-dictionary: Key - Each node in the graph.
        #                     Value - All parents of a node.
        parents = defaultdict(list)

        def add_to_parents_dict(node):
            """Add 'node' as a parent in the parents-dictionary of each of its
            children OpNodes.

            Args:
                node (OpNode): The OpNode to be added to parents-dictionary of
                    connected children.
            """
            if node.is_op:
                for nr, vals in enumerate(node.values):
                    if vals.is_op:
                        parents[vals.node].append((node, nr))

        # Traverse the graph breadth-first, and record parent information of all
        # OpNodes
        traverse_graph(self._root, fun=add_to_parents_dict, skip_params=True)

        def prune_node(node, child_to_kill=None, kill_self=False):
            """Perform pruning on a single OpNode 'node'.

            For a Sum node: (i) Remove all children OpNodes whose weights are
            below a certain threshold, and (ii) remove the child OpNode which is
            indexed by 'child_to_kill'.

            For a Product node, if 'kill_self' is True, then remove it from all
            it's connected parents, thereby removing it from the graph.

            Args:
                node (OpNode): An OpNode to be evaluated to be for pruning.
                child_to_kill (int): Index of a Sum node's child to be removed
                    from it's list of value inputs. Will always be None for a
                    Product node.
                kill_self (bool): To indicate if a Product node has to be removed
                    from the network, by removing itself from each of it's parents.
                    Will always be False for a Sum node.
            """

            def remove_from_parents_dict(node):
                """Remove 'node' from parents-dictionary of each of its children.

                Args:
                    node (OpNode): The OpNode to be removed from
                        parents-dictionary of connected children.
                """

                for vals in node.values:
                    if vals.is_op:
                        try:
                            for i, (p, nr) in enumerate(parents[vals.node]):
                                if p is node:
                                    del parents[vals.node][i]
                        except KeyError:
                            pass

            # Sum node
            if isinstance(node, Sum):
                # Remove Sum node as a parent from each of its connected children's
                # parents-dictionary.
                remove_from_parents_dict(node)

                # Get values, weights and ivs inputs of the Sum node.
                values = list(node.values)
                weights = node.weights.node.current_value
                ivs = node.ivs
                if ivs and ivs.indices:
                    ivs_ind = ivs.indices
                else:
                    ivs_ind = list(range(len(values)))

                new_weights = []
                new_values = []
                new_ivs_ind = []

                if child_to_kill is None:
                    # Evaluate each connected value input, and if the weight is
                    # below the threshold, then remove it.
                    for w, v, iv in zip(weights, values, ivs_ind):
                        if(w > self._threshold):
                            new_weights.append(w)
                            new_values.append(v)
                            new_ivs_ind.append(iv)

                else:
                    # Remove the designated child node from its value input, and
                    # the corresponding weight.
                    del values[child_to_kill]
                    del weights[child_to_kill]
                    del ivs_ind[child_to_kill]
                    new_values = values
                    new_weights = weights
                    new_ivs_ind = ivs_ind

                if len(new_values) > 0:
                    # Set the surviving list of children and their respective
                    # weights as inputs.
                    node.set_values(*new_values)
                    node.generate_weights(new_weights)
                    node.weights.node.set_current_value(new_weights)

                    # TODO: @Andrzej: If input-size of a Sum node is 1, should
                    #                 ivs then be disconnected?
                    # if len(new_values) > 1 and ivs:
                    if ivs:
                        node.set_ivs((ivs.node, new_ivs_ind))
                    else:
                        node.set_ivs(None)

                    # Add node as as parent to parents-dictionary of surviving
                    # children
                    add_to_parents_dict(node)
                else:
                    # If there are no surviving children, then this Sum node has
                    # to be killed by killing each of its parent Product nodes.
                    node.set_values(None)
                    node.set_weights(None)
                    node.set_ivs(None)
                    for (p, nr) in parents[node]:
                        prune_node(p, kill_self=True)
            # Product node
            elif isinstance(node, Product):
                if kill_self:
                    # Kill the Product node by removing itself from value inputs
                    # list of each of its parents.
                    while len(parents[node]) > 0:
                        for (p, nr) in parents[node]:
                            prune_node(p, child_to_kill=nr)
                    node.set_values(None)
                else:
                    # Do nothing
                    pass

        # Traverse the graph breadth-first, pruning each OpNode at a time
        traverse_graph(self._root, fun=prune_node, skip_params=True)

        return None

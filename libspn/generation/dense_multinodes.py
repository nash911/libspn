# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from enum import Enum
from collections import deque
from libspn import utils
from libspn.graph.node import Input
from libspn.graph.sum import Sum
from libspn.graph.sums import Sums  # TODO: Remove once replaced with Concat
from libspn.graph.parallelsums import ParallelSums
from libspn.graph.product import Product
from libspn.graph.permproducts import PermProducts
from libspn.graph.concat import Concat
from libspn.log import get_logger
from libspn.exceptions import StructureError
import tensorflow as tf


class DenseSPNGeneratorMultiNodes:
    """Generates a dense SPN according to the algorithm described in
    Poon&Domingos UAI'11.

    Attributes:
        num_decomps (int): Number of decompositions at each level.
        num_subsets (int): Number of variable sub-sets for each decomposition.
        num_mixtures (int): Number of mixtures (sums) for each variable subset.
        input_dist (InputDist): Determines how inputs sharing the same scope
                                (for instance IVs for different values of a
                                random variable) should be included into the
                                generated structure.
        num_input_mixtures (int): Number of mixtures used for combining all
                                  inputs sharing scope when ``input_dist`` is
                                  set to ``MIXTURE``. If set to ``None``,
                                  ``num_mixtures`` is used.
        balanced (bool): Use only balanced decompositions, into subsets of
                         similar cardinality (differing by max 1).
        multi_nodes (bool): Use multi-nodes implementation of Sums and Products
                            while generating the dense graph.
    """

    logger = get_logger()
    debug1 = logger.debug1
    debug2 = logger.debug2

    class InputDist(Enum):
        """Determines how inputs sharing the same scope (for instance IVs for
        different values of a random variable) should be included into the
        generated structure."""

        RAW = 0
        """Each input is considered a different distribution over the scope and
        used directly instead of a mixture as an input to product nodes for
        singleton variable subsets."""

        MIXTURE = 1
        """``input_num_mixtures`` mixtures are created over all the inputs
        sharing a scope, effectively creating ``input_num_mixtures``
        distributions over the scope. These mixtures are then used as inputs
        to product nodes for singleton variable subsets."""

    class SubsetInfo:
        """Stores information about a single subset to be decomposed.

        Attributes:
            level(int): Number of the SPN layer where the subset is decomposed.
            subset(list of tuple of tuple): Subset of inputs to decompose
                                            grouped by scope.
            parents(list of Sum): List of sum nodes mixing the outputs of the
                                  generated decompositions. Should be the root
                                  node at the very top.
        """

        def __init__(self, level, subset, parents):
            self.level = level
            self.subset = subset
            self.parents = parents

    def __init__(self, num_decomps, num_subsets, num_mixtures,
                 input_dist=InputDist.MIXTURE, num_input_mixtures=None,
                 balanced=False, multi_nodes=True):
        # Args
        if not isinstance(num_decomps, int) or num_decomps < 1:
            raise ValueError("num_decomps must be a positive integer")
        if not isinstance(num_subsets, int) or num_subsets < 1:
            raise ValueError("num_subsets must be a positive integer")
        if not isinstance(num_mixtures, int) or num_mixtures < 1:
            raise ValueError("num_mixtures must be a positive integer")
        if input_dist not in DenseSPNGeneratorMultiNodes.InputDist:
            raise ValueError("Incorrect input_dist: %s", input_dist)
        if (num_input_mixtures is not None and
                (not isinstance(num_input_mixtures, int)
                 or num_input_mixtures < 1)):
            raise ValueError("num_input_mixtures must be None"
                             " or a positive integer")

        # Attributes
        self.num_decomps = num_decomps
        self.num_subsets = num_subsets
        self.num_mixtures = num_mixtures
        self.input_dist = input_dist
        self.balanced = balanced
        self.multi_nodes = multi_nodes
        if num_input_mixtures is None:
            self.num_input_mixtures = num_mixtures
        else:
            self.num_input_mixtures = num_input_mixtures

        # Stirling numbers and ratios for partition sampling
        self.__stirling = utils.Stirling()

    def generate(self, *inputs):
        """Generate the SPN.

        Args:
            inputs (input_like): Inputs to the generated SPN.

        Returns:
           Sum: Root node of the generated SPN.
        """
        DenseSPNGeneratorMultiNodes.debug1(
            "Generating dense SPN (num_decomps=%s, num_subsets=%s,"
            " num_mixtures=%s, input_dist=%s, num_input_mixtures=%s)",
            self.num_decomps, self.num_subsets,
            self.num_mixtures, self.input_dist, self.num_input_mixtures)
        inputs = [Input.as_input(i) for i in inputs]
        input_set = self.__generate_set(inputs)
        DenseSPNGeneratorMultiNodes.debug1("Found %s distinct input scopes",
                                           len(input_set))
        root = Sum()

        # Subsets left to process
        subsets = deque()
        subsets.append(DenseSPNGeneratorMultiNodes.SubsetInfo(level=1,
                                                              subset=input_set,
                                                              parents=[root]))

        # Process subsets layer by layer
        self.__decomp_id = 1  # Id number of a decomposition, for info only
        while subsets:
            # Process whole layer (all subsets at the same level)
            level = subsets[0].level
            DenseSPNGeneratorMultiNodes.debug1("Processing level %s", level)
            while subsets and subsets[0].level == level:
                subset = subsets.popleft()
                new_subsets = self.__add_decompositions(subset)
                for s in new_subsets:
                    subsets.append(s)
        return root

    def __generate_set(self, inputs):
        """Generate a set of inputs to the generated SPN grouped by scope.

        Args:
            inputs (list of Input): List of inputs.

        Returns:
           list of tuple of tuple: A list where each elements is a tuple of
               all inputs to the generated SPN which share the same scope.
               Each of that scopes is guaranteed to be unique. That tuple
               contains tuples ``(node, index)`` which uniquely identify
               specific inputs.
        """
        scope_dict = {}  # Dict indexed by scope

        def add_input(scope, node, index):
            try:
                # Try appending to existing scope
                scope_dict[scope].add((node, index))
            except KeyError:
                # Scope not in dict, check if it overlaps with other scopes
                for s in scope_dict:
                    if s & scope:
                        raise StructureError("Differing scopes of inputs overlap")
                # Add to dict
                scope_dict[scope] = set([(node, index)])

        # Process inputs and group by scope
        for inpt in inputs:
            node_scopes = inpt.node.get_scope()
            if inpt.indices is None:
                for index, scope in enumerate(node_scopes):
                    add_input(scope, inpt.node, index)
            else:
                for index in inpt.indices:
                    add_input(node_scopes[index], inpt.node, index)

        # Convert to hashable tuples and sort
        # Sorting might improve performance due to branch prediction
        return [tuple(sorted(i)) for i in scope_dict.values()]

    def __add_decompositions(self, subset_info: SubsetInfo):
        """Add nodes for a single subset, i.e. an instance of ``num_decomps``
        decompositions of ``subset`` into ``num_subsets`` sub-subsets with
        ``num_mixures`` mixtures per sub-subset.

        Args:
            subset_info(SubsetInfo): Info about the subset being decomposed.

        Returns:
            list of SubsetInfo: Info about each new generated subset, which
            requires further decomposition.
        """
        # Get subset partitions
        DenseSPNGeneratorMultiNodes.debug2("Decomposing subset:\n%s", subset_info.subset)
        num_elems = len(subset_info.subset)
        num_subsubsets = min(num_elems, self.num_subsets)  # Requested num subsets
        partitions = utils.random_partitions(subset_info.subset, num_subsubsets,
                                             self.num_decomps,
                                             stirling=self.__stirling,
                                             balanced=self.balanced)
        DenseSPNGeneratorMultiNodes.debug1("Randomized %s decompositions of a subset"
                                           " of %s elements into %s sets",
                                           len(partitions), num_elems, num_subsubsets)

        # Generate nodes for each decomposition/partition
        subsubset_infos = []
        for part in partitions:
            DenseSPNGeneratorMultiNodes.debug1("Decomposition %s: into %s subsubsets",
                                               " of cardinality %s", self.__decomp_id,
                                               len(part), [len(s) for s in part])
            DenseSPNGeneratorMultiNodes.debug2("Decomposition %s subsubsets:\n%s",
                                               self.__decomp_id, part)
            # Handle each subsubset
            sums_id = 1

            prod_inputs = []
            for subsubset in part:
                if self.multi_nodes:
                    if len(subsubset) > 1:  # Decomposable further
                        # Add mixtures
                        with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                            sums = ParallelSums(num_sums=self.num_mixtures,
                                                name="ParallelSums%s.%s" %
                                                (self.__decomp_id, sums_id))
                            sums_id += 1
                        # Register the mixtures as inputs of products
                        prod_inputs.append(sums)
                        # Generate subsubset info
                        subsubset_infos.append(DenseSPNGeneratorMultiNodes.SubsetInfo(
                                               level=subset_info.level + 1,
                                               subset=subsubset, parents=[sums]))
                    else:  # Non-decomposable
                        if self.input_dist == DenseSPNGeneratorMultiNodes.InputDist.RAW:
                            # Concat the content of subset and register as inputs to products
                            # TODO: Concat node makes it's parent node PermProducts, not valid.
                            # prod_inputs.append(Concat(*(list(next(iter(subsubset))))))

                            # TODO: This should be replaced with Concat Op.
                            num_vals = next(iter(subsubset))[0][0]._num_vals
                            sums = Sums(*(list(next(iter(subsubset)))),
                                        num_sums=num_vals,
                                        name="Sums%s.%s" %
                                        (self.__decomp_id, sums_id))

                            prod_inputs.append(sums)
                        elif self.input_dist == DenseSPNGeneratorMultiNodes.InputDist.MIXTURE:
                            # Add mixtures
                            with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                                sums = ParallelSums(*(list(next(iter(subsubset)))),
                                                    num_sums=self.num_input_mixtures,
                                                    name="ParallelSums%s.%s" %
                                                    (self.__decomp_id, sums_id))
                                sums_id += 1
                            # Register the mixtures as inputs of products
                            prod_inputs.append(sums)
                else:
                    if len(subsubset) > 1:  # Decomposable further
                        # Add mixtures
                        with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                            sums = [Sum(name="Sum%s" % (i + 1))
                                    for i in range(self.num_mixtures)]
                            sums_id += 1
                        # Register the mixtures as inputs of products
                        prod_inputs.append([(s, 0) for s in sums])
                        # Generate subsubset info
                        subsubset_infos.append(DenseSPNGeneratorMultiNodes.SubsetInfo(
                            level=subset_info.level + 1, subset=subsubset,
                            parents=sums))
                    else:  # Non-decomposable
                        if self.input_dist == DenseSPNGeneratorMultiNodes.InputDist.RAW:
                            # Register the content of subset as inputs to products
                            prod_inputs.append(next(iter(subsubset)))
                        elif self.input_dist == DenseSPNGeneratorMultiNodes.InputDist.MIXTURE:
                            # Add mixtures
                            with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                                sums = [Sum(name="Sum%s" % (i + 1))
                                        for i in range(self.num_input_mixtures)]
                                sums_id += 1
                            # Register the mixtures as inputs of products
                            prod_inputs.append([(s, 0) for s in sums])
                            # Connect inputs to mixtures
                            for s in sums:
                                s.add_values(*(list(next(iter(subsubset)))))
            # Add product nodes
            if self.multi_nodes:
                products = [PermProducts(*prod_inputs,
                            name="PermProducts%s" % self.__decomp_id)]
            else:
                products = self.__add_products(prod_inputs)
            # Connect products to each parent Sum
            for p in subset_info.parents:
                p.add_values(*products)
            # Increment decomposition id
            self.__decomp_id += 1
        return subsubset_infos

    def __add_products(self, prod_inputs):
        """
        Add product nodes for a single decomposition and connect them to their
        input nodes.

        Args:
            prod_inputs (list of list of Node): A list of lists of nodes
                being inputs to the products, grouped by scope.

        Returns:
            list of Product: A list of product nodes.
        """
        selected = [0 for _ in prod_inputs]  # Input selected for each scope
        cont = True
        products = []
        product_num = 1

        with tf.name_scope("Products%s" % self.__decomp_id):
            while cont:
                # Add a product node
                products.append(Product(*[pi[s] for (pi, s) in
                                          zip(prod_inputs, selected)],
                                        name="Product%s" % product_num))
                product_num += 1
                # Increment selected
                cont = False
                for i, group in enumerate(prod_inputs):
                    if selected[i] < len(group) - 1:
                        selected[i] += 1
                        for j in range(i):
                            selected[j] = 0
                        cont = True
                        break
        return products

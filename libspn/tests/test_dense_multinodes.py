#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import itertools
import os
import tensorflow as tf
import numpy as np
from context import libspn as spn

spn.config_logger(spn.DEBUG1)
logger = spn.get_logger()


def printc(string):
    COLOR = '\033[1m\033[93m'
    ENDC = '\033[0m'
    print(COLOR + string + ENDC)


class TestDenseSPNGeneratorMultiNodes(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def generic_dense_test(self, name, num_decomps, num_subsets, num_mixtures,
                           input_dist, num_input_mixtures, balanced, multi_nodes,
                           write_log, case):
        """A generic test for DenseSPNGeneratorMultiNodes."""
        self.tearDown()

        # Inputs
        v1 = spn.IVs(num_vars=3, num_vals=2, name="IVs1")
        v2 = spn.IVs(num_vars=3, num_vals=2, name="IVs2")

        gen = spn.DenseSPNGeneratorMultiNodes(num_decomps=num_decomps,
                                              num_subsets=num_subsets,
                                              num_mixtures=num_mixtures,
                                              input_dist=input_dist,
                                              balanced=balanced,
                                              num_input_mixtures=num_input_mixtures,
                                              multi_nodes=multi_nodes)

        logger.info("Generating SPN...")
        root = gen.generate(v1, v2)

        logger.info("Generating random weights...")
        with tf.name_scope("Weights"):
            spn.generate_weights(root, spn.ValueType.RANDOM_UNIFORM())

        logger.info("Generating weight initializers...")
        init = spn.initialize_weights(root)

        logger.info("Testing validity...")
        self.assertTrue(root.is_valid())

        logger.info("Generating value ops...")
        v = root.get_value()
        v_log = root.get_log_value()

        printc("Case: %s" % case)
        printc("- num_decomps: %s" % num_decomps)
        printc("- num_subsets: %s" % num_subsets)
        printc("- num_mixtures: %s" % num_mixtures)
        printc("- input_dist: %s" % ("MIXTURE" if input_dist ==
               spn.DenseSPNGeneratorMultiNodes.InputDist.MIXTURE else "RAW"))
        printc("- balanced: %s" % balanced)
        printc("- num_input_mixtures: %s" % num_input_mixtures)
        printc("- multi_nodes: %s" % multi_nodes)

        logger.info("Creating session...")
        with tf.Session() as sess:
            logger.info("Initializing weights...")
            init.run()
            logger.info("Computing all values...")
            feed = np.array(list(itertools.product(range(2), repeat=6)))
            feed_v1 = feed[:, :3]
            feed_v2 = feed[:, 3:]
            out = sess.run(v, feed_dict={v1: feed_v1, v2: feed_v2})
            out_log = sess.run(tf.exp(v_log), feed_dict={v1: feed_v1, v2: feed_v2})
            # Test if partition function is 1.0
            self.assertAlmostEqual(out.sum(), 1.0, places=6)
            self.assertAlmostEqual(out_log.sum(), 1.0, places=6)
            logger.info("Partition function: normal: %.10f, log: %.10f" %
                        (out.sum(), out_log.sum()))
            if write_log:
                logger.info("Writing log...")
                writer = tf.train.SummaryWriter(
                    os.path.realpath(os.path.join(
                        os.getcwd(), os.path.dirname(__file__),
                        "logs", "test_dense", name)),
                    sess.graph)
                writer.add_graph(sess.graph)
                writer.close()

    def test_generate_spn(self):
        """Generate and test dense SPNs with varying combination of parameters"""
        num_decomps = [1, 2]
        num_subsets = [2, 3, 6]
        num_mixtures = [1, 2]
        num_input_mixtures = [None, 1, 2]
        input_dist = [spn.DenseSPNGeneratorMultiNodes.InputDist.MIXTURE,
                      spn.DenseSPNGeneratorMultiNodes.InputDist.RAW]
        balanced = [True, False]
        multi_nodes = [True, False]
        name = ["mixture", "raw"]
        case = 0

        for n_dec in num_decomps:
            for n_sub in num_subsets:
                for n_mix in num_mixtures:
                    for n_imix in num_input_mixtures:
                        for dist, n in zip(input_dist, name):
                            for bal in balanced:
                                for m_nodes in multi_nodes:
                                    case += 1
                                    self.generic_dense_test(name=n,
                                                            num_decomps=n_dec,
                                                            num_subsets=n_sub,
                                                            num_mixtures=n_mix,
                                                            input_dist=dist,
                                                            num_input_mixtures=n_imix,
                                                            balanced=bal,
                                                            multi_nodes=m_nodes,
                                                            write_log=False,
                                                            case=case)


if __name__ == '__main__':
    unittest.main()

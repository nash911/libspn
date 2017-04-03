#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import tensorflow as tf
import numpy as np
from itertools import repeat
import time
from context import libspn as spn


class TestPerformancePermProducts(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_performance(self):
        """Testing performance of PermProducts op"""

        def test(inputs, feed, output, num_prods=1, num_graphs=1):
            with self.subTest(inputs=inputs, feed=feed, output=output):
                # Create multiple graphs, each with the structure:
                # Root <-- PermProducts
                roots = []
                for i in range(0, num_graphs):
                    roots = roots + [spn.Sum(spn.PermProducts(*inputs))]
                    roots[-1].generate_weights()

                # Create multiple ops, assigning each op to a graph
                ops = [r.get_value(spn.InferenceType.MARGINAL) for r in roots]

                # Create a single session and execute all the ops
                with tf.Session() as sess:
                    for r in roots:
                        spn.initialize_weights(r).run()

                    start_time = time.time()
                    out = sess.run(ops, feed_dict=feed)
                    total_time = time.time() - start_time

                print("\nPermProducts - %s " % ("Single-graph" if num_graphs == 1
                      else "Multi-graph (%d)" % (num_graphs)))
                print("No. of products per graph: ", num_prods)
                print("Total no. of products:     ", (num_graphs * num_prods))
                print("Total time taken:           %.5f s" % total_time)
                if num_graphs == 1:
                    tf_graph = tf.get_default_graph()
                    print("Total no. of TF ops per graph: ",
                          len(tf_graph.get_operations()))

                # Check all the outputs
                for o in out:
                    np.testing.assert_array_almost_equal(o, output, decimal=6)

        batch = 100
        features = 3
        num_inputs = 2
        num_prods = pow(features, num_inputs)
        num_graphs = 100

        # Create inputs
        inp = spn.ContVars(num_vars=features)
        inputs = list(repeat(inp, num_inputs))
        inputs_feed = np.ones((batch, features), dtype=spn.conf.dtype.as_numpy_dtype())

        test(inputs,
             {inputs[0]: inputs_feed},
             np.ones((batch, 1), dtype=spn.conf.dtype.as_numpy_dtype()),
             num_prods, num_graphs)


if __name__ == '__main__':
    unittest.main()

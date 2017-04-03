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
import time
from context import libspn as spn


class TestPerformanceSum(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_performance(self):
        """Testing performance of Sum op"""

        def test(inputs, ivs, feed, output, num_sums=1, num_graphs=1):
            with self.subTest(inputs=inputs, feed=feed, output=output):
                # Create multiple graphs, each with the structure:
                # Root <-- Sum
                roots = []
                for i in range(0, num_graphs):
                    s = []
                    for j in range(0, num_sums):
                        s = s + [spn.Sum(*inputs, ivs=ivs)]
                        s[-1].generate_weights()

                    roots = roots + [spn.Sum(*s)]
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

                print("\nSums - %s " % ("Single-graph" if num_graphs == 1
                      else "Multi-graph (%d)" % (num_graphs)))
                print("No. of sums per graph: ", num_sums)
                print("Total no. of sums:     ", (num_graphs * num_sums))
                print("Total time taken:       %.5f s" % total_time)
                if num_graphs == 1:
                    tf_graph = tf.get_default_graph()
                    print("Total no. of TF ops per graph: ",
                          len(tf_graph.get_operations()))

                # Check all the outputs
                if ivs:
                    output = output * (1.0/inputs[0]._compute_out_size())
                for o in out:
                    np.testing.assert_array_almost_equal(o, output, decimal=4)

        batch = 100
        features = 1000
        num_sums = 9
        num_graphs = 100

        # Create inputs
        inputs = spn.ContVars(num_vars=features)
        inputs_feed = np.ones((batch, features), dtype=spn.conf.dtype.as_numpy_dtype())

        # Create ivs
        ivs = spn.IVs(num_vars=1, num_vals=features)
        ivs_feed = np.zeros((batch, 1), dtype=np.int)

        # Create outputs
        outputs = np.ones((batch, 1), dtype=spn.conf.dtype.as_numpy_dtype())

        test([inputs],
             ivs,
             {inputs: inputs_feed,
              ivs: ivs_feed},
             outputs, num_sums, num_graphs)


if __name__ == '__main__':
    unittest.main()

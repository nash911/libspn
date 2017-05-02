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


class TestPerformanceSums(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_performance(self):
        """Testing performance of Sums op"""

        def test(inputs, ivs, feed, true_value_output, num_sums=1,
                 num_graphs=1, true_path_output=None):
            with self.subTest(inputs=inputs, feed=feed,
                              true_value_output=true_value_output,
                              true_path_output=true_path_output):
                # Create multiple graphs, each with the structure:
                # Root <-- Sums
                roots = []
                weights = []
                for i in range(0, num_graphs):
                    # Create a single Sums node, modeling 'num_sums' sums within,
                    # connecting it to inputs and ivs
                    s = spn.Sums(*inputs, num_sums=num_sums, ivs=ivs)
                    W = s.generate_weights()

                    # List of weights of the Sums node of each graph
                    weights = weights + [W]

                    # Connect the Sums nodes to the root node and generate weights
                    roots = roots + [spn.Sum(s)]
                    roots[-1].generate_weights()

                # Create multiple Values, assigning each value to the root node
                # of a graph
                values = [spn.Value(inference_type=spn.InferenceType.MARGINAL)
                          for i in range(0, num_graphs)]

                # Create multiple Value ops, assigning each op to a graph
                value_ops = [v.get_value(r) for v, r in zip(values, roots)]

                # Create a single session and execute all the Value ops
                with tf.Session() as sess:
                    for r in roots:
                        spn.initialize_weights(r).run()

                    start_time = time.time()
                    value_output = sess.run(value_ops, feed_dict=feed)
                    total_time = time.time() - start_time

                print("\nSums (%s) - %s " % (("With IVs" if ivs else "No IVs"),
                      ("Single-graph" if num_graphs == 1 else "Multi-graph (%d)"
                      % (num_graphs))))
                print("No. of sums per graph:       ", num_sums)
                print("Total no. of sums:           ", (num_graphs * num_sums))
                print("Up pass - Total time taken:   %.5f s" % total_time)

                if num_graphs == 1:
                    # Count number of TF ops in the graph for Up-pass
                    tf_graph = tf.get_default_graph()
                    up_graph_num_ops = len(tf_graph.get_operations())

                # Check all the Value outputs
                if ivs:
                    true_value_output = true_value_output * \
                     (num_sums/inputs[0]._compute_out_size())
                for vo in value_output:
                    np.testing.assert_array_almost_equal(vo, true_value_output,
                                                         decimal=4)

                # Test performance of Down-pass
                if true_path_output is not None:
                    # Create a MPEPath per graph
                    mpe_path_gen = [spn.MPEPath(value=v, log=False)
                                    for v in values]

                    # Generate mpe_path per graph, starting from root
                    for path_gen, r in zip(mpe_path_gen, roots):
                        path_gen.get_mpe_path(r)

                    # Create multiple Path ops, assigning each op to a graph
                    path_ops = [path_gen.counts[w] for path_gen, w in
                                zip(mpe_path_gen, weights)]

                    # Create a single session and execute all the Path ops
                    with tf.Session() as sess:
                        for r in roots:
                            spn.initialize_weights(r).run()

                        start_time = time.time()
                        path_output = sess.run(path_ops, feed_dict=feed)
                        total_time = time.time() - start_time

                    print("Down pass - Total time taken: %.5f s" % total_time)

                    if num_graphs == 1:
                        # Count number of TF ops in the graph for Down-pass
                        tf_graph = tf.get_default_graph()
                        down_graph_num_ops = len(tf_graph.get_operations()) -  \
                            up_graph_num_ops

                    # Check all the Path outputs
                    for po in path_output:
                        np.testing.assert_array_almost_equal(po, true_path_output,
                                                             decimal=4)

                # Print number of TF ops per graph
                if num_graphs == 1:
                    print("Total no. of TF ops per graph for Up pass:   ",
                          up_graph_num_ops)
                    if true_path_output is not None:
                        print("Total no. of TF ops per graph for Down pass: ",
                              down_graph_num_ops)

        batch = 100
        features = 1000
        num_sums = 4
        num_graphs = 10

        # Create inputs
        inputs = spn.ContVars(num_vars=features*num_sums)
        inputs_feed = np.ones((batch, features*num_sums),
                              dtype=spn.conf.dtype.as_numpy_dtype())

        # Create ivs
        ivs = spn.IVs(num_vars=num_sums, num_vals=features)
        ivs_feed = np.zeros((batch, num_sums), dtype=np.int)

        # Create outputs
        value_output = np.ones((batch, 1), dtype=spn.conf.dtype.as_numpy_dtype())
        path_output = np.zeros((num_sums, batch, features),
                               dtype=spn.conf.dtype.as_numpy_dtype())
        path_output[0, :, 0] = 1.0
        path_output = np.sum(path_output, axis=-2)

        test([inputs], None,
             {inputs: inputs_feed,
              ivs: ivs_feed},
             value_output, num_sums,
             num_graphs, path_output)

        batch = 100
        features = 1000
        num_sums = 100
        num_graphs = 10

        # Create inputs
        inputs = spn.ContVars(num_vars=features*num_sums)
        inputs_feed = np.ones((batch, features*num_sums),
                              dtype=spn.conf.dtype.as_numpy_dtype())

        # Create ivs
        ivs = spn.IVs(num_vars=num_sums, num_vals=features)
        ivs_feed = np.zeros((batch, num_sums), dtype=np.int)

        # Create outputs
        value_output = np.ones((batch, 1), dtype=spn.conf.dtype.as_numpy_dtype())
        path_output = np.zeros((num_sums, batch, features),
                               dtype=spn.conf.dtype.as_numpy_dtype())
        path_output[0, :, 0] = 1.0
        path_output = np.sum(path_output, axis=-2)

        test([inputs], None,
             {inputs: inputs_feed,
              ivs: ivs_feed},
             value_output, num_sums,
             num_graphs, path_output)


if __name__ == '__main__':
    unittest.main()

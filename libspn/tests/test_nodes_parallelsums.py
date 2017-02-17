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
from context import libspn as spn


class TestNodesParallelSums(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_marginal_value(self):
        """Calculating marginal value of Sum."""
        def test(values, num_sums, ivs, weights, feed, output):
            with self.subTest(values=values, num_sums=num_sums, ivs=ivs, weights=weights,
                              feed=feed):
                n = spn.ParallelSums(*values, num_sums=num_sums, ivs=ivs)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)

                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.ContVars(num_vars=2, name="ContVars1")
        v2 = spn.ContVars(num_vars=2, name="ContVars2")

        # MULTIPLE PARALLEL-SUM NODES
        # ---------------------------
        num_sums = 2

        # Multiple inputs, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3),
               (0.11*0.1 + 0.12*0.2 + 0.13*0.3 + 0.14*0.4)]])

        test([(v1, [1]), (v2, [0])],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.2*0.4 + 0.3*0.6), (0.2*0.2 + 0.3*0.8)],
              [(0.12*0.4 + 0.13*0.6), (0.12*0.2 + 0.13*0.8)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3),
               (0.11*0.1 + 0.12*0.2 + 0.13*0.3 + 0.14*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1, 2],
                    [1, -1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3), (0.3*0.3)],
              [(0.12*0.2), (0.11*0.1 + 0.12*0.2 + 0.13*0.3 + 0.14*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[3, 2],
                    [0, 1]]},
             [[(0.4*0.3), (0.3*0.3)],
              [(0.11*0.2), (0.12*0.2)]])

        test([(v1, [0, 1]), (v2, [1, 0])],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[3, 2],
                    [0, 1]]},
             [[(0.3*0.3), (0.4*0.3)],
              [(0.11*0.2), (0.12*0.2)]])

        # Single input with 1 value, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]]},
             [[(0.1*0.4 + 0.2*0.6), (0.1*0.2 + 0.2*0.8)],
              [(0.11*0.4 + 0.12*0.6), (0.11*0.2 + 0.12*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[-1, -1],
                   [-1, -1]]},
             [[(0.1*0.4 + 0.2*0.6), (0.1*0.2 + 0.2*0.8)],
              [(0.11*0.4 + 0.12*0.6), (0.11*0.2 + 0.12*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[1, -1],
                   [-1, 0]]},
             [[(0.2*0.6), (0.1*0.2 + 0.2*0.8)],
              [(0.11*0.4 + 0.12*0.6), (0.11*0.2)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[0, 1],
                   [1, 0]]},
             [[(0.1*0.4), (0.2*0.8)],
              [(0.12*0.6), (0.11*0.2)]])

        # Multiple inputs, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[-1, -1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[2, -1]]},
             [[(0.3*0.3), (0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[3, 2]]},
             [[(0.4*0.3), (0.3*0.3)]])

        # Single input with 1 value, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2]]},
             [[(0.1*0.4 + 0.2*0.6), (0.1*0.2 + 0.2*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[-1, -1]]},
             [[(0.1*0.4 + 0.2*0.6), (0.1*0.2 + 0.2*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[1, -1]]},
             [[(0.2*0.6), (0.1*0.2 + 0.2*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[0, 1]]},
             [[(0.1*0.4), (0.2*0.8)]])

        # SINGLE PARALLEL-SUM NODE
        # ------------------------
        num_sums = 1

        # Multiple inputs, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1],
                    [-1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1],
                    [3]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)],
              [(0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[3],
                    [0]]},
             [[(0.4*0.3)],
              [(0.11*0.2)]])

        # Single input with 1 value, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]]},
             [[0.1*0.4 + 0.2*0.6],
              [0.11*0.4 + 0.12*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[-1],
                   [-1]]},
             [[0.1*0.4 + 0.2*0.6],
              [0.11*0.4 + 0.12*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[1],
                   [-1]]},
             [[0.2*0.6],
              [0.11*0.4 + 0.12*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[0],
                   [1]]},
             [[0.1*0.4],
              [0.12*0.6]])

        # Multiple inputs, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[-1]]},
             [[0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[2]]},
             [[0.3*0.3]])

        # Single input with 1 value, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2]]},
             [[0.1*0.4 + 0.2*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[-1]]},
             [[0.1*0.2 + 0.2*0.8]])

        test([v1],
             num_sums,
             ivs,
             [0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[1]]},
             [[0.2*0.8]])

    def test_compute_mpe_value(self):
        """Calculating MPE value of Parallel Sums."""
        def test(values, num_sums, ivs, weights, feed, output):
            with self.subTest(values=values, num_sums=num_sums, ivs=ivs, weights=weights,
                              feed=feed):
                n = spn.Sums(*values, num_sums=num_sums, ivs=ivs)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MPE)
                op_log = n.get_log_value(spn.InferenceType.MPE)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)

                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.ContVars(num_vars=2, name="ContVars1")
        v2 = spn.ContVars(num_vars=2, name="ContVars2")

        # MULTIPLE PARALLEL-SUM NODES
        # ---------------------------
        num_sums = 2

        # Multiple inputs, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.4*0.3), (0.4*0.4)],
              [(0.14*0.3), (0.14*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.3, 0.3, 0.2, 0.2, 0.4, 0.3, 0.2, 0.1],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.4*0.2), (0.3*0.2)],
              [(0.12*0.3), (0.11*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1, 2],
                    [1, -1]]},
             [[(0.4*0.3), (0.3*0.3)],
              [(0.12*0.2), (0.14*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[3, 2],
                    [0, 1]]},
             [[(0.4*0.3), (0.3*0.3)],
              [(0.11*0.2), (0.12*0.2)]])

        # Single input with 1 value, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6, 0.8, 0.2],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]]},
             [[(0.2*0.6), (0.1*0.8)],
              [(0.12*0.6), (0.11*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.6, 0.4, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[-1, -1],
                   [-1, -1]]},
             [[(0.2*0.4), (0.2*0.8)],
              [(0.11*0.6), (0.12*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[1, -1],
                   [-1, 0]]},
             [[(0.2*0.6), (0.2*0.8)],
              [(0.12*0.6), (0.11*0.2)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[0, 1],
                   [1, 0]]},
             [[(0.1*0.4), (0.2*0.8)],
              [(0.12*0.6), (0.11*0.2)]])

        # Multiple inputs, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[(0.4*0.3), (0.4*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.3, 0.3, 0.2, 0.2, 0.4, 0.3, 0.2, 0.1],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[-1, -1]]},
             [[(0.4*0.2), (0.2*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[2, -1]]},
             [[(0.3*0.3), (0.4*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[3, 2]]},
             [[(0.4*0.3), (0.3*0.3)]])

        # Single input with 1 value, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.6, 0.4, 0.2, 0.8],
             {v1: [[0.1, 0.2]]},
             [[(0.2*0.4), (0.2*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.8, 0.2],
             {v1: [[0.1, 0.2]],
             ivs: [[-1, -1]]},
             [[(0.2*0.6), (0.1*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[0, -1]]},
             [[(0.1*0.4), (0.2*0.8)]])

        test([v1],
             num_sums,
             ivs,
             [0.6, 0.4, 0.8, 0.2],
             {v1: [[0.1, 0.2]],
             ivs: [[1, 0]]},
             [[(0.2*0.4), (0.1*0.8)]])

        # SINGLE PARALLEL-SUM NODE
        # ------------------------
        num_sums = 1

        # Multiple inputs, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.4*0.3)],
              [(0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.4, 0.3, 0.2, 0.1],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1],
                    [-1]]},
             [[(0.3*0.2)],
              [(0.11*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1],
                    [3]]},
             [[(0.4*0.3)],
              [(0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[3],
                    [0]]},
             [[(0.4*0.3)],
              [(0.11*0.2)]])

        # Single input with 1 value, multi-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]]},
             [[0.2*0.6],
              [0.12*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.6, 0.4],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[-1],
                   [-1]]},
             [[0.2*0.4],
              [0.11*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[0],
                   [-1]]},
             [[0.1*0.4],
              [0.12*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
             ivs: [[1],
                   [0]]},
             [[0.2*0.6],
              [0.11*0.4]])

        # Multiple inputs, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[0.4*0.3]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.4, 0.3, 0.2, 0.1],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[-1]]},
             [[0.2*0.3]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[2]]},
             [[0.3*0.3]])

        # Single input with 1 value, single-element batch
        ivs = spn.IVs(num_vars=num_sums, num_vals=2)

        test([v1],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2]]},
             [[0.2*0.6]])

        test([v1],
             num_sums,
             ivs,
             [0.6, 0.4],
             {v1: [[0.1, 0.2]],
             ivs: [[-1]]},
             [[0.2*0.4]])

        test([v1],
             num_sums,
             ivs,
             [0.2, 0.8],
             {v1: [[0.1, 0.2]],
             ivs: [[0]]},
             [[0.1*0.2]])


if __name__ == '__main__':
    unittest.main()

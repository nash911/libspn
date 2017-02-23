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


class TestNodesPermProduct(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_value(self):
        """Calculating value of Product"""

        def test(inputs, feed, output):
            with self.subTest(inputs=inputs, feed=feed):
                n = spn.PermProducts(*inputs)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                op_mpe = n.get_value(spn.InferenceType.MPE)
                op_log_mpe = n.get_log_value(spn.InferenceType.MPE)
                with tf.Session() as sess:
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)
                    out_mpe = sess.run(op_mpe, feed_dict=feed)
                    out_log_mpe = sess.run(tf.exp(op_log_mpe), feed_dict=feed)
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.ContVars(num_vars=3)
        v2 = spn.ContVars(num_vars=3)
        v3 = spn.ContVars(num_vars=3)

        # Multiple Product nodes
        # ----------------------

        # Case 1: No. of inputs > Input size
        # No. of inputs = 3
        # Input size = 2

        # Multi-element batch
        test([(v1, [0, 1]), (v2, [1, 2]), (v3, [0, 2])],
             {v1: [[0.1, 0.2, 0.3],       # 0.1  0.2
                   [0.4, 0.5, 0.6]],      # 0.4  0.5
              v2: [[0.7, 0.8, 0.9],       # 0.8  0.9
                   [0.11, 0.12, 0.13]],   # 0.12 0.13
              v3: [[0.14, 0.15, 0.16],    # 0.14 0.16
                   [0.17, 0.18, 0.19]]},  # 0.17 0.19
             [[(0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.16), (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.16),
               (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.16), (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.16)],
              [(0.4 * 0.12 * 0.17), (0.4 * 0.12 * 0.19), (0.4 * 0.13 * 0.17), (0.4 * 0.13 * 0.19),
               (0.5 * 0.12 * 0.17), (0.5 * 0.12 * 0.19), (0.5 * 0.13 * 0.17), (0.5 * 0.13 * 0.19)]])

        # Single-element batch
        test([(v1, [0, 1]), (v2, [1, 2]), (v3, [0, 2])],
             {v1: [[0.1, 0.2, 0.3]],      # 0.1  0.2
              v2: [[0.7, 0.8, 0.9]],      # 0.8  0.9
              v3: [[0.14, 0.15, 0.16]]},   # 0.14 0.16
             [[(0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.16), (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.16),
               (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.16), (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.16)]])

        # Case 2: No. of inputs < Input size
        # No. of inputs = 2
        # Input size = 3

        # Multi-element batch
        test([v1, v2],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7, 0.8, 0.9],
                   [0.11, 0.12, 0.13]]},
             [[(0.1 * 0.7), (0.1 * 0.8), (0.1 * 0.9),
               (0.2 * 0.7), (0.2 * 0.8), (0.2 * 0.9),
               (0.3 * 0.7), (0.3 * 0.8), (0.3 * 0.9)],
              [(0.4 * 0.11), (0.4 * 0.12), (0.4 * 0.13),
               (0.5 * 0.11), (0.5 * 0.12), (0.5 * 0.13),
               (0.6 * 0.11), (0.6 * 0.12), (0.6 * 0.13)]])

        # Single-element batch
        test([v1, v2],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7, 0.8, 0.9]]},
             [[(0.1 * 0.7), (0.1 * 0.8), (0.1 * 0.9),
               (0.2 * 0.7), (0.2 * 0.8), (0.2 * 0.9),
               (0.3 * 0.7), (0.3 * 0.8), (0.3 * 0.9)]])

        # Case 3: No. of inputs == Input size
        # No. of inputs = 3
        # Input size = 3

        # Multi-element batch
        test([v1, v2, v3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7, 0.8, 0.9],
                   [0.11, 0.12, 0.13]],
              v3: [[0.14, 0.15, 0.16],
                   [0.17, 0.18, 0.19]]},
             [[(0.1 * 0.7 * 0.14), (0.1 * 0.7 * 0.15), (0.1 * 0.7 * 0.16),
               (0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.15), (0.1 * 0.8 * 0.16),
               (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.15), (0.1 * 0.9 * 0.16),
               (0.2 * 0.7 * 0.14), (0.2 * 0.7 * 0.15), (0.2 * 0.7 * 0.16),
               (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.15), (0.2 * 0.8 * 0.16),
               (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.15), (0.2 * 0.9 * 0.16),
               (0.3 * 0.7 * 0.14), (0.3 * 0.7 * 0.15), (0.3 * 0.7 * 0.16),
               (0.3 * 0.8 * 0.14), (0.3 * 0.8 * 0.15), (0.3 * 0.8 * 0.16),
               (0.3 * 0.9 * 0.14), (0.3 * 0.9 * 0.15), (0.3 * 0.9 * 0.16)],
              [(0.4 * 0.11 * 0.17), (0.4 * 0.11 * 0.18), (0.4 * 0.11 * 0.19),
               (0.4 * 0.12 * 0.17), (0.4 * 0.12 * 0.18), (0.4 * 0.12 * 0.19),
               (0.4 * 0.13 * 0.17), (0.4 * 0.13 * 0.18), (0.4 * 0.13 * 0.19),
               (0.5 * 0.11 * 0.17), (0.5 * 0.11 * 0.18), (0.5 * 0.11 * 0.19),
               (0.5 * 0.12 * 0.17), (0.5 * 0.12 * 0.18), (0.5 * 0.12 * 0.19),
               (0.5 * 0.13 * 0.17), (0.5 * 0.13 * 0.18), (0.5 * 0.13 * 0.19),
               (0.6 * 0.11 * 0.17), (0.6 * 0.11 * 0.18), (0.6 * 0.11 * 0.19),
               (0.6 * 0.12 * 0.17), (0.6 * 0.12 * 0.18), (0.6 * 0.12 * 0.19),
               (0.6 * 0.13 * 0.17), (0.6 * 0.13 * 0.18), (0.6 * 0.13 * 0.19)]])

        # Single-element batch
        test([v1, v2, v3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7, 0.8, 0.9]],
              v3: [[0.14, 0.15, 0.16]]},
             [[(0.1 * 0.7 * 0.14), (0.1 * 0.7 * 0.15), (0.1 * 0.7 * 0.16),
               (0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.15), (0.1 * 0.8 * 0.16),
               (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.15), (0.1 * 0.9 * 0.16),
               (0.2 * 0.7 * 0.14), (0.2 * 0.7 * 0.15), (0.2 * 0.7 * 0.16),
               (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.15), (0.2 * 0.8 * 0.16),
               (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.15), (0.2 * 0.9 * 0.16),
               (0.3 * 0.7 * 0.14), (0.3 * 0.7 * 0.15), (0.3 * 0.7 * 0.16),
               (0.3 * 0.8 * 0.14), (0.3 * 0.8 * 0.15), (0.3 * 0.8 * 0.16),
               (0.3 * 0.9 * 0.14), (0.3 * 0.9 * 0.15), (0.3 * 0.9 * 0.16)]])

        # Single Product node
        # -------------------

        # Case 4: No. of inputs > Input size
        # No. of inputs = 3
        # Input size = 1

        # Multi-element batch
        test([(v1, [1]), (v2, [2]), (v3, [0])],
             {v1: [[0.1, 0.2, 0.3],       # 0.2
                   [0.4, 0.5, 0.6]],      # 0.5
              v2: [[0.7, 0.8, 0.9],       # 0.9
                   [0.11, 0.12, 0.13]],   # 0.13
              v3: [[0.14, 0.15, 0.16],    # 0.14
                   [0.17, 0.18, 0.19]]},  # 0.17
             [[(0.2 * 0.9 * 0.14)],
              [(0.5 * 0.13 * 0.17)]])

        # Single-element batch
        test([(v1, [1]), (v2, [2]), (v3, [0])],
             {v1: [[0.1, 0.2, 0.3]],      # 0.2
              v2: [[0.7, 0.8, 0.9]],      # 0.9
              v3: [[0.14, 0.15, 0.16]]},  # 0.14
             [[(0.2 * 0.9 * 0.14)]])

        # Case 5: No. of inputs < Input size
        # No. of inputs = 1
        # Input size = 3

        # Multi-element batch
        test([v1],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]]},
             [[(0.1 * 0.2 * 0.3)],
              [(0.4 * 0.5 * 0.6)]])

        # Single-element batch
        test([v1],
             {v1: [[0.1, 0.2, 0.3]]},
             [[(0.1 * 0.2 * 0.3)]])

        # Case 6: No. of inputs == Input size
        # No. of inputs = 1
        # Input size = 1

        # Multi-element batch
        test([(v2, [1])],
             {v2: [[0.7, 0.8, 0.9],       # 0.8
                   [0.11, 0.12, 0.13]]},  # 0.12
             [[0.8],
              [0.12]])

        # Single-element batch
        test([(v2, [2])],
             {v2: [[0.7, 0.8, 0.9]]},       # 0.9
             [[0.9]])


if __name__ == '__main__':
    unittest.main()

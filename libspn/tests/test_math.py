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


class TestMath(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_gather_cols_errors(self):
        # Should work
        spn.utils.gather_cols(tf.constant([1, 2, 3]),
                              [0, 1, 2])
        spn.utils.gather_cols(tf.constant([[1, 2, 3]]),
                              [0, 1, 2])
        spn.utils.gather_cols(tf.placeholder(tf.float32,
                                             shape=(None, 3)),
                              [0, 1, 2])

        # Param dim number
        with self.assertRaises(RuntimeError):
            spn.utils.gather_cols(tf.placeholder(tf.float32,
                                                 shape=(None, None)),
                                  [0, 1, 2])
        # Param size defined
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([[[1, 2, 3]]]),
                                  [0, 1, 2])
        # Index dims
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([1, 2, 3]),
                                  [[0, -1, 2]])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([1, 2, 3]),
                                  1)
        # Indices empty
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([1, 2, 3]),
                                  [])
        # Index values
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([1, 2, 3]),
                                  [0, -1, 2])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([1, 2, 3]),
                                  [0, 3, 2])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([1, 2, 3]),
                                  [0.1, 3, 2])

    def test_gather_cols(self):
        def test(params, indices, dtype, true_output):
            with self.subTest(params=params, indices=indices,
                              dtype=dtype):
                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)
                p2d2 = tf.constant(np.array([np.array(params),
                                             np.array(params) * 2,
                                             np.array(params) * 3]), dtype=dtype)
                op1df = spn.utils.gather_cols(p1d, indices, use_gather_nd=False)
                op1dt = spn.utils.gather_cols(p1d, indices, use_gather_nd=True)
                op2d1f = spn.utils.gather_cols(p2d1, indices, use_gather_nd=False)
                op2d1t = spn.utils.gather_cols(p2d1, indices, use_gather_nd=True)
                op2d2f = spn.utils.gather_cols(p2d2, indices, use_gather_nd=False)
                op2d2t = spn.utils.gather_cols(p2d2, indices, use_gather_nd=True)
                with tf.Session() as sess:
                    out1df = sess.run(op1df)
                    out1dt = sess.run(op1dt)
                    out2d1f = sess.run(op2d1f)
                    out2d1t = sess.run(op2d1t)
                    out2d2f = sess.run(op2d2f)
                    out2d2t = sess.run(op2d2t)
                np.testing.assert_array_almost_equal(out1df, true_output)
                np.testing.assert_array_almost_equal(out1dt, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1df.dtype)
                self.assertEqual(dtype.as_numpy_dtype, out1dt.dtype)
                true_output_2d1 = [np.array(true_output)]
                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * 2,
                                   np.array(true_output) * 3]
                np.testing.assert_array_almost_equal(out2d1f, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d1t, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d2f, true_output_2d2)
                np.testing.assert_array_almost_equal(out2d2t, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d1f.dtype)
                self.assertEqual(dtype.as_numpy_dtype, out2d1t.dtype)
                self.assertEqual(dtype.as_numpy_dtype, out2d2f.dtype)
                self.assertEqual(dtype.as_numpy_dtype, out2d2t.dtype)

        # Single column input tensor
        test([1],
             [0],
             tf.float32,
             [1.0])
        test([1],
             [0],
             tf.float64,
             [1.0])

        # Single index
        test([1, 2, 3],
             [1],
             tf.float32,
             [2.0])
        test([1, 2, 3],
             [1],
             tf.float64,
             [2.0])

        # Multiple indices
        test([1, 2, 3],
             [2, 1, 0],
             tf.float32,
             [3.0, 2.0, 1.0])
        test([1, 2, 3],
             [2, 1, 0],
             tf.float64,
             [3.0, 2.0, 1.0])
        test([1, 2, 3],
             [0, 2],
             tf.float32,
             [1.0, 3.0])
        test([1, 2, 3],
             [0, 2],
             tf.float64,
             [1.0, 3.0])

        # Gathering single column tensor should return that tensor directly
        t = tf.constant([1])
        out = spn.utils.gather_cols(t, [0])
        self.assertIs(out, t)
        t = tf.constant([[1], [2]])
        out = spn.utils.gather_cols(t, [0])
        self.assertIs(out, t)

    def test_gather_columns(self):
        def test(params, indices, dtype, true_output):
            with self.subTest(params=params, indices=indices,
                              dtype=dtype):
                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)
                p2d2 = tf.constant(np.array([np.array(params),
                                             np.array(params) * 2,
                                             np.array(params) * 3]), dtype=dtype)

                op1d = spn.utils.gather_columns_module.gather_columns(p1d, indices)
                op2d1 = spn.utils.gather_columns_module.gather_columns(p2d1, indices)
                op2d2 = spn.utils.gather_columns_module.gather_columns(p2d2, indices)

                with tf.Session() as sess:
                    out1d = sess.run(op1d)
                    out2d1 = sess.run(op2d1)
                    out2d2 = sess.run(op2d2)

                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)

                true_output_2d1 = [np.array(true_output)]
                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * 2,
                                   np.array(true_output) * 3]

                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)
                self.assertEqual(dtype.as_numpy_dtype, out2d2.dtype)

        # Single column input tensor
        test([1],
             [0],
             tf.float32,
             [1.0])
        test([1],
             [0],
             tf.float64,
             [1.0])

        # Single index
        test([1, 2, 3],
             [1],
             tf.float32,
             [2.0])
        test([1, 2, 3],
             [1],
             tf.float64,
             [2.0])

        # Multiple indices
        test([1, 2, 3],
             [2, 1, 0],
             tf.float32,
             [3.0, 2.0, 1.0])
        test([1, 2, 3],
             [2, 1, 0],
             tf.float64,
             [3.0, 2.0, 1.0])
        test([1, 2, 3],
             [0, 2],
             tf.float32,
             [1.0, 3.0])
        test([1, 2, 3],
             [0, 2],
             tf.float64,
             [1.0, 3.0])

        # Gathering single column tensor should return that tensor directly
        t = tf.constant([1])
        out = spn.utils.gather_cols(t, [0])
        self.assertIs(out, t)
        t = tf.constant([[1], [2]])
        out = spn.utils.gather_cols(t, [0])
        self.assertIs(out, t)

    def test_scatter_cols_errors(self):
        # TODO
        pass

    def test_scatter_cols(self):
        def test(params, indices, num_cols, dtype, true_output):
            with self.subTest(params=params, indices=indices,
                              num_cols=num_cols, dtype=dtype):
                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)
                p2d2 = tf.constant(np.array([np.array(params),
                                             np.array(params) * 2,
                                             np.array(params) * 3]), dtype=dtype)
                op1d = spn.utils.scatter_cols(p1d, indices, num_cols)
                op2d1 = spn.utils.scatter_cols(p2d1, indices, num_cols)
                op2d2 = spn.utils.scatter_cols(p2d2, indices, num_cols)
                with tf.Session() as sess:
                    out1d = sess.run(op1d)
                    out2d1 = sess.run(op2d1)
                    out2d2 = sess.run(op2d2)
                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)
                true_output_2d1 = [np.array(true_output)]
                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * 2,
                                   np.array(true_output) * 3]
                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)
                self.assertEqual(dtype.as_numpy_dtype, out2d2.dtype)

        # Single column output
        test([10],
             [0],
             1,
             tf.float32,
             [10.0])
        test([10],
             [0],
             1,
             tf.float64,
             [10.0])

        # Multi-column output, single-column input
        test([10],
             [1],
             4,
             tf.float32,
             [0.0, 10.0, 0.0, 0.0])
        test([10],
             [0],
             4,
             tf.float64,
             [10.0, 0.0, 0.0, 0.0])

        # Multi-column output, multi-column input
        test([10, 20, 30],
             [1, 3, 0],
             4,
             tf.float32,
             [30.0, 10.0, 0.0, 20.0])
        test([10, 20, 30],
             [1, 3, 0],
             4,
             tf.float64,
             [30.0, 10.0, 0.0, 20.0])

    def test_broadcast_value(self):
        """broadcast_value for various value types"""

        v1 = spn.utils.broadcast_value(spn.ValueType.RANDOM_UNIFORM(0, 1),
                                       (2, 3), dtype=tf.float64)

        v2 = spn.utils.broadcast_value(1,
                                       (2, 3), dtype=tf.float64)

        v3 = spn.utils.broadcast_value(1.0,
                                       (2, 3), dtype=tf.float64)

        v4 = spn.utils.broadcast_value([1],
                                       (2, 3), dtype=tf.float64)

        with tf.Session() as sess:
            out1 = sess.run(v1)
            out2 = sess.run(v2)
            out3 = sess.run(v3)
            out4 = sess.run(v4)
            self.assertEqual(out1.dtype, np.float64)
            self.assertEqual(out2.dtype, np.float64)
            self.assertEqual(out3.dtype, np.float64)
            self.assertEqual(out4.dtype, np.float64)
            self.assertGreaterEqual(out1.min(), 0)
            self.assertLessEqual(out1.max(), 1)
            np.testing.assert_array_almost_equal(out2, [[1.0, 1.0, 1.0],
                                                        [1.0, 1.0, 1.0]])
            np.testing.assert_array_almost_equal(out3, [[1.0, 1.0, 1.0],
                                                        [1.0, 1.0, 1.0]])
            np.testing.assert_array_almost_equal(out4, [1.0])

    def test_normalize_tensor(self):
        """normalize_tensor"""

        v1 = spn.utils.normalize_tensor([1])
        v2 = spn.utils.normalize_tensor(tf.constant([1], dtype=tf.float32))
        v3 = spn.utils.normalize_tensor(tf.constant([1], dtype=tf.float64))
        v4 = spn.utils.normalize_tensor([1.0])
        v5 = spn.utils.normalize_tensor([0.1])
        v6 = spn.utils.normalize_tensor([1, 0.5, 1])
        v7 = spn.utils.normalize_tensor([[1, 1],
                                         [1, 2]])
        v8 = spn.utils.normalize_tensor([0.25, 0.25, 0.25, 0.25])
        v9 = spn.utils.normalize_tensor([[0.25, 0.25],
                                         [0.25, 0.25]])

        with tf.Session() as sess:
            out1 = sess.run(v1)
            out2 = sess.run(v2)
            out3 = sess.run(v3)
            out4 = sess.run(v4)
            out5 = sess.run(v5)
            out6 = sess.run(v6)
            out7 = sess.run(v7)
            out8 = sess.run(v8)
            out9 = sess.run(v9)
            self.assertEqual(out1.dtype, np.float64)
            self.assertEqual(out2.dtype, np.float32)
            self.assertEqual(out3.dtype, np.float64)
            self.assertEqual(out4.dtype, np.float32)
            self.assertEqual(out5.dtype, np.float32)
            self.assertEqual(out6.dtype, np.float32)
            self.assertEqual(out7.dtype, np.float64)
            self.assertEqual(out8.dtype, np.float32)
            self.assertEqual(out9.dtype, np.float32)

            np.testing.assert_array_almost_equal(out1, [1.0])
            np.testing.assert_array_almost_equal(out2, [1.0])
            np.testing.assert_array_almost_equal(out3, [1.0])
            np.testing.assert_array_almost_equal(out4, [1.0])
            np.testing.assert_array_almost_equal(out5, [1.0])
            np.testing.assert_array_almost_equal(out6, [0.4, 0.2, 0.4])
            np.testing.assert_array_almost_equal(out7, [[0.2, 0.2],
                                                        [0.2, 0.4]])
            np.testing.assert_array_almost_equal(out8, [0.25, 0.25, 0.25, 0.25])
            np.testing.assert_array_almost_equal(out9, [[0.25, 0.25],
                                                        [0.25, 0.25]])

    def test_reduce_log_sum(self):

        def test(dtype):
            with self.subTest(dtype=dtype):
                inpt = [[0.0, 0.0, 0.0],
                        [0.0, 0.1, 0.0],
                        [0.1, 0.2, 0.3],
                        [1.0, 0.0, 2.0],
                        [0.0000001, 0.0000002, 0.0000003],
                        [1e-10, 2e-10, 3e-10]]
                inpt_array = np.array(inpt, dtype=dtype.as_numpy_dtype())
                inpt_tensor = tf.constant(inpt, dtype=dtype)
                log_inpt_tensor = tf.log(inpt_tensor)
                op_log = spn.utils.reduce_log_sum(log_inpt_tensor)
                op = tf.exp(op_log)

                with tf.Session() as sess:
                    out = sess.run(op)

                np.testing.assert_array_almost_equal(out,
                                                     np.sum(inpt_array, axis=1,
                                                            keepdims=True))
                self.assertEqual(out.dtype, dtype.as_numpy_dtype())

        test(tf.float32)
        test(tf.float64)

    def test_split(self):
        value1 = tf.constant(np.r_[:7])
        value2 = tf.constant(np.r_[:21].reshape(-1, 7))
        op1 = spn.utils.split(split_dim=0, split_sizes=(1, 3, 1, 2), value=value1)
        op2 = spn.utils.split(split_dim=1, split_sizes=(1, 3, 1, 2), value=value2)
        op3 = spn.utils.split(split_dim=0, split_sizes=(7), value=value1)
        op4 = spn.utils.split(split_dim=1, split_sizes=(7), value=value2)
        with tf.Session() as sess:
            out1 = sess.run(op1)
            out2 = sess.run(op2)
            out3 = sess.run(op3)
            out4 = sess.run(op4)

        np.testing.assert_array_equal(out1[0], np.array([0]))
        np.testing.assert_array_equal(out1[1], np.array([1, 2, 3]))
        np.testing.assert_array_equal(out1[2], np.array([4]))
        np.testing.assert_array_equal(out1[3], np.array([5, 6]))

        np.testing.assert_array_equal(out2[0], np.array([[0],
                                                         [7],
                                                         [14]]))
        np.testing.assert_array_equal(out2[1], np.array([[1, 2, 3],
                                                         [8, 9, 10],
                                                         [15, 16, 17]]))
        np.testing.assert_array_equal(out2[2], np.array([[4],
                                                         [11],
                                                         [18]]))
        np.testing.assert_array_equal(out2[3], np.array([[5, 6],
                                                         [12, 13],
                                                         [19, 20]]))

        np.testing.assert_array_equal(out3[0], np.array([0, 1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(out4[0], np.array([[0, 1, 2, 3, 4, 5, 6],
                                                         [7, 8, 9, 10, 11, 12, 13],
                                                         [14, 15, 16, 17, 18, 19, 20]]))

if __name__ == '__main__':
    unittest.main()

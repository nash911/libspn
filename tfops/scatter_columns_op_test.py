#!/usr/bin/env python3

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn

from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes


class TestScatterColumns(tf.test.TestCase):
    num_cols = 1000
    num_rows = 25000

    def tearDown(self):
        tf.reset_default_graph()

    def testEmptyIndices(self):
        with self.test_session(use_gpu=False) as sess:
            params = [0, 1, 2]
            indices = tf.constant([], dtype=tf.int32)
            num_out_cols = 10
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError("Indices cannot be empty."):
                sess.run(scatter)

    def testScalarParams(self):
        params = 10
        indices = [1, 2, 3]
        num_out_cols = 10
        pad_elem = 0
        with self.assertRaises(ValueError):
            spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)

    def testScalarIndices(self):
        params = [1, 2, 3]
        indices = 1
        num_out_cols = 10
        pad_elem = 0
        with self.assertRaises(ValueError):
            spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)

    def test3DParams(self):
        params = [[[0, 1, 2]]]
        indices = [1, 2, 3]
        num_out_cols = 10
        pad_elem = 0
        with self.assertRaises(ValueError):
            spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)

    def test2DIndices(self):
        params = [[0, 1, 2]]
        indices = [[1, 2, 3]]
        num_out_cols = 10
        pad_elem = 0
        with self.assertRaises(ValueError):
            spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)

    def test2DPadElem(self):
        params = [[0, 1, 2]]
        indices = [1, 2, 3]
        num_out_cols = 5
        pad_elem = [[0]]
        with self.assertRaises(ValueError):
            spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)

    def testNegativeIndices_CPU(self):
        with self.test_session(use_gpu=False) as sess:
            params = tf.constant([[1, 2, 3]], dtype=tf.float64)
            indices = [1, -1, 0]
            num_out_cols = 6
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError("Indices\(1\): -1 is not in range "
                                          "\(0, 6\]."):
                sess.run(scatter)

    def testNegativeIndices_GPU(self):
        with self.test_session(use_gpu=True) as sess:
            params = tf.constant([[1, 2, 3]], dtype=tf.float64)
            indices = [1, -1, 0]
            num_out_cols = 6
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError("Indices\(1\): -1 is not in range "
                                          "\(0, 6\]."):
                sess.run(scatter)

    def testBadIndices_CPU(self):
        with self.test_session(use_gpu=False) as sess:
            params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
            indices = tf.constant([2, 1, 10, 6, 5], dtype=tf.int32)
            num_out_cols = 7
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError("Indices\(2\): 10 is not in range "
                                          "\(0, 7\]."):
                sess.run(scatter)

    def testBadIndices_GPU(self):
        with self.test_session(use_gpu=True) as sess:
            params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
            indices = tf.constant([2, 1, 10, 6, 5], dtype=tf.int32)
            num_out_cols = 7
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError("Indices\(2\): 10 is not in range "
                                          "\(0, 7\]."):
                sess.run(scatter)

    def testDuplicateIndices_CPU(self):
        with self.test_session(use_gpu=False) as sess:
            params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
            indices = tf.constant([0, 1, 2, 2, 4], dtype=tf.int32)
            num_out_cols = 5
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError(
                    "Indices cannot contain duplicates. Total no. of indices: "
                    "5 != no. of unique indices: 4."):
                sess.run(scatter)

    def testDuplicateIndices_GPU(self):
        with self.test_session(use_gpu=True) as sess:
            params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
            indices = tf.constant([0, 1, 2, 2, 4], dtype=tf.int32)
            num_out_cols = 5
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError(
                    "Indices cannot contain duplicates. Total no. of indices: "
                    "5 != no. of unique indices: 4."):
                sess.run(scatter)

    def testWrongOutNumCols(self):
        with self.test_session(use_gpu=False) as sess:
            params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
            indices = tf.constant([4, 3, 2, 1, 0], dtype=tf.int32)
            num_out_cols = 4
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError("num_out_cols: 4 must be >= size of "
                                          "the indexed dimension of params: 5"):
                sess.run(scatter)

    def testIncorrectIndicesSize(self):
        with self.test_session(use_gpu=False) as sess:
            params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
            indices = tf.constant([11, 10, 9, 8], dtype=tf.int32)
            num_out_cols = 12
            pad_elem = 0
            scatter = spn.ops.scatter_cols(
                params, indices, pad_elem, num_out_cols)
            with self.assertRaisesOpError(
                    "Size of indices: 4 and the indexed dimension "
                    "of params - 5 - must be the same."):
                sess.run(scatter)

    def test_scatter_cols(self):
        ops.RegisterShape("ScatterColumns")(common_shapes.call_cpp_shape_fn)

        def test(params, indices, num_out_cols, pad_elem, dtype,
                 true_output, use_gpu=False, large_case=False):
            with self.test_session(use_gpu=use_gpu) as sess:
                if dtype == bool:
                    row1 = row2 = row3 = True
                else:
                    row1 = 1
                    row2 = 0
                    row3 = -1

                npdtype = dtype.as_numpy_dtype()

                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)

                if not large_case:
                    p2d2 = tf.constant(np.array([np.array(params) * row1,
                                                 np.array(params) * row2,
                                                 np.array(params) * row3]),
                                       dtype=dtype)
                else:
                    # For testing the large case example, create a matrix of size
                    # (num_rows, num_cols), rather than of size (3, num_cols) as
                    # per the non-large case example.
                    params_matrix = np.empty([self.num_rows, self.num_cols],
                                             dtype=npdtype)
                    params_row = np.array(params, dtype=npdtype)
                    for i in range(0, self.num_rows):
                        params_matrix[i, :] = params_row * (i + 1)
                    p2d2 = tf.constant(params_matrix, dtype=dtype)

                ind_32 = tf.constant(indices, dtype=tf.int32)
                ind_64 = tf.constant(indices, dtype=tf.int64)

                op1d = spn.ops.scatter_cols(
                    p1d, ind_64, pad_elem, num_out_cols)
                op2d1 = spn.ops.scatter_cols(
                    p2d1, ind_32, pad_elem, num_out_cols)
                op2d2 = spn.ops.scatter_cols(
                    p2d2, ind_64, pad_elem, num_out_cols)

                out1d = sess.run(op1d)
                out2d1 = sess.run(op2d1)
                out2d2 = sess.run(op2d2)

                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)
                np.testing.assert_array_equal(op1d.get_shape(),
                                              np.array([num_out_cols]))

                true_output_2d1 = [np.array(true_output)]
                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)
                np.testing.assert_array_equal(op2d1.get_shape(),
                                              np.array([1, num_out_cols]))

                if not large_case:
                    r_1 = np.array(true_output)
                    r_2 = np.array(true_output)
                    r_3 = np.array(true_output)
                    ind = np.array(indices)

                    r_1[ind] = r_1[ind] * row1
                    r_2[ind] = r_2[ind] * row2
                    r_3[ind] = r_3[ind] * row3

                    true_output_2d2 = [r_1,
                                       r_2,
                                       r_3]

                    true_shape = np.array([3, num_out_cols])
                else:
                    # For large test case, again create a large output matrix,
                    # based on the true output parameter, to compare the op
                    # output against.
                    params_matrix = np.empty([self.num_rows, self.num_cols * 2],
                                             dtype=npdtype)
                    true_output_row = np.array(true_output, dtype=npdtype)
                    ind = np.array(indices)
                    for i in range(0, self.num_rows):
                        params_matrix[i, :] = true_output_row
                        params_matrix[i, ind] = true_output_row[ind] * (i + 1)
                    true_output_2d2 = params_matrix

                    true_shape = np.array([self.num_rows, num_out_cols])

                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d2.dtype)
                np.testing.assert_array_equal(op2d2.get_shape(), true_shape)

        float_val = 1.23456789
        int_32_upper = 2147483647
        int_64_upper = 9223372036854775807

        pad_elem = 333

        # Single column output
        # float
        # CPU
        test([float_val],
             [0],
             1,
             pad_elem,
             tf.float32,
             [float_val],
             use_gpu=False)
        test([float_val],
             [0],
             1,
             pad_elem,
             tf.float64,
             [float_val],
             use_gpu=False)

        # GPU
        test([float_val],
             [0],
             1,
             pad_elem,
             tf.float32,
             [float_val],
             use_gpu=True)
        test([float_val],
             [0],
             1,
             pad_elem,
             tf.float64,
             [float_val],
             use_gpu=True)

        # int
        # CPU
        test([int_32_upper],
             [0],
             1,
             pad_elem,
             tf.int32,
             [int_32_upper],
             use_gpu=False)
        test([int_64_upper],
             [0],
             1,
             pad_elem,
             tf.int64,
             [int_64_upper],
             use_gpu=False)

        # GPU
        test([int_32_upper],
             [0],
             1,
             pad_elem,
             tf.int32,
             [int_32_upper],
             use_gpu=True)
        test([int_64_upper],
             [0],
             1,
             pad_elem,
             tf.int64,
             [int_64_upper],
             use_gpu=True)

        # bool
        # CPU
        test([True],
             [0],
             1,
             False,
             tf.bool,
             [True],
             use_gpu=False)

        # GPU
        test([True],
             [0],
             1,
             False,
             tf.bool,
             [True],
             use_gpu=True)

        # Multi-column output, single-column input
        # CPU
        test([float_val],
             [1],
             4,
             pad_elem,
             tf.float32,
             [pad_elem, float_val, pad_elem, pad_elem],
             use_gpu=False)
        test([float_val],
             [0],
             4,
             pad_elem,
             tf.float64,
             [float_val, pad_elem, pad_elem, pad_elem],
             use_gpu=False)

        # GPU
        test([float_val],
             [1],
             4,
             pad_elem,
             tf.float32,
             [pad_elem, float_val, pad_elem, pad_elem],
             use_gpu=True)
        test([float_val],
             [0],
             4,
             pad_elem,
             tf.float64,
             [float_val, pad_elem, pad_elem, pad_elem],
             use_gpu=True)

        # int
        # CPU
        test([int_32_upper],
             [2],
             5,
             pad_elem,
             tf.int32,
             [pad_elem, pad_elem, int_32_upper, pad_elem, pad_elem],
             use_gpu=False)
        test([int_64_upper],
             [4],
             5,
             pad_elem,
             tf.int64,
             [pad_elem, pad_elem, pad_elem, pad_elem, int_64_upper],
             use_gpu=False)

        # GPU
        test([int_32_upper],
             [2],
             5,
             pad_elem,
             tf.int32,
             [pad_elem, pad_elem, int_32_upper, pad_elem, pad_elem],
             use_gpu=True)
        test([int_64_upper],
             [4],
             5,
             pad_elem,
             tf.int64,
             [pad_elem, pad_elem, pad_elem, pad_elem, int_64_upper],
             use_gpu=True)

        # bool
        # CPU
        test([True],
             [3],
             5,
             False,
             tf.bool,
             [False, False, False, True, False],
             use_gpu=False)

        # GPU
        test([True],
             [3],
             5,
             False,
             tf.bool,
             [False, False, False, True, False],
             use_gpu=True)

        # Multi-column output, multi-column input
        # float
        # No consecutive padded columns
        # CPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6],
             [5, 3, 9, 1, 8, 6],
             10,
             pad_elem,
             tf.float32,
             [pad_elem, float_val * 4, pad_elem, float_val * 2, pad_elem,
              float_val, float_val * 6, pad_elem, float_val * 5, float_val * 3],
             use_gpu=False)

        # GPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6],
             [5, 3, 9, 1, 8, 6],
             10,
             pad_elem,
             tf.float32,
             [pad_elem, float_val * 4, pad_elem, float_val * 2, pad_elem,
              float_val, float_val * 6, pad_elem, float_val * 5, float_val * 3],
             use_gpu=True)

        # Consecutive padded columns in the end
        # CPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6],
             [5, 3, 9, 1, 8, 6],
             15,
             pad_elem,
             tf.float64,
             [pad_elem, float_val * 4, pad_elem, float_val * 2, pad_elem,
              float_val, float_val * 6, pad_elem, float_val * 5, float_val * 3,
              pad_elem, pad_elem, pad_elem, pad_elem, pad_elem],
             use_gpu=False)

        # GPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6],
             [5, 3, 9, 1, 8, 6],
             15,
             pad_elem,
             tf.float64,
             [pad_elem, float_val * 4, pad_elem, float_val * 2, pad_elem,
              float_val, float_val * 6, pad_elem, float_val * 5, float_val * 3,
              pad_elem, pad_elem, pad_elem, pad_elem, pad_elem],
             use_gpu=True)

        # int
        # Consecutive padded columns in the beginning
        # CPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6, float_val * 7, float_val * 8,
              float_val * 9],
             [7, 14, 5, 9, 10, 11, 6, 3, 8],
             15,
             pad_elem,
             tf.float32,
             [pad_elem, pad_elem, pad_elem, float_val * 8, pad_elem,
              float_val * 3, float_val * 7, float_val, float_val * 9,
              float_val * 4, float_val * 5, float_val * 6, pad_elem,
              pad_elem, float_val * 2],
             use_gpu=False)

        # GPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6, float_val * 7, float_val * 8,
              float_val * 9],
             [7, 14, 5, 9, 10, 11, 6, 3, 8],
             15,
             pad_elem,
             tf.float32,
             [pad_elem, pad_elem, pad_elem, float_val * 8, pad_elem,
              float_val * 3, float_val * 7, float_val, float_val * 9,
              float_val * 4, float_val * 5, float_val * 6, pad_elem, pad_elem,
              float_val * 2],
             use_gpu=True)

        # Consecutive padded columns in the middle
        # CPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6, float_val * 7, float_val * 8,
              float_val * 9],
             [13, 8, 4, 1, 2, 3, 11, 0, 9],
             15,
             pad_elem,
             tf.float64,
             [float_val * 8, float_val * 4, float_val * 5, float_val * 6,
              float_val * 3, pad_elem, pad_elem, pad_elem, float_val * 2,
              float_val * 9, pad_elem, float_val * 7, pad_elem, float_val,
              pad_elem],
             use_gpu=False)

        # GPU
        test([float_val, float_val * 2, float_val * 3, float_val * 4,
              float_val * 5, float_val * 6, float_val * 7, float_val * 8,
              float_val * 9],
             [13, 8, 4, 1, 2, 3, 11, 0, 9],
             15,
             pad_elem,
             tf.float64,
             [float_val * 8, float_val * 4, float_val * 5, float_val * 6,
              float_val * 3, pad_elem, pad_elem, pad_elem, float_val * 2,
              float_val * 9, pad_elem, float_val * 7, pad_elem, float_val,
              pad_elem],
             use_gpu=True)

        # bool
        # No padded columns
        # CPU
        test([True, False, False, True],
             [2, 1, 3, 0],
             4,
             False,
             tf.bool,
             [True, False, True, False],
             use_gpu=False)

        # GPU
        test([True, False, False, True],
             [2, 1, 3, 0],
             4,
             False,
             tf.bool,
             [True, False, True, False],
             use_gpu=True)

        # Consecutive padded columns in the beginning, middle and end
        # CPU
        test([True, False, False, True],
             [5, 11, 3, 9],
             15,
             False,
             tf.bool,
             [False, False, False, False, False, True, False, False,
              False, True, False, False, False, False, False],
             use_gpu=False)

        # GPU
        test([True, False, False, True],
             [5, 11, 3, 9],
             15,
             False,
             tf.bool,
             [False, False, False, False, False, True, False, False,
              False, True, False, False, False, False, False],
             use_gpu=True)


if __name__ == '__main__':
    unittest.main()

 #!/usr/bin/env python3

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn
import collections
import re

from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes


class TestScatterValues(tf.test.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    # def testEmptyIndices(self):
    #     with self.test_session(use_gpu=False) as sess:
    #         params = [0, 1, 2]
    #         indices = tf.constant([], dtype=tf.int32)
    #         num_out_cols = 10
    #         scatter = spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #         with self.assertRaisesOpError("Indices cannot be empty."):
    #             sess.run(scatter)
    #
    # def testScalarParams(self):
    #     params = 10
    #     indices = [1, 2, 3]
    #     num_out_cols = 10
    #     with self.assertRaises(ValueError):
    #         spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #
    # def testScalarIndices(self):
    #     params = [1, 2, 3]
    #     indices = 1
    #     num_out_cols = 10
    #     with self.assertRaises(ValueError):
    #         spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #
    # def test3DParams(self):
    #     params = [[[0, 1, 2]]]
    #     indices = [1, 2, 3]
    #     num_out_cols = 10
    #     with self.assertRaises(ValueError):
    #         spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #
    # def test2DIndices(self):
    #     params = [[0, 1, 2]]
    #     indices = [[1, 2, 3]]
    #     num_out_cols = 10
    #     with self.assertRaises(ValueError):
    #         spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #
    # # def test2DPadElem(self):
    # #     params = [[0, 1, 2]]
    # #     indices = [1, 2, 3]
    # #     num_out_cols = 5
    # #     with self.assertRaises(ValueError):
    # #         spn.ops.scatter_values(
    # #             params, indices, num_out_cols)
    #
    # def testNegativeIndices_CPU(self):
    #     with self.test_session(use_gpu=False) as sess:
    #         params = tf.constant([[1, 2, 3]], dtype=tf.float64)
    #         indices = [1, -1, 0]
    #         num_out_cols = 6
    #         scatter = spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #         with self.assertRaisesOpError("Indices\(1\): -1 is not in range "
    #                                       "\(0, 6\]."):
    #             sess.run(scatter)
    #
    # # def testNegativeIndices_GPU(self):
    # #     with self.test_session(use_gpu=True) as sess:
    # #         params = tf.constant([[1, 2, 3]], dtype=tf.float64)
    # #         indices = [1, -1, 0]
    # #         num_out_cols = 6
    # #         scatter = spn.ops.scatter_values(
    # #             params, indices, num_out_cols)
    # #         with self.assertRaisesOpError("Indices\(1\): -1 is not in range "
    # #                                       "\(0, 6\]."):
    # #             sess.run(scatter)
    #
    # def testBadIndices_CPU(self):
    #     with self.test_session(use_gpu=False) as sess:
    #         params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
    #         indices = tf.constant([2, 1, 10, 6, 5], dtype=tf.int32)
    #         num_out_cols = 7
    #         scatter = spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #         with self.assertRaisesOpError("Indices\(2\): 10 is not in range "
    #                                       "\(0, 7\]."):
    #             sess.run(scatter)
    #
    # # def testBadIndices_GPU(self):
    # #     with self.test_session(use_gpu=True) as sess:
    # #         params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
    # #         indices = tf.constant([2, 1, 10, 6, 5], dtype=tf.int32)
    # #         num_out_cols = 7
    # #         scatter = spn.ops.scatter_values(
    # #             params, indices, num_out_cols)
    # #         with self.assertRaisesOpError("Indices\(2\): 10 is not in range "
    # #                                       "\(0, 7\]."):
    # #             sess.run(scatter)
    #
    # def testDuplicateIndices_CPU(self):
    #     with self.test_session(use_gpu=False) as sess:
    #         params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
    #         indices = tf.constant([0, 1, 2, 2, 4], dtype=tf.int32)
    #         num_out_cols = 5
    #         scatter = spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #         with self.assertRaisesOpError(
    #                 "Indices cannot contain duplicates. Total no. of indices: "
    #                 "5 != no. of unique indices: 4."):
    #             sess.run(scatter)
    #
    # # def testDuplicateIndices_GPU(self):
    # #     with self.test_session(use_gpu=True) as sess:
    # #         params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
    # #         indices = tf.constant([0, 1, 2, 2, 4], dtype=tf.int32)
    # #         num_out_cols = 5
    # #         scatter = spn.ops.scatter_values(
    # #             params, indices, num_out_cols)
    # #         with self.assertRaisesOpError(
    # #                 "Indices cannot contain duplicates. Total no. of indices: "
    # #                 "5 != no. of unique indices: 4."):
    # #             sess.run(scatter)
    #
    # def testWrongOutNumCols(self):
    #     with self.test_session(use_gpu=False) as sess:
    #         params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
    #         indices = tf.constant([4, 3, 2, 1, 0], dtype=tf.int32)
    #         num_out_cols = 4
    #         scatter = spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #         with self.assertRaisesOpError("num_out_cols: 4 must be >= size of "
    #                                       "the indexed dimension of params: 5"):
    #             sess.run(scatter)
    #
    # def testIncorrectIndicesSize(self):
    #     with self.test_session(use_gpu=False) as sess:
    #         params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
    #         indices = tf.constant([11, 10, 9, 8], dtype=tf.int32)
    #         num_out_cols = 12
    #         scatter = spn.ops.scatter_values(
    #             params, indices, num_out_cols)
    #         with self.assertRaisesOpError(
    #                 "Size of indices: 4 and the indexed dimension "
    #                 "of params - 5 - must be the same."):
    #             sess.run(scatter)

    def test_scatter_values(self):
        ops.RegisterShape("ScatterValues")(common_shapes.call_cpp_shape_fn)

        def test(params, indices, num_out_cols, param_dtype, ind_dtype,
                 true_output, use_gpu=False):

            if use_gpu:
                device = [False, True]
            else:
                device = [False]

            for p_dt in param_dtype:
                for i_dt in ind_dtype:
                    for dev in device:
                        with self.test_session(use_gpu=dev) as sess:
                            row1 = 1
                            row2 = -1
                            row3 = 2

                            print("\nDevice: %s" % (("GPU" if dev is True else "CPU")))
                            print("    params_dtype: %s    indices_dtype: %s" %
                                  (re.search('<dtype: (.+?)>', str(p_dt)).group(1),
                                   re.search('<dtype: (.+?)>', str(i_dt)).group(1)))

                            # Convert params and output to appropriate data types
                            if p_dt == tf.float32 or p_dt == tf.float64:
                                par = list(map(float, params))
                                if isinstance(true_output[0], collections.Iterable):
                                    t_out = [list(map(float, to)) for to in true_output]
                                else:
                                    t_out = list(map(float, true_output))
                            else:
                                par = list(map(int, params))
                                if isinstance(true_output[0], collections.Iterable):
                                    t_out = [list(map(int, to)) for to in true_output]
                                else:
                                    t_out = list(map(int, true_output))

                            p1d = tf.constant(np.array(par), dtype=p_dt)
                            p2d1 = tf.constant(np.array([np.array(par)]),
                                                        dtype=p_dt)
                            p2d2 = tf.constant(np.array([np.array(par) * row1,
                                                         np.array(par) * row2,
                                                         np.array(par) * row3]),
                                               dtype=p_dt)

                            ind1d = tf.constant(np.array(indices), dtype=i_dt)
                            ind2d1 = tf.constant(np.array([np.array(indices)]),
                                                 dtype=i_dt)
                            ind2d2 = tf.constant(np.array([np.array(indices),
                                                           np.array(indices),
                                                           np.array(indices)]),
                                                 dtype=i_dt)

                            op1d = spn.ops.scatter_values(p1d, ind1d, num_out_cols)
                            op2d1 = spn.ops.scatter_values(p2d1, ind2d1, num_out_cols)
                            op2d2 = spn.ops.scatter_values(p2d2, ind2d2, num_out_cols)

                            out1d = sess.run(op1d)
                            out2d1 = sess.run(op2d1)
                            out2d2 = sess.run(op2d2)

                            # Test outputs
                            np.testing.assert_array_almost_equal(out1d,
                                                       np.array(t_out))
                            self.assertEqual(p_dt.as_numpy_dtype, out1d.dtype)
                            np.testing.assert_array_equal(op1d.get_shape(),
                                                    list(np.array(t_out).shape))

                            t_out_2d1 = [np.array(t_out)]
                            np.testing.assert_array_almost_equal(out2d1,
                                                                 t_out_2d1)
                            self.assertEqual(p_dt.as_numpy_dtype, out2d1.dtype)
                            np.testing.assert_array_equal(op2d1.get_shape(),
                                                list(np.array(t_out_2d1).shape))

                            t_out_2d2 = [np.array(t_out) * row1,
                                         np.array(t_out) * row2,
                                         np.array(t_out) * row3]
                            np.testing.assert_array_almost_equal(out2d2,
                                                      np.array(t_out_2d2))
                            self.assertEqual(p_dt.as_numpy_dtype, out2d2.dtype)
                            np.testing.assert_array_equal(op2d2.get_shape(),
                                                list(np.array(t_out_2d2).shape))


        float_val = 1.23456789

        # Single param, single index
        # Without padding - Only scatter
        test([float_val],
             [0],
             1,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[float_val]],
             use_gpu=True)

        # With padding
        test([float_val],
             [1],
             4,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[0.0, float_val, 0.0, 0.0]],
             use_gpu=True)

        # Multiple params, multiple indices
        # Without padding - Only scatter
        test([float_val, float_val*2, float_val*3, float_val*4],
             [0, 0, 0, 0],
             1,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[float_val],
              [float_val*2],
              [float_val*3],
              [float_val*4]],
             use_gpu=True)

        # With padding
        test([float_val, float_val*2, float_val*3, float_val*4],
             [1, 4, 2, 0],
             5,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[0.0, float_val, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, float_val*2],
              [0.0, 0.0, float_val*3, 0.0, 0.0],
              [float_val*4, 0.0, 0.0, 0.0, 0.0]],
             use_gpu=True)


if __name__ == '__main__':
    unittest.main()

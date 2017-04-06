#!/usr/bin/env python3

import tensorflow as tf
import sys
import numpy as np
from context import libspn as spn
import time
import argparse

libspn_ops_module = tf.load_op_library('./libspn_ops.so')


def fun_custom(params, indices, out_num_cols):
    scatter_out = libspn_ops_module.scatter_columns(params, indices, 0, out_num_cols)
    return libspn_ops_module.gather_columns(scatter_out, indices)


def fun_tfindexing(params, indices, out_num_cols):
    zero_col = tf.zeros((tf.shape(params)[0], 1), dtype=params.dtype)
    with_zeros = tf.concat_v2(values=(zero_col, params), axis=1)
    gather_indices = np.zeros(out_num_cols, dtype=int)
    gather_indices[indices] = np.arange(indices.size) + 1
    scatter_out = tf.stack([with_zeros[:, c] for c in gather_indices], -1)
    return libspn_ops_module.gather_columns(scatter_out, indices)


def fun_scatter(params, indices, out_num_cols):
    scatter_out = spn.utils.scatter_cols(params, indices, out_num_cols)
    return libspn_ops_module.gather_columns(scatter_out, indices)


def printc(string):
    COLOR = '\033[1m\033[93m'
    ENDC = '\033[0m'
    print(COLOR + string + ENDC)


class TestScatterColumnsPerformance(tf.test.TestCase):

    @classmethod
    def setUpClass(self):
        # Params
        self.dtype = tf.float32
        printc("Info:")
        printc("- num_cols: %s" % self.num_cols)
        printc("- num_rows: %s" % self.num_rows)
        printc("- out_num_cols: %s" % self.out_num_cols)
        printc("- num_stacked_ops: %s" % self.num_stacked_ops)
        printc("- log_device_placement: %s" % self.log_device_placement)
        printc("- dtype: %s" % self.dtype)
        # Generate params matrix
        self.params = np.random.rand(self.num_rows, self.num_cols)
        self.params = np.asarray(self.params,
                                 dtype=self.dtype.as_numpy_dtype())

    def run_test(self, fun, indices, device_name):
        with self.test_session(config=tf.ConfigProto(
                log_device_placement=self.log_device_placement)) as sess:
            with tf.device(device_name):
                indices = np.asarray(indices, dtype=np.int32)

                # Create an op stack
                op = tf.constant(self.params, dtype=self.dtype)
                for i in range(self.num_stacked_ops):
                    op = fun(op, indices, self.out_num_cols)

            # Run
            start_time = time.time()
            op_out = sess.run(op)
            total_time = time.time() - start_time

            # Print stats
            # To print processing time of each individual op, use 'make debug'
            # instead, which enables the EXEC_TIME_CALC debug flag.
            printc("Total time for case %s on %s: %.5f s" %
                   (self.id().split('.')[2].upper(), device_name, total_time))

            # Test generated output
            np.testing.assert_array_almost_equal(op_out, self.params)

    def run_test_opt0(self, fun, device_name):
        """Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test(
            fun,
            list(range(0, self.out_num_cols, 2)),  # indices: [0, 2, 4, ..., 2n-2]
            device_name=device_name)

    def run_test_opt25(self, fun, device_name):
        """Case: In-op optimization 25%"""
        # indices: [0, 1, 4, 5, 8, 9, ..., 2n-4, 2n-3]
        ind = spn.utils.range_with_blocksize(0, (self.num_cols * 2), 2, 4)
        self.run_test(fun, ind, device_name=device_name)

    def run_test_opt37(self, fun, device_name):
        """Case: In-op optimization 37%"""
        # indices: [0, 1, 2, 3, 8, 9, 10, 11 ..., 2n-7, 2n-6, 2n-5, 2n-4]
        ind = spn.utils.range_with_blocksize(0, (self.num_cols * 2), 4, 8)
        self.run_test(fun, ind, device_name=device_name)

    def run_test_opt50(self, fun, device_name):
        """Case: In-op optimization 50%"""
        # indices: [0, 1, 2, ..., (n/2)-1, (3n/2), (3n/2)+1, ..., 2n-2, 2n-1]
        ind = spn.utils.range_with_blocksize(0, (self.num_cols * 2),
                                             (self.num_cols // 2),
                                             (self.num_cols // 2) * 3)
        self.run_test(fun, ind, device_name=device_name)

    def run_test_tfindexing_opt0(self, device_name):
        """Method: TF Indexing
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_opt0(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt0(self):
        """Method: TF Indexing
           Device: CPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_tfindexing_opt0('/cpu:0')

    def test_tfindexing_gpu_opt0(self):
        """Method: TF Indexing
           Device: GPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_tfindexing_opt0('/gpu:0')

    def run_test_scatter_opt0(self, device_name):
        """Method: spn scatter
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_opt0(fun=fun_scatter, device_name=device_name)

    def test_scatter_cpu_opt0(self):
        """Method: spn scatter
           Device: CPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_scatter_opt0('/cpu:0')

    def test_scatter_gpu_opt0(self):
        """Method: spn scatter
           Device: GPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_scatter_opt0('/gpu:0')

    def run_test_custom_opt0(self, device_name):
        """Method: custom scatter_cols op
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_opt0(fun=fun_custom,
                           device_name=device_name)

    def test_custom_cpu_opt0(self):
        """Method: custom scatter_cols op
           Device: CPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_custom_opt0('/cpu:0')

    def test_custom_gpu_opt0(self):
        """Method: custom scatter_cols op
           Device: GPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_custom_opt0('/gpu:0')

    def run_test_tfindexing_opt25(self, device_name):
        """Method: TF Indexing
           Case: In-op optimization 25%"""

        self.run_test_opt25(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt25(self):
        """Method: TF Indexing
           Device: CPU
           Case: In-op optimization 25%"""

        self.run_test_tfindexing_opt25('/cpu:0')

    def test_tfindexing_gpu_opt25(self):
        """Method: TF Indexing
           Device: GPU
           Case: In-op optimization 25%"""

        self.run_test_tfindexing_opt25('/gpu:0')

    def run_test_scatter_opt25(self, device_name):
        """Method: spn scatter
           Case: In-op optimization 25%"""

        self.run_test_opt25(fun=fun_scatter, device_name=device_name)

    def test_scatter_cpu_opt25(self):
        """Method: spn scatter
           Device: CPU
           Case: In-op optimization 25%"""

        self.run_test_scatter_opt25('/cpu:0')

    def test_scatter_gpu_opt25(self):
        """Method: spn scatter
           Device: GPU
           Case: In-op optimization 25%"""

        self.run_test_scatter_opt25('/gpu:0')

    def run_test_custom_opt25(self, device_name):
        """Method: custom scatter_cols op
           Case: In-op optimization 25%"""

        self.run_test_opt25(fun=fun_custom,
                            device_name=device_name)

    def test_custom_cpu_opt25(self):
        """Method: custom scatter_cols op
           Device: CPU
           Case: In-op optimization 25%"""

        self.run_test_custom_opt25('/cpu:0')

    def test_custom_gpu_opt25(self):
        """Method: custom scatter_cols op
           Device: GPU
           Case: In-op optimization 25%"""

        self.run_test_custom_opt25('/gpu:0')

    def run_test_tfindexing_opt37(self, device_name):
        """Method: TF Indexing
           Case: In-op optimization 37%"""

        self.run_test_opt37(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt37(self):
        """Method: TF Indexing
           Device: CPU
           Case: In-op optimization 37%"""

        self.run_test_tfindexing_opt37('/cpu:0')

    def test_tfindexing_gpu_opt37(self):
        """Method: TF Indexing
           Device: GPU
           Case: In-op optimization 37%"""

        self.run_test_tfindexing_opt37('/gpu:0')

    def run_test_scatter_opt37(self, device_name):
        """Method: spn scatter
           Case: In-op optimization 37%"""

        self.run_test_opt37(fun=fun_scatter, device_name=device_name)

    def test_scatter_cpu_opt37(self):
        """Method: spn scatter
           Device: CPU
           Case: In-op optimization 37%"""

        self.run_test_scatter_opt37('/cpu:0')

    def test_scatter_gpu_opt37(self):
        """Method: spn scatter
           Device: GPU
           Case: In-op optimization 37%"""

        self.run_test_scatter_opt37('/gpu:0')

    def run_test_custom_opt37(self, device_name):
        """Method: custom scatter_cols op
           Case: In-op optimization 37%"""

        self.run_test_opt37(fun=fun_custom,
                            device_name=device_name)

    def test_custom_cpu_opt37(self):
        """Method: custom scatter_cols op
           Device: CPU
           Case: In-op optimization 37%"""

        self.run_test_custom_opt37('/cpu:0')

    def test_custom_gpu_opt37(self):
        """Method: custom scatter_cols op
           Device: GPU
           Case: In-op optimization 37%"""

        self.run_test_custom_opt37('/gpu:0')

    def run_test_tfindexing_opt50(self, device_name):
        """Method: TF Indexing
           Case: In-op optimization 50%"""

        self.run_test_opt50(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt50(self):
        """Method: TF Indexing
           Device: CPU
           Case: In-op optimization 50%"""

        self.run_test_tfindexing_opt50('/cpu:0')

    def test_tfindexing_gpu_opt50(self):
        """Method: TF Indexing
           Device: GPU
           Case: In-op optimization 50%"""

        self.run_test_tfindexing_opt50('/gpu:0')

    def run_test_scatter_opt50(self, device_name):
        """Method: spn scatter
           Case: In-op optimization 50%"""

        self.run_test_opt50(fun=fun_scatter, device_name=device_name)

    def test_scatter_cpu_opt50(self):
        """Method: spn scatter
           Device: CPU
           Case: In-op optimization 50%"""

        self.run_test_scatter_opt50('/cpu:0')

    def test_scatter_gpu_opt50(self):
        """Method: spn scatter
           Device: GPU
           Case: In-op optimization 50%"""

        self.run_test_scatter_opt50('/gpu:0')

    def run_test_custom_opt50(self, device_name):
        """Method: custom scatter_cols op
           Case: In-op optimization 50%"""

        self.run_test_opt50(fun=fun_custom,
                            device_name=device_name)

    def test_custom_cpu_opt50(self):
        """Method: custom scatter_cols op
           Device: CPU
           Case: In-op optimization 50%"""

        self.run_test_custom_opt50('/cpu:0')

    def test_custom_gpu_opt50(self):
        """Method: custom scatter_cols op
           Device: GPU
           Case: In-op optimization 50%"""

        self.run_test_custom_opt50('/gpu:0')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cols', default=40, type=int)
    parser.add_argument('--num-rows', default=1000, type=int)
    parser.add_argument('--out-num-cols', default=80, type=int)
    parser.add_argument('--num-stacked-ops', default=300, type=int)
    parser.add_argument('--log-device', default=False, type=bool)
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()

    # Verify args
    if args.num_cols % 20:
        args.num_cols = (args.num_cols // 20) * 20 + 20

    if args.out_num_cols / args.num_cols != 2.0:
        args.out_num_cols = args.num_cols * 2

    TestScatterColumnsPerformance.num_cols = args.num_cols
    TestScatterColumnsPerformance.num_rows = args.num_rows
    TestScatterColumnsPerformance.out_num_cols = args.out_num_cols
    TestScatterColumnsPerformance.num_stacked_ops = args.num_stacked_ops
    TestScatterColumnsPerformance.log_device_placement = args.log_device
    sys.argv[1:] = args.unittest_args

    tf.test.main()

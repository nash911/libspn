#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from context import libspn as spn
import time
import argparse
import colorama as col
import sys
col.init()


def print1(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.YELLOW + str + col.Style.RESET_ALL)


def print2(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.BLUE + str + col.Style.RESET_ALL)


class Sum:

    def create_place_holders(inputs, ivs_inp, ivs_hidden, num_sums):
        # Create placeholders for inputs and ivs
        input_size = inputs.shape[1]
        inputs_pl = spn.ContVars(num_vars=input_size)

        if ivs_inp is not None:
            ivs_inp_pl = spn.IVs(num_vars=1, num_vals=input_size)
            ivs_hiden_pl = spn.IVs(num_vars=1, num_vals=num_sums)
        else:
            ivs_inp_pl = None
            ivs_hiden_pl = None

        return inputs_pl, ivs_inp_pl, ivs_hiden_pl

    def create_graph(inputs, indices_inp, indices_hidden, ivs_inp, ivs_hidden,
                     num_sums, num_layers):
        if indices_inp is None:
            inputs = [inputs]
        else:
            inputs = [(inputs, indices_inp)]

        L = [] # List of layers

        # Create 'num_layers' layers of Sum nodes, connecting each later to
        # the layer below
        for l in range(0, num_layers):
            l_n = [] # List os Sum nodes in a single later
            # Generate 'num_sums' Sum nodes, connecting each to inputs/later below
            # and their respective ivs
            for _ in range(0, num_sums):
                if l == 0: # First layer connected to inputs and Ivs
                    l_n = l_n + [spn.Sum(*inputs, ivs=ivs_inp)]
                else: # Second layer onwards connected to the layer below
                    l_n = l_n + [spn.Sum(*L[-1], ivs=ivs_hidden)]

                # Generate weights for each Sum node
                l_n[-1].generate_weights()

            if indices_hidden is None or l == num_layers-1: # No indices for the root node
                L.append(l_n)
            else:
                L.append([(s, indices_hidden) for s in l_n])

        # Connect all sum nodes in the top most hidden layer to a single root
        # Sum node and generate its weights
        root = spn.Sum(*L[-1])
        root.generate_weights()
        return root

    def true_value(inputs, ivs_inp, num_sums, num_layers):
        input_size = inputs.shape[1]
        input_weight = 1.0 / input_size
        hidden_weight = 1.0 / num_sums

        # Compute true output with numpy
        if ivs_inp is None:
            true_value = np.sum((inputs * input_weight), axis=1, keepdims=True)
        else:
            ivs_inp_oh = np.eye(input_size)[np.squeeze(ivs_inp)]
            true_value = np.sum((inputs * ivs_inp_oh) * input_weight, axis=1,
                                keepdims=True) * np.power(hidden_weight,
                                num_layers) * num_sums
        return true_value

class Sums:

    def create_place_holders(inputs, ivs_inp, ivs_hidden, num_sums):
        # Create placeholders for inputs and ivs
        input_size = inputs.shape[1]
        inputs_pl = spn.ContVars(num_vars=input_size)
        if ivs_inp is not None:
            ivs_inp_pl = spn.IVs(num_vars=num_sums,
                                 num_vals=int(input_size / num_sums))
            ivs_hidden_pl = spn.IVs(num_vars=num_sums, num_vals=num_sums)
        else:
            ivs_inp_pl = None
            ivs_hidden_pl = None
        return inputs_pl, ivs_inp_pl, ivs_hidden_pl

    def create_graph(inputs, indices_inp, indices_hidden, ivs_inp, ivs_hidden,
             num_sums, num_layers):
        if indices_inp is None:
            inputs = [inputs]
        else:
            inputs = [(inputs, indices_inp)]

        L = [] # List of layers

        # Create a 'num_layers' layers with each layer containing a single Sums
        # node, modeling 'num_sums' sums within
        for l in range(0, num_layers):
            if l == 0: # First layer connected to inputs and Ivs
                l_n = spn.Sums(*inputs, num_sums=num_sums, ivs=ivs_inp)
            else: # Second layer onwards connected to the layer below
                l_n = spn.Sums(*[L[-1] for _ in range(num_sums)],
                               num_sums=num_sums, ivs=ivs_hidden)

            # Generate weights for each Sums node
            l_n.generate_weights()

            if indices_hidden is None or l == num_layers-1: # No indices for the root node
                L.append(l_n)
            else:
                L.append((l_n, indices_hidden))

        # Connect the Sums node in the top most hidden layer to a single root
        # Sum node and generate its weights
        root = spn.Sum(L[-1])
        root.generate_weights()
        return root

    def true_value(inputs, ivs_inp, num_sums, num_layers):
        batch_size = inputs.shape[0]
        input_size = int(inputs.shape[1] / num_sums)
        input_weight = 1.0 / input_size
        hidden_weight = 1.0 / num_sums
        input_slice = np.split(inputs, num_sums, axis=1)[0]

        # Compute true output with numpy
        if ivs_inp is None:
            true_value = np.sum((input_slice * input_weight), axis=1,
                                keepdims=True)# * np.power((hidden_weight * num_sums), num_layers)
        else:
            ivs_inp_slice = np.split(ivs_inp, num_sums, axis=1)[0]
            ivs_inp_oh = np.eye(input_size)[np.squeeze(ivs_inp_slice)]
            true_value = np.sum((input_slice * ivs_inp_oh) * input_weight, axis=1,
                                keepdims=True) * np.power(hidden_weight,
                                num_layers) * num_sums
        return true_value


class ParallelSums:

    def create_place_holders(inputs, ivs_inp, ivs_hidden, num_sums):
        # Create placeholders for inputs and ivs
        input_size = inputs.shape[1]
        inputs_pl = spn.ContVars(num_vars=input_size)
        if ivs_inp is not None:
            ivs_inp_pl = spn.IVs(num_vars=num_sums, num_vals=input_size)
            ivs_hidden_pl = spn.IVs(num_vars=num_sums, num_vals=num_sums)
        else:
            ivs_inp_pl = None
            ivs_hidden_pl = None
        return inputs_pl, ivs_inp_pl, ivs_hidden_pl

    def create_graph(inputs, indices_inp, indices_hidden, ivs_inp, ivs_hidden,
             num_sums, num_layers):
        if indices_inp is None:
            inputs = [inputs]
        else:
            inputs = [(inputs, indices_inp)]

        L = [] # List of layers

        # Create a 'num_layers' layers with each layer containing a single
        # ParallelSums node, modeling 'num_sums' sums within
        for l in range(0, num_layers):
            if l == 0: # First layer connected to inputs and Ivs
                l_n = spn.ParallelSums(*inputs, num_sums=num_sums, ivs=ivs_inp)
            else: # Second layer onwards connected to the layer below
                l_n = spn.ParallelSums(L[-1], num_sums=num_sums, ivs=ivs_hidden)

            # Generate weights for each Sums node
            l_n.generate_weights()

            if indices_hidden is None or l == num_layers-1: # No indices for the root node
                L.append(l_n)
            else:
                L.append((l_n, indices_hidden))

        # Connect the ParallelSums node in the top most hidden layer to a single
        # root Sum node and generate its weights
        root = spn.Sum(L[-1])
        root.generate_weights()
        return root

    def true_value(inputs, ivs_inp, num_sums, num_layers):
        input_size = int(inputs.shape[1])
        input_weight = 1.0 / input_size
        hidden_weight = 1.0 / num_sums

        # Compute true output with numpy
        if ivs_inp is None:
            true_value = np.sum((inputs * input_weight), axis=1, keepdims=True)
        else:
            ivs_inp_slice = np.split(ivs_inp, num_sums, axis=1)[0]
            ivs_inp_oh = np.eye(input_size)[np.squeeze(ivs_inp_slice)]
            true_value = np.sum((inputs * ivs_inp_oh) * input_weight, axis=1,
                                keepdims=True) * np.power(hidden_weight,
                                num_layers) * num_sums

        return true_value


class NodeTestResult:
    """Result of a single test of a single op."""

    def __init__(self, node_name, on_gpu, with_indices, with_ivs, graph_size,
                 setup_time, run_times, output_correct):
        self.node_name = node_name
        self.on_gpu = on_gpu
        self.with_indices = with_indices
        self.with_ivs = with_ivs
        self.graph_size = graph_size
        self.setup_time = setup_time
        self.run_times = run_times
        self.output_correct = output_correct


class TestResults:
    """Results for a single test for multiple ops and devices."""

    def __init__(self, test_name, cpu_results, gpu_results):
        self.test_name = test_name
        self.cpu_results = cpu_results
        self.gpu_results = gpu_results

    def print(self, file):
        def get_header(dev):
            return ("%3s %11s %5s %5s %5s %11s %15s %14s %10s" %
                    (dev, 'op', 'Indices', 'IVs', 'size', 'setup_time',
                     'first_run_time', 'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%15s %5s %7s %5d %11.2f %15.2f %14.2f %10s" %
                    (res.node_name, ("Yes" if res.with_indices else "No"),
                    ("Yes" if res.with_ivs else "No"),
                     res.graph_size,
                     res.setup_time * 1000, res.run_times[0] * 1000,
                     np.mean(res.run_times[1:]) * 1000,
                     res.output_correct))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: x.node_name):
            print1(get_res(res), file)
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: x.node_name):
            print1(get_res(res), file)


class PerformanceTest:

    def __init__(self, batch_size, input_size, num_sums,
                 num_layers, num_runs, without_cpu,
                 without_gpu, log_devs, file):
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_sums = num_sums
        self.num_layers = num_layers
        self.num_runs = num_runs
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.file = file


        print1("Params:", file)
        print1("- batch_size=%s" % batch_size, file)
        print1("- input_size=%s" % input_size, file)
        print1("- num_sums=%s" % num_sums, file)
        print1("- num_layers=%s" % num_layers, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("", file=file)

    def _run_op_test(self, node, inputs, indices_inp, indices_hidden,
                     ivs_inp, ivs_hidden, on_gpu):
        """Run a single test for a single op."""
        # Preparations
        node_name = node.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, with_ivs=%s, with_ivs=%s, params_shape=%s, indices_shape=%s"
               % (node_name, on_gpu, ("No" if indices_inp is None else "Yes"),
                  ("No" if ivs_hidden is None else "Yes"), inputs.shape, 1),
                  self.file)

        # Compute true output with numpy
        true_value_out = node.true_value(inputs, ivs_inp, self.num_sums,
                                         self.num_layers)

        # Clear any previous graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input and IVs
            inputs_pl, ivs_inp_pl, ivs_hidden_pl = \
              node.create_place_holders(inputs, ivs_inp, ivs_hidden, self.num_sums)

            # Create graph
            start_time = time.time()
            root = node.create_graph(inputs_pl, indices_inp, indices_hidden,
                                     ivs_inp_pl, ivs_hidden_pl, self.num_sums,
                                     self.num_layers)
            ops = root.get_value(inference_type=spn.InferenceType.MARGINAL)
            setup_time = time.time() - start_time

        # Get num of graph ops
        graph_size = len(tf.get_default_graph().get_operations())

        # Run multiple times
        output_correct = True
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            #Initialize weights of all the sum nodes in the graph
            spn.initialize_weights(root).run()

            # Create feed dictionary
            feed = {inputs_pl: inputs}
            if ivs_inp is not None:
                feed[ivs_inp_pl] = ivs_inp
                feed[ivs_hidden_pl] = ivs_hidden

            run_times = []
            for n in range(self.num_runs):
                # Run
                start_time = time.time()
                value_out = sess.run(ops, feed_dict=feed)
                run_times.append(time.time() - start_time)

                # Test value
                try:
                    np.testing.assert_array_almost_equal(value_out, true_value_out,
                                                         decimal=5)
                except AssertionError:
                    output_correct = False
        # Return stats
        return NodeTestResult(node_name, on_gpu, (False if indices_inp is None else True),
                              (False if ivs_hidden is None else True), graph_size,
                              setup_time, run_times, output_correct)

    def _run_test(self, test_name, nodes, inputs, indices_inp, indices_hidden,
                  ivs_inp, ivs_hidden):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for node in nodes:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(node, inputs, indices_inp, indices_hidden,
                                      None, None, on_gpu=False))
                cpu_results.append(
                    self._run_op_test(node, inputs, indices_inp, indices_hidden,
                                      ivs_inp, ivs_hidden,
                                      on_gpu=False))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(node, inputs, indices_inp, indices_hidden,
                                      None, None, on_gpu=True))
                gpu_results.append(
                    self._run_op_test(node, inputs, indices_inp, indices_hidden,
                                      ivs_inp, ivs_hidden,
                                      on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)


    def _run_all_nodes(self):
        """Run all 2D tests."""

        results = []

        inputs = np.random.rand(self.batch_size, self.input_size)
        indices_inp = list(range(self.input_size))
        indices_hidden = [0]
        ivs_inp = np.expand_dims(np.random.randint(self.input_size,
                                               size=self.batch_size), axis=1)
        ivs_hidden = np.expand_dims(np.random.randint(self.num_sums,
                                               size=self.batch_size), axis=1)
        r = self._run_test('Sum',
                           [Sum],
                           inputs, indices_inp, indices_hidden,
                           ivs_inp, ivs_hidden)
        results.append(r)

        inputs = np.random.rand(self.batch_size, self.input_size)
        inputs_tiled = np.tile(inputs, self.num_sums)
        indices_inp = list(range(self.input_size * self.num_sums))
        indices_hidden = list(range(self.num_sums))
        ivs_inp = np.expand_dims(np.random.randint(self.input_size,
                                                   size=self.batch_size), axis=1)
        ivs_hidden = np.expand_dims(np.random.randint(self.num_sums,
                                                      size=self.batch_size), axis=1)
        ivs_inp_tiled = np.tile(ivs_inp, (1, self.num_sums))
        ivs_hidden_tiled = np.tile(ivs_hidden, (1, self.num_sums))
        r = self._run_test('Sums',
                           [Sums],
                           inputs_tiled, indices_inp, indices_hidden,
                           ivs_inp_tiled, ivs_hidden_tiled)
        results.append(r)

        inputs = np.random.rand(self.batch_size, self.input_size)
        indices_inp = list(range(self.input_size))
        indices_hidden = list(range(self.num_sums))
        ivs_inp = np.expand_dims(np.random.randint(self.input_size,
                                                   size=self.batch_size), axis=1)
        ivs_hidden = np.expand_dims(np.random.randint(self.num_sums,
                                                      size=self.batch_size), axis=1)
        ivs_inp_tiled = np.tile(ivs_inp, (1, self.num_sums))
        ivs_hidden_tiled = np.tile(ivs_hidden, (1, self.num_sums))
        r = self._run_test('ParallelSums',
                           [ParallelSums],
                           inputs, indices_inp, indices_hidden,
                           ivs_inp_tiled, ivs_hidden_tiled)
        results.append(r)

        return results

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []
        results += self._run_all_nodes()

        # Print results
        for res in results:
            res.print(self.file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=200, type=int,
                        help="Batch size of inputs")
    parser.add_argument('--input-size', default=100, type=int,
                        help="Num of input variables")
    parser.add_argument('--num-sums', default=100, type=int,
                        help="Num of indices used for SOME tests")
    parser.add_argument('--num-layers', default=10, type=int,
                        help="Num of sums modeled per graph")
    parser.add_argument('--num-runs', default=10, type=int,
                        help="Number of times each test is run")
    parser.add_argument('--log-devices', action='store_true',
                        help="Log on which device op is run. Affects run time!")
    parser.add_argument('--without-cpu', action='store_true',
                        help="Do not run CPU tests")
    parser.add_argument('--without-gpu', action='store_true',
                        help="Do not run GPU tests")
    parser.add_argument('--save-to', default='', type=str,
                        help="Save results to file")
    dtype = tf.float32
    args = parser.parse_args()

    if args.num_layers < 1:
        sys.exit('ERROR: num_layers must a positive integer')

    if args.num_sums < 1:
        sys.exit('ERROR: num_sums must a positive integer')


    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.batch_size, args.input_size, args.num_sums,
                            args.num_layers, args.num_runs, args.without_cpu,
                            args.without_gpu, args.log_devices, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()

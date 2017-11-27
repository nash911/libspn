#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestInference(TestCase):

    def test_marginal_value(self):
        """Calculation of SPN marginal value"""
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Set default inference type for each node
        model.root.set_inference_types(spn.InferenceType.MARGINAL)
        # Get values
        init = spn.initialize_weights(model.root)
        val_marginal = model.root.get_value(
            inference_type=spn.InferenceType.MARGINAL)
        val_default = model.root.get_value()
        val_log_marginal = model.root.get_log_value(
            inference_type=spn.InferenceType.MARGINAL)
        val_log_default = model.root.get_log_value()
        with tf.Session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={model.ivs: model.feed})
            out_marginal = sess.run(val_marginal, feed_dict={model.ivs: model.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={model.ivs: model.feed})
            out_log_marginal = sess.run(tf.exp(val_log_marginal),
                                        feed_dict={model.ivs: model.feed})
        # Check if values sum to 1
        # WARNING: Below does not pass test for places=7 with float32 dtype
        self.assertAlmostEqual(out_default[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_marginal[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_log_default[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_log_marginal[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        # Check joint probabilities
        np.testing.assert_array_almost_equal(out_default, model.true_values)
        np.testing.assert_array_almost_equal(out_marginal, model.true_values)
        np.testing.assert_array_almost_equal(out_log_default, model.true_values)
        np.testing.assert_array_almost_equal(out_log_marginal, model.true_values)

    def test_mpe_value(self):
        """Calculation of SPN MPE value"""
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Set default inference type for each node
        model.root.set_inference_types(spn.InferenceType.MPE)
        # Get values
        init = spn.initialize_weights(model.root)
        val_mpe = model.root.get_value(inference_type=spn.InferenceType.MPE)
        val_default = model.root.get_value()
        val_log_mpe = model.root.get_log_value(inference_type=spn.InferenceType.MPE)
        val_log_default = model.root.get_log_value()
        with tf.Session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={model.ivs: model.feed})
            out_mpe = sess.run(val_mpe, feed_dict={model.ivs: model.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={model.ivs: model.feed})
            out_log_mpe = sess.run(tf.exp(val_log_mpe),
                                   feed_dict={model.ivs: model.feed})
        # Check joint probabilities
        np.testing.assert_array_almost_equal(out_default, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_mpe, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_default, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_mpe, model.true_mpe_values)

    def test_mixed_value(self):
        """Calculation of a mixed MPE/marginal value"""
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Set default inference type for each node
        model.root.set_inference_types(spn.InferenceType.MARGINAL)
        model.root.inference_type = spn.InferenceType.MPE
        # Get values
        init = spn.initialize_weights(model.root)
        val_marginal = model.root.get_value(inference_type=spn.InferenceType.MARGINAL)
        val_mpe = model.root.get_value(inference_type=spn.InferenceType.MPE)
        val_default = model.root.get_value()
        val_log_marginal = model.root.get_log_value(inference_type=spn.InferenceType.MARGINAL)
        val_log_mpe = model.root.get_log_value(inference_type=spn.InferenceType.MPE)
        val_log_default = model.root.get_log_value()
        with tf.Session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={model.ivs: model.feed})
            out_marginal = sess.run(val_marginal, feed_dict={model.ivs: model.feed})
            out_mpe = sess.run(val_mpe, feed_dict={model.ivs: model.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={model.ivs: model.feed})
            out_log_marginal = sess.run(tf.exp(val_log_marginal),
                                        feed_dict={model.ivs: model.feed})
            out_log_mpe = sess.run(tf.exp(val_log_mpe),
                                   feed_dict={model.ivs: model.feed})
        # Check joint probabilities
        true_default = [[0.5],
                        [0.35],
                        [0.15],
                        [0.2],
                        [0.14],
                        [0.06],
                        [0.3],
                        [0.216],
                        [0.09]]
        np.testing.assert_array_almost_equal(out_default, true_default)
        np.testing.assert_array_almost_equal(out_marginal, model.true_values)
        np.testing.assert_array_almost_equal(out_mpe, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_default, true_default)
        np.testing.assert_array_almost_equal(out_log_marginal, model.true_values)
        np.testing.assert_array_almost_equal(out_log_mpe, model.true_mpe_values)

    def test_mpe_path(self):
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Add ops
        init = spn.initialize_weights(model.root)
        mpe_path_gen = spn.MPEPath(value_inference_type=spn.InferenceType.MPE,
                                   log=False)
        mpe_path_gen_log = spn.MPEPath(value_inference_type=spn.InferenceType.MPE,
                                       log=True)
        mpe_path_gen.get_mpe_path(model.root)
        mpe_path_gen_log.get_mpe_path(model.root)
        # Run
        with tf.Session() as sess:
            init.run()
            out = sess.run(mpe_path_gen.counts[model.ivs],
                           feed_dict={model.ivs: model.feed})
            out_log = sess.run(mpe_path_gen_log.counts[model.ivs],
                               feed_dict={model.ivs: model.feed})

        true_ivs_counts = np.array([[0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 0., 1.],
                                    [1., 0., 1., 0.],
                                    [1., 0., 1., 0.],
                                    [1., 0., 0., 1.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 0., 1.]],
                                   dtype=spn.conf.dtype.as_numpy_dtype)

        np.testing.assert_array_equal(out, true_ivs_counts)
        np.testing.assert_array_equal(out_log, true_ivs_counts)

    def test_mpe_state(self):
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Add ops
        init = spn.initialize_weights(model.root)
        mpe_state_gen = spn.MPEState(value_inference_type=spn.InferenceType.MPE,
                                     log=False)
        mpe_state_gen_log = spn.MPEState(value_inference_type=spn.InferenceType.MPE,
                                         log=True)
        ivs_state, = mpe_state_gen.get_state(model.root, model.ivs)
        ivs_state_log, = mpe_state_gen_log.get_state(model.root, model.ivs)
        # Run
        with tf.Session() as sess:
            init.run()
            out = sess.run(ivs_state, feed_dict={model.ivs: [[-1, -1]]})
            out_log = sess.run(ivs_state_log, feed_dict={model.ivs: [[-1, -1]]})

        # For now we only compare the actual MPE state for input IVs -1
        np.testing.assert_array_equal(out.ravel(), model.true_mpe_state)
        np.testing.assert_array_equal(out_log.ravel(), model.true_mpe_state)

    def test_probable_path(self):
        # Generate SPN
        iv_x = spn.IVs(num_vars=2, num_vals=2, name="iv_x")

        sum_11 = spn.Sum((iv_x, [0, 1]), name="sum_11")
        sum_11.generate_weights([0.001, 0.999])

        sum_12 = spn.Sum((iv_x, [0, 1]), name="sum_12")
        sum_12.generate_weights([0.9, 0.1])

        sum_21 = spn.Sum((iv_x, [2, 3]), name="sum_21")
        sum_21.generate_weights([0.3, 0.7])

        sum_22 = spn.Sum((iv_x, [2, 3]), name="sum_22")
        sum_22.generate_weights([0.999, 0.001])

        prod_1 = spn.Product(sum_11, sum_21, name="prod_1")
        prod_2 = spn.Product(sum_11, sum_22, name="prod_2")
        prod_3 = spn.Product(sum_12, sum_22, name="prod_3")

        root = spn.Sum(prod_1, prod_2, prod_3, name="root")
        root.generate_weights([0.001, 0.998, 0.001])

        # Add ops
        init = spn.initialize_weights(root)
        sampled_path_gen = spn.SampledPath(value_inference_type=spn.InferenceType.MPE,
                                           log=False)
        sampled_path_gen.get_probable_path(root)
        # Run
        with tf.Session() as sess:
            init.run()
            out = sess.run(sampled_path_gen.counts[iv_x],
                           feed_dict={iv_x: np.ones((5, 2)) * -1})

        true_ivs_counts = np.array([[0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 1., 0.]],
                                   dtype=spn.conf.dtype.as_numpy_dtype)

        np.testing.assert_array_equal(out, true_ivs_counts)

    def test_probable_state(self):
        # Generate SPN
        iv_x12 = spn.IVs(num_vars=2, num_vals=4, name="iv_x12")
        iv_x34 = spn.IVs(num_vars=2, num_vals=4, name="iv_x34")

        # Sub-SPN 1
        sum_11 = spn.Sum((iv_x12, [0, 1, 2, 3]), (iv_x34, [0, 1, 2, 3]), name="sum_11")
        sum_11.generate_weights([0.001, 0.001, 0.001, 0.001,
                                 0.001, 0.001,  0.993, 0.001])

        sum_12 = spn.Sum((iv_x12, [0, 1, 2, 3]), (iv_x34, [0, 1, 2, 3]), name="sum_12")
        sum_12.generate_weights([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        sum_21 = spn.Sum((iv_x12, [4, 5, 6, 7]), (iv_x34, [4, 5, 6, 7]), name="sum_21")
        sum_21.generate_weights([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        sum_22 = spn.Sum((iv_x12, [4, 5, 6, 7]), (iv_x34, [4, 5, 6, 7]), name="sum_22")
        sum_22.generate_weights([0.993, 0.001, 0.001, 0.001,
                                 0.001, 0.001, 0.001, 0.001])

        prod_1 = spn.Product(sum_11, sum_21, name="prod_1")
        prod_2 = spn.Product(sum_11, sum_22, name="prod_2")
        prod_3 = spn.Product(sum_12, sum_22, name="prod_3")

        sub_spn_1 = spn.Sum(prod_1, prod_2, prod_3, name="sub_spn_1")
        sub_spn_1.generate_weights([0.001, 0.998, 0.001])

        # Sub-SPN 2
        sum_31 = spn.Sum((iv_x12, [0, 1, 2, 3]), (iv_x12, [4, 5, 6, 7]), name="sum_31")
        sum_31.generate_weights([0.001, 0.993, 0.001, 0.001,
                                 0.001, 0.001, 0.001, 0.001])

        sum_32 = spn.Sum((iv_x12, [0, 1, 2, 3]), (iv_x12, [4, 5, 6, 7]), name="sum_32")
        sum_32.generate_weights([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        sum_41 = spn.Sum((iv_x34, [0, 1, 2, 3]), (iv_x34, [4, 5, 6, 7]), name="sum_41")
        sum_41.generate_weights([0.001, 0.001, 0.001, 0.001,
                                 0.001, 0.001, 0.001, 0.993])

        sum_42 = spn.Sum((iv_x34, [0, 1, 2, 3]), (iv_x34, [4, 5, 6, 7]), name="sum_42")
        sum_42.generate_weights([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        prod_4 = spn.Product(sum_31, sum_41, name="prod_4")
        prod_5 = spn.Product(sum_31, sum_42, name="prod_5")
        prod_6 = spn.Product(sum_32, sum_42, name="prod_6")

        sub_spn_2 = spn.Sum(prod_4, prod_5, prod_6, name="sub_spn_2")
        sub_spn_2.generate_weights([0.998, 0.001, 0.001])

        # Root
        root = spn.Sum(sub_spn_1, sub_spn_2, name="root")
        root.generate_weights([0.5, 0.5])
        ivs = root.generate_ivs()

        # Add ops
        init = spn.initialize_weights(root)
        sampled_state_gen = spn.SampledState(value_inference_type=spn.InferenceType.MPE,
                                             log=False)
        iv_x12_state, iv_x34_state = sampled_state_gen.get_state(root, iv_x12,
                                                                 iv_x34)
        # Run
        with tf.Session() as sess:
            init.run()
            for i in range(50):
                out = sess.run([iv_x12_state, iv_x34_state],
                               feed_dict={iv_x12: np.ones((2, 2)) * -1,
                                          iv_x34: np.ones((2, 2)) * -1,
                                          ivs: [[0], [1]]})

                np.testing.assert_array_equal(out[0][0][1], [0])
                np.testing.assert_array_equal(out[1][0][0], [2])

                np.testing.assert_array_equal(out[0][0][0], [1])
                np.testing.assert_array_equal(out[1][0][1], [3])


if __name__ == '__main__':
    tf.test.main()

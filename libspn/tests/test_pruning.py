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


class TestPruning(TestCase):

    def test_pruning_noivs(self):
        """Pruning a network without IVs"""
        def full_network():
            """Build a custom SPN to be pruned
               Figure 1 in https://github.com/pronobis/libspn/wiki/Pruning
            """
            # Input variables
            inp_var_1 = spn.IVs(num_vars=1, num_vals=4, name="Val_Input_1")
            inp_var_2_3 = spn.IVs(num_vars=2, num_vals=4, name="Val_Input_2_3")
            # ContVar
            inp_var_4 = spn.ContVars(num_vars=1, name="Val_Input_4")

            # Layer - 1
            # Decomposition - 1
            s_11 = spn.Sum((inp_var_1), name="S_1.1")
            s_11.generate_weights([0.3, 0.1, 0.3, 0.3])
            s_12 = spn.Sum((inp_var_1, None), name="S_1.2")
            s_12.generate_weights([0.1, 0.4, 0.1, 0.4])

            s_21 = spn.Sum((inp_var_2_3, 0), (inp_var_2_3, 1), (inp_var_2_3, 2),
                           (inp_var_2_3, 3), name="S_2.1")
            s_21.generate_weights([0.25, 0.25, 0.25, 0.25])
            s_22 = spn.Sum((inp_var_2_3, [0, 1]), (inp_var_2_3, [2, 3]),
                           name="S_2.2")
            s_22.generate_weights([0.1, 0.2, 0.3, 0.4])

            # Decomposition - 2
            s_31 = spn.Sum((inp_var_2_3, [4, 5, 6, 7]), name="S_3.1")
            s_31.generate_weights([0.2, 0.2, 0.4, 0.2])
            s_32 = spn.Sum((inp_var_2_3, 4), (inp_var_2_3, [5, 6]),
                           (inp_var_2_3, [7]), name="S_3.2")
            s_32.generate_weights([0.25, 0.25, 0.25, 0.25])

            s_41 = spn.Sum(inp_var_4, (inp_var_4, None),  (inp_var_4, 0),
                           (inp_var_4, [0]), name="S_4.1")
            s_41.generate_weights([0.27, 0.28, 0.29, 0.16])
            s_42 = spn.Sum(inp_var_4, (inp_var_4, None),  (inp_var_4, 0),
                           (inp_var_4, [0]), name="S_4.2")
            s_42.generate_weights([0.1, 0.35, 0.1, 0.45])

            # Layer - 2
            # Decomposition - 1
            p_11 = spn.Product(*[s_11, s_21], name="P_1.1")
            p_12 = spn.Product(*[s_11, s_22], name="P_1.2")
            p_13 = spn.Product(*[s_12, s_21], name="P_1.3")
            p_14 = spn.Product(*[s_12, s_22], name="P_1.4")
            # Decomposition - 2
            p_21 = spn.Product(*[s_31, s_41], name="P_2.1")
            p_22 = spn.Product(*[s_31, s_42], name="P_2.2")
            p_23 = spn.Product(*[s_32, s_41], name="P_2.3")
            p_24 = spn.Product(*[s_32, s_42], name="P_2.4")

            # Layer - 3
            # Decomposition - 1
            s_51 = spn.Sum(*[p_11, p_12, p_13, p_14], name="S_5.1")
            s_51.generate_weights([0.25, 0.25, 0.25, 0.25])
            s_52 = spn.Sum(*[p_11, p_12, p_13, p_14], name="S_5.2")
            s_52.generate_weights([0.1, 0.3, 0.3, 0.3])
            s_53 = spn.Sum(*[p_11, p_12, p_13, p_14], name="S_5.3")
            s_53.generate_weights([0.2, 0.2, 0.4, 0.2])
            # Decomposition - 2
            s_61 = spn.Sum(*[p_21, p_22, p_23, p_24], name="S_6.1")
            s_61.generate_weights([0.4, 0.2, 0.2, 0.4])
            s_62 = spn.Sum(*[p_21, p_22, p_23, p_24], name="S_6.2")
            s_62.generate_weights([0.3, 0.35, 0.3, 0.05])
            s_63 = spn.Sum(*[p_21, p_22, p_23, p_24], name="S_6.3")
            s_63.generate_weights([0.3, 0.3, 0.3, 0.1])

            # Layer - 4
            p_31 = spn.Product(*[s_51, s_61], name="P_3.1")
            p_32 = spn.Product(*[s_51, s_62], name="P_3.2")
            p_33 = spn.Product(*[s_51, s_63], name="P_3.3")
            p_34 = spn.Product(*[s_52, s_61], name="P_3.4")
            p_35 = spn.Product(*[s_52, s_62], name="P_3.5")
            p_36 = spn.Product(*[s_52, s_63], name="P_3.6")
            p_37 = spn.Product(*[s_53, s_61], name="P_3.7")
            p_38 = spn.Product(*[s_53, s_62], name="P_3.8")
            p_39 = spn.Product(*[s_53, s_63], name="P_3.9")

            # Root node
            root = spn.Sum(*[p_31, p_32, p_33,
                             p_34, p_35, p_36,
                             p_37, p_38, p_39], name="root")
            root.generate_weights([0.27, 0.03, 0.03,
                                   0.04, 0.27, 0.03,
                                   0.03, 0.03, 0.27])

            return root, [inp_var_1, inp_var_2_3, inp_var_4]

        def hand_pruned_network():
            """Build a custom SPN representing post-pruning network
               Figure 3 in https://github.com/pronobis/libspn/wiki/Pruning
            """
            # Input variables
            inp_var_1 = spn.IVs(num_vars=1, num_vals=4, name="Val_Input_1")
            inp_var_2_3 = spn.IVs(num_vars=2, num_vals=4, name="Val_Input_2_3")
            # ContVar
            inp_var_4 = spn.ContVars(num_vars=1, name="Val_Input_4")

            # Layer - 1
            # Decomposition - 1
            s_11 = spn.Sum((inp_var_1, 0), (inp_var_1, 2), (inp_var_1, 3),
                           name="S_1.1")
            s_11.generate_weights([0.3, 0.3, 0.3])
            s_12 = spn.Sum((inp_var_1, 1), (inp_var_1, 3), name="S_1.2")
            s_12.generate_weights([0.4, 0.4])

            s_22 = spn.Sum((inp_var_2_3, 2), (inp_var_2_3, 3), name="S_2.2")
            s_22.generate_weights([0.3, 0.4])

            # Decomposition - 2
            s_31 = spn.Sum((inp_var_2_3, 6), name="S_3.1")
            s_31.generate_weights([0.4])

            s_41 = spn.Sum(inp_var_4, inp_var_4, inp_var_4, name="S_4.1")
            s_41.generate_weights([0.27, 0.28, 0.29])
            s_42 = spn.Sum(inp_var_4, inp_var_4, name="S_4.2")
            s_42.generate_weights([0.35, 0.45])

            # Layer - 2
            # Decomposition - 1
            p_12 = spn.Product(*[s_11, s_22], name="P_1.2")
            p_14 = spn.Product(*[s_12, s_22], name="P_1.4")
            # Decomposition - 2
            p_21 = spn.Product(*[s_31, s_41], name="P_2.1")
            p_22 = spn.Product(*[s_31, s_42], name="P_2.2")

            # Layer - 3
            # Decomposition - 1
            s_52 = spn.Sum(*[p_12, p_14], name="S_5.2")
            s_52.generate_weights([0.3, 0.3])
            # Decomposition - 2
            s_62 = spn.Sum(*[p_21, p_22], name="S_6.2")
            s_62.generate_weights([0.3, 0.35])

            # Layer - 4
            p_35 = spn.Product(*[s_52, s_62], name="P_3.5")

            # Root node
            root = spn.Sum(*[p_35], name="root")
            root.generate_weights([0.27])

            return root, [inp_var_1, inp_var_2_3, inp_var_4]

        # Build 2 custom SPNs: (1) A network to be pruned [Full-network]
        #                      (2) A network representing post-pruning [Hand-pruned]
        root_full_network, inps_full_network = full_network()
        root_hand_pruned_network, inps_hand_pruned_network = hand_pruned_network()

        init_weights_full_network = spn.initialize_weights(root_full_network)
        pruning = spn.Pruning(root_full_network)

        # Initialize weights of the Full-network, and save its weight values back
        # in respective ParamNodes.
        with spn.session() as (sess, run):
            sess.run(init_weights_full_network)
            pruning.save_param_values()

        # Performing pruning, and check if the pruned network is valid.
        pruning.prune(threshold=0.26)
        self.assertTrue(root_full_network.is_valid())

        reinit_weights_full_network = spn.initialize_weights(root_full_network)
        init_weights_hand_pruned_network = \
            spn.initialize_weights(root_hand_pruned_network)

        value_full_network = root_full_network.get_value()
        log_value_full_network = root_full_network.get_log_value()
        value_hand_pruned_network = root_hand_pruned_network.get_value()

        # Feed
        values = np.arange(-1, 4)
        points = np.array(np.meshgrid(*[values for i in range(3)])).T
        inps_feed = points.reshape(-1, points.shape[-1])
        inp_4_feed = np.random.rand(inps_feed.shape[0], 1)

        with spn.session() as (sess, run):
            # Initialise weights of pruned and hand-pruned networks
            sess.run(reinit_weights_full_network)
            sess.run(init_weights_hand_pruned_network)

            # Calculate Value and LogValue of pruned and hand-pruned networks.
            full_network_value_out, full_network_log_value_out = \
                sess.run([value_full_network, log_value_full_network],
                         feed_dict={inps_full_network[0]: inps_feed[:, 0:1],
                                    inps_full_network[1]: inps_feed[:, 1:3],
                                    inps_full_network[2]: inp_4_feed})
            hand_pruned_network_value_out = \
                sess.run(value_hand_pruned_network,
                         feed_dict={inps_hand_pruned_network[0]: inps_feed[:, 0:1],
                                    inps_hand_pruned_network[1]: inps_feed[:, 1:3],
                                    inps_hand_pruned_network[2]: inp_4_feed})

        # Assertain if the output of pruned network is consistant with hand-pruned
        # network.
        np.testing.assert_array_almost_equal(full_network_value_out,
                                             hand_pruned_network_value_out)
        np.testing.assert_array_almost_equal(np.exp(full_network_log_value_out),
                                             hand_pruned_network_value_out)

    def test_pruning_ivs(self):
        """Pruning a network with IVs"""
        def full_network():
            """Build a custom SPN to be pruned"""
            # Input variables
            inp_var_1 = spn.IVs(num_vars=1, num_vals=4, name="Val_Input_1")
            inp_var_2_3 = spn.IVs(num_vars=2, num_vals=4, name="Val_Input_2_3")
            # ContVar
            inp_var_4 = spn.ContVars(num_vars=1, name="Val_Input_4")

            # Indicator variables
            ivs_1 = spn.IVs(num_vars=1, num_vals=4, name="IVs_1")
            ivs_3_4 = spn.IVs(num_vars=2, num_vals=4, name="IVs_2")

            # Layer - 1
            # Decomposition - 1
            s_11 = spn.Sum((inp_var_1), name="S_1.1")
            s_11.generate_weights([0.3, 0.1, 0.3, 0.3])
            s_11.set_ivs((ivs_1, [0, 1, 2, 3]))
            s_12 = spn.Sum((inp_var_1, None), name="S_1.2")
            s_12.generate_weights([0.1, 0.4, 0.1, 0.4])
            s_12.set_ivs(ivs_1)

            s_21 = spn.Sum((inp_var_2_3, 0), (inp_var_2_3, 1), (inp_var_2_3, 2),
                           (inp_var_2_3, 3), name="S_2.1")
            s_21.generate_weights([0.25, 0.25, 0.25, 0.25])
            s_22 = spn.Sum((inp_var_2_3, [0, 1]), (inp_var_2_3, [2, 3]),
                           name="S_2.2")
            s_22.generate_weights([0.1, 0.2, 0.3, 0.4])

            # Decomposition - 2
            s_31 = spn.Sum((inp_var_2_3, [4, 5, 6, 7]), name="S_3.1")
            s_31.generate_weights([0.2, 0.2, 0.4, 0.2])
            s_31.set_ivs((ivs_3_4, [0, 1, 2, 3]))
            s_32 = spn.Sum((inp_var_2_3, 4), (inp_var_2_3, [5, 6]),
                           (inp_var_2_3, [7]), name="S_3.2")
            s_32.generate_weights([0.25, 0.25, 0.25, 0.25])
            s_32.set_ivs((ivs_3_4, [0, 1, 2, 3]))

            s_41 = spn.Sum(inp_var_4, (inp_var_4, None),  (inp_var_4, 0),
                           (inp_var_4, [0]), name="S_4.1")
            s_41.generate_weights([0.27, 0.28, 0.29, 0.16])
            s_41.set_ivs((ivs_3_4, [4, 5, 6, 7]))
            s_42 = spn.Sum(inp_var_4, (inp_var_4, None),  (inp_var_4, 0),
                           (inp_var_4, [0]), name="S_4.2")
            s_42.generate_weights([0.1, 0.35, 0.1, 0.45])
            s_42.set_ivs((ivs_3_4, [4, 5, 6, 7]))

            # Layer - 2
            # Decomposition - 1
            p_11 = spn.Product(*[s_11, s_21], name="P_1.1")
            p_12 = spn.Product(*[s_11, s_22], name="P_1.2")
            p_13 = spn.Product(*[s_12, s_21], name="P_1.3")
            p_14 = spn.Product(*[s_12, s_22], name="P_1.4")
            # Decomposition - 2
            p_21 = spn.Product(*[s_31, s_41], name="P_2.1")
            p_22 = spn.Product(*[s_31, s_42], name="P_2.2")
            p_23 = spn.Product(*[s_32, s_41], name="P_2.3")
            p_24 = spn.Product(*[s_32, s_42], name="P_2.4")

            # Layer - 3
            # Decomposition - 1
            s_51 = spn.Sum(*[p_11, p_12, p_13, p_14], name="S_5.1")
            s_51.generate_weights([0.25, 0.25, 0.25, 0.25])
            s_52 = spn.Sum(*[p_11, p_12, p_13, p_14], name="S_5.2")
            s_52.generate_weights([0.1, 0.3, 0.3, 0.3])
            s_53 = spn.Sum(*[p_11, p_12, p_13, p_14], name="S_5.3")
            s_53.generate_weights([0.2, 0.2, 0.4, 0.2])
            # Decomposition - 2
            s_61 = spn.Sum(*[p_21, p_22, p_23, p_24], name="S_6.1")
            s_61.generate_weights([0.4, 0.2, 0.2, 0.4])
            s_62 = spn.Sum(*[p_21, p_22, p_23, p_24], name="S_6.2")
            s_62.generate_weights([0.3, 0.35, 0.3, 0.05])
            s_63 = spn.Sum(*[p_21, p_22, p_23, p_24], name="S_6.3")
            s_63.generate_weights([0.3, 0.3, 0.3, 0.1])

            # Layer - 4
            p_31 = spn.Product(*[s_51, s_61], name="P_3.1")
            p_32 = spn.Product(*[s_51, s_62], name="P_3.2")
            p_33 = spn.Product(*[s_51, s_63], name="P_3.3")
            p_34 = spn.Product(*[s_52, s_61], name="P_3.4")
            p_35 = spn.Product(*[s_52, s_62], name="P_3.5")
            p_36 = spn.Product(*[s_52, s_63], name="P_3.6")
            p_37 = spn.Product(*[s_53, s_61], name="P_3.7")
            p_38 = spn.Product(*[s_53, s_62], name="P_3.8")
            p_39 = spn.Product(*[s_53, s_63], name="P_3.9")

            # Root node
            root = spn.Sum(*[p_31, p_32, p_33,
                             p_34, p_35, p_36,
                             p_37, p_38, p_39], name="root")
            root.generate_weights([0.27, 0.03, 0.03,
                                   0.04, 0.27, 0.03,
                                   0.03, 0.03, 0.27])
            ivs_root = root.generate_ivs()

            return root, [inp_var_1, inp_var_2_3, inp_var_4], \
                [ivs_1, ivs_3_4, ivs_root]

        def hand_pruned_network():
            """Build a custom SPN representing post-pruning network"""
            # Input variables
            inp_var_1 = spn.IVs(num_vars=1, num_vals=4, name="Val_Input_1")
            inp_var_2_3 = spn.IVs(num_vars=2, num_vals=4, name="Val_Input_2_3")
            # ContVar
            inp_var_4 = spn.ContVars(num_vars=1, name="Val_Input_4")

            # Indicator variables
            ivs_1 = spn.IVs(num_vars=1, num_vals=4, name="IVs_1")
            ivs_3_4 = spn.IVs(num_vars=2, num_vals=4, name="IVs_2")

            # Layer - 1
            # Decomposition - 1
            s_11 = spn.Sum((inp_var_1, 0), (inp_var_1, 2), (inp_var_1, 3),
                           name="S_1.1")
            s_11.generate_weights([0.3, 0.3, 0.3])
            s_11.set_ivs((ivs_1, [0, 2, 3]))
            s_12 = spn.Sum((inp_var_1, 1), (inp_var_1, 3), name="S_1.2")
            s_12.generate_weights([0.4, 0.4])
            s_12.set_ivs((ivs_1, [1, 3]))

            s_22 = spn.Sum((inp_var_2_3, 2), (inp_var_2_3, 3), name="S_2.2")
            s_22.generate_weights([0.3, 0.4])

            # Decomposition - 2
            s_31 = spn.Sum((inp_var_2_3, 6), name="S_3.1")
            s_31.generate_weights([0.4])
            s_31.set_ivs((ivs_3_4, 2))

            s_41 = spn.Sum(inp_var_4, inp_var_4, inp_var_4, name="S_4.1")
            s_41.generate_weights([0.27, 0.28, 0.29])
            s_41.set_ivs((ivs_3_4, [4, 5, 6]))
            s_42 = spn.Sum(inp_var_4, inp_var_4, name="S_4.2")
            s_42.generate_weights([0.35, 0.45])
            s_42.set_ivs((ivs_3_4, [5, 7]))

            # Layer - 2
            # Decomposition - 1
            p_12 = spn.Product(*[s_11, s_22], name="P_1.2")
            p_14 = spn.Product(*[s_12, s_22], name="P_1.4")
            # Decomposition - 2
            p_21 = spn.Product(*[s_31, s_41], name="P_2.1")
            p_22 = spn.Product(*[s_31, s_42], name="P_2.2")

            # Layer - 3
            # Decomposition - 1
            s_52 = spn.Sum(*[p_12, p_14], name="S_5.2")
            s_52.generate_weights([0.3, 0.3])
            # Decomposition - 2
            s_62 = spn.Sum(*[p_21, p_22], name="S_6.2")
            s_62.generate_weights([0.3, 0.35])

            # Layer - 4
            p_35 = spn.Product(*[s_52, s_62], name="P_3.5")

            # Root node
            root = spn.Sum(*[p_35], name="root")
            root.generate_weights([0.27])

            return root, [inp_var_1, inp_var_2_3, inp_var_4], [ivs_1, ivs_3_4]

        # Build 2 custom SPNs: (1) A network to be pruned [Full-network]
        #                      (2) A network representing post-pruning [Hand-pruned]
        root_full_network, inps_full_network, ivs_full_network = full_network()
        root_hand_pruned_network, inps_hand_pruned_network, \
            ivs_hand_pruned_network = hand_pruned_network()

        init_weights_full_network = spn.initialize_weights(root_full_network)
        pruning = spn.Pruning(root_full_network)

        # Initialize weights of the Full-network, and save its weight values back
        # in respective ParamNodes.
        with spn.session() as (sess, run):
            sess.run(init_weights_full_network)
            pruning.save_param_values()

        # Performing pruning, and check if the pruned network is valid.
        pruning.prune(threshold=0.26)
        self.assertTrue(root_full_network.is_valid())

        reinit_weights_full_network = spn.initialize_weights(root_full_network)
        init_weights_hand_pruned_network = \
            spn.initialize_weights(root_hand_pruned_network)

        value_full_network = root_full_network.get_value()
        log_value_full_network = root_full_network.get_log_value()
        value_hand_pruned_network = root_hand_pruned_network.get_value()

        # Inputs feed
        values = np.arange(-1, 4)
        points = np.array(np.meshgrid(*[values for i in range(3)])).T
        inps_feed = points.reshape(-1, points.shape[-1])
        inp_4_feed = np.random.rand(inps_feed.shape[0], 1)

        # IVs feed
        ivs_1_feed = np.ones((inps_feed.shape[0], 1)) * -1
        ivs_3_4_feed = np.ones((inps_feed.shape[0], 2)) * -1
        ivs_root_feed = np.ones((inps_feed.shape[0], 1)) * -1

        with spn.session() as (sess, run):
            # Initialise weights of pruned and hand-pruned networks
            sess.run(reinit_weights_full_network)
            sess.run(init_weights_hand_pruned_network)

            # Calculate Value and LogValue of pruned and hand-pruned networks.
            full_network_value_out, full_network_log_value_out = \
                sess.run([value_full_network, log_value_full_network],
                         feed_dict={inps_full_network[0]: inps_feed[:, 0:1],
                                    inps_full_network[1]: inps_feed[:, 1:3],
                                    inps_full_network[2]: inp_4_feed,
                                    ivs_full_network[0]: ivs_1_feed,
                                    ivs_full_network[1]: ivs_3_4_feed,
                                    ivs_full_network[2]: ivs_root_feed})
            hand_pruned_network_value_out = \
                sess.run(value_hand_pruned_network,
                         feed_dict={inps_hand_pruned_network[0]: inps_feed[:, 0:1],
                                    inps_hand_pruned_network[1]: inps_feed[:, 1:3],
                                    inps_hand_pruned_network[2]: inp_4_feed,
                                    ivs_hand_pruned_network[0]: ivs_1_feed,
                                    ivs_hand_pruned_network[1]: ivs_3_4_feed})

        # Assertain if the output of pruned network is consistant with hand-pruned
        # network.
        np.testing.assert_array_almost_equal(full_network_value_out,
                                             hand_pruned_network_value_out)
        np.testing.assert_array_almost_equal(np.exp(full_network_log_value_out),
                                             hand_pruned_network_value_out)


if __name__ == '__main__':
    tf.test.main()

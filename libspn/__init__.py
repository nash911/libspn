# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

# Import public interface of the library

# Graph
from libspn.graph.scope import Scope
from libspn.graph.node import Input
from libspn.graph.node import Node
from libspn.graph.node import OpNode
from libspn.graph.node import VarNode
from libspn.graph.node import ParamNode
from libspn.graph.ivs import IVs
from libspn.graph.contvars import ContVars
from libspn.graph.concat import Concat
from libspn.graph.sum import Sum
from libspn.graph.sums import Sums
from libspn.graph.product import Product
from libspn.graph.weights import Weights
from libspn.graph.weights import assign_weights
from libspn.graph.weights import initialize_weights
from libspn.graph.saver import Saver, JSONSaver
from libspn.graph.loader import Loader, JSONLoader
from libspn.graph.algorithms import compute_graph_up
from libspn.graph.algorithms import compute_graph_up_down
from libspn.graph.algorithms import traverse_graph

# Generators
from libspn.generation.dense import DenseSPNGenerator
from libspn.generation.weights import WeightsGenerator
from libspn.generation.weights import generate_weights
from libspn.generation.test import TestSPNGenerator

# Inference and learning
from libspn.inference.type import InferenceType
from libspn.inference.value import Value
from libspn.inference.value import LogValue
from libspn.inference.mpe_path import MPEPath
from libspn.inference.mpe_state import MPEState
from libspn.learning.em import EMLearning
from libspn.learning.gd import GDLearning

# Data
from libspn.data.dataset import Dataset
from libspn.data.file import FileDataset
from libspn.data.file import CSVFileDataset
from libspn.data.generated import GaussianMixtureDataset
from libspn.data.generated import IntGridDataset
from libspn.data.writer import DataWriter
from libspn.data.writer import CSVDataWriter

# Session
from libspn.session import session

# Visualization
from libspn.visual.plot import plot_2d
from libspn.visual.tf_graph import display_tf_graph
from libspn.visual.spn_graph import display_spn_graph

# Logging
from libspn.log import config_logger
from libspn.log import get_logger
from libspn.log import WARNING
from libspn.log import INFO
from libspn.log import DEBUG1
from libspn.log import DEBUG2

# Utils and config
from libspn import conf
from libspn import utils
from libspn.utils import ValueType

# Exceptions
from libspn.exceptions import StructureError

# All
__all__ = [
    # Graph
    'Scope', 'Input', 'Node', 'ParamNode', 'OpNode', 'VarNode',
    'Concat', 'IVs', 'ContVars', 'Sum', 'Sums', 'Product',
    'Weights', 'assign_weights', 'initialize_weights',
    'Saver', 'Loader', 'JSONSaver', 'JSONLoader',
    'compute_graph_up', 'compute_graph_up_down',
    'traverse_graph',
    # Generators
    'DenseSPNGenerator', 'WeightsGenerator', 'TestSPNGenerator',
    'generate_weights',
    # Inference and learning
    'InferenceType', 'Value', 'LogValue', 'MPEPath', 'MPEState',
    'EMLearning', 'GDLearning',
    # Data
    'Dataset', 'FileDataset', 'CSVFileDataset', 'GaussianMixtureDataset',
    'IntGridDataset', 'DataWriter', 'CSVDataWriter',
    # Session
    'session',
    # Visualization
    'plot_2d', 'display_tf_graph', 'display_spn_graph',
    # Logging
    'config_logger', 'get_logger', 'WARNING', 'INFO', 'DEBUG1', 'DEBUG2',
    # Utils and config
    'conf', 'utils', 'ValueType',
    # Exceptions
    'StructureError']

# Configure the logger to show INFO and WARNING by default
config_logger(level=INFO)

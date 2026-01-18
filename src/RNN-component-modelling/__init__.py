"""
RNN Component Library

A modular collection of RNN components for JAX/Flax that can be imported
into notebooks for experimentation.

Usage:
    from RNN_component_modelling.rnn_cells import VanillaRNNCell
    from RNN_component_modelling.rnn_layers import RNN
    from RNN_component_modelling.rnn_utils import initialize_carry
"""

from .rnn_cells import VanillaRNNCell
from .rnn_layers import RNN
from .rnn_utils import initialize_carry

__all__ = [
    'VanillaRNNCell',
    'RNN',
    'initialize_carry',
]

__version__ = '0.1.0'

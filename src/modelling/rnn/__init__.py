"""
RNN Component Library

A complete, modular collection of RNN components for JAX/Flax.

Components:
- Cells: VanillaRNNCell, LSTMCell, GRUCell
- Layers: RNN, LSTM, GRU, Bidirectional
- Utilities: Padding, masking, initialization

Usage:
    from modelling.rnn import RNN, LSTM, GRU
    from modelling.rnn import VanillaRNNCell, LSTMCell, GRUCell
    from modelling.rnn import Bidirectional
    from modelling.rnn import pad_sequences, create_padding_mask

Example:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from modelling.rnn import LSTM
    
    class SentimentClassifier(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Embed(8000, 128)(x)
            x = LSTM(64, return_sequences=False)(x)
            x = nn.Dense(1)(x)
            return x
    
    model = SentimentClassifier()
    x = jnp.ones((2, 50), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), x)
    logits = model.apply(params, x)

Author: Your Name
Date: 2026-01-18
Version: 1.0.0
"""

# Import cells
from .rnn_cells import (
    VanillaRNNCell,
    LSTMCell,
    GRUCell,
)

# Import layers
from .rnn_layers import (
    RNN,
    LSTM,
    GRU,
    Bidirectional,
)

# Import utilities
from .rnn_utils import (
    initialize_carry,
    initialize_lstm_carry,
    pad_sequences,
    create_padding_mask,
    create_causal_mask,
    reverse_sequences,
    get_sequence_lengths,
)

__all__ = [
    # Cells
    'VanillaRNNCell',
    'LSTMCell',
    'GRUCell',
    # Layers
    'RNN',
    'LSTM',
    'GRU',
    'Bidirectional',
    # Utilities
    'initialize_carry',
    'initialize_lstm_carry',
    'pad_sequences',
    'create_padding_mask',
    'create_causal_mask',
    'reverse_sequences',
    'get_sequence_lengths',
]

__version__ = '1.0.0'

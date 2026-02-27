"""
RNN Layers - High-level sequence processing

This module contains RNN layer implementations that process entire sequences
using the cells from rnn_cells.py with JAX's efficient lax.scan.

Layers:
- RNN: Vanilla RNN layer
- LSTM: LSTM layer
- GRU: GRU layer

Author: Your Name
Date: 2026-01-18
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Optional, Tuple

from .rnn_cells import VanillaRNNCell, LSTMCell, GRUCell


# ============================================================================
# VANILLA RNN LAYER
# ============================================================================

class RNN(nn.Module):
    """
    Vanilla RNN Layer
    
    Processes entire sequences using VanillaRNNCell and lax.scan.
    
    Attributes:
        hidden_size (int): Dimension of hidden state
        return_sequences (bool): Return all hidden states or just final
    
    Example:
        >>> rnn = RNN(hidden_size=64, return_sequences=False)
        >>> x = jnp.ones((32, 50, 128))  # (batch, seq_len, input_dim)
        >>> params = rnn.init(jax.random.PRNGKey(0), x)
        >>> output = rnn.apply(params, x)  # (32, 64)
    """
    
    hidden_size: int
    return_sequences: bool = False
    
    def setup(self):
        """Initialize the RNN cell"""
        self.cell = VanillaRNNCell(self.hidden_size)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Process a batch of sequences.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
        
        Returns:
            If return_sequences=True: (batch, seq_len, hidden_size)
            If return_sequences=False: (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state to zeros
        h_0 = jnp.zeros((batch_size, self.hidden_size))
        
        # Transpose for lax.scan: (seq_len, batch, input_dim)
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        # Apply RNN cell across all timesteps
        # lax.scan signature: scan(f, init, xs)
        # Returns: (final_carry, all_outputs)
        final_carry, all_outputs = lax.scan(
            self.cell,
            init=h_0,
            xs=x_transposed
        )
        
        if self.return_sequences:
            # Transpose back to (batch, seq_len, hidden_size)
            return jnp.transpose(all_outputs, (1, 0, 2))
        else:
            # Return only final hidden state
            return final_carry


# ============================================================================
# LSTM LAYER
# ============================================================================

class LSTM(nn.Module):
    """
    LSTM Layer
    
    Processes entire sequences using LSTMCell and lax.scan.
    
    LSTM maintains two states:
    - Hidden state (h): Short-term memory
    - Cell state (c): Long-term memory
    
    Attributes:
        hidden_size (int): Dimension of hidden and cell states
        return_sequences (bool): Return all hidden states or just final
    
    Example:
        >>> lstm = LSTM(hidden_size=64, return_sequences=False)
        >>> x = jnp.ones((32, 50, 128))  # (batch, seq_len, input_dim)
        >>> params = lstm.init(jax.random.PRNGKey(0), x)
        >>> output = lstm.apply(params, x)  # (32, 64)
    """
    
    hidden_size: int
    return_sequences: bool = False
    
    def setup(self):
        """Initialize the LSTM cell"""
        self.cell = LSTMCell(self.hidden_size)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Process a batch of sequences.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
        
        Returns:
            If return_sequences=True: (batch, seq_len, hidden_size)
            If return_sequences=False: (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states to zeros
        h_0 = jnp.zeros((batch_size, self.hidden_size))
        c_0 = jnp.zeros((batch_size, self.hidden_size))
        
        # Transpose for lax.scan
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        # Apply LSTM cell across all timesteps
        # Note: carry is (h, c) tuple for LSTM
        (final_h, final_c), all_outputs = lax.scan(
            self.cell,
            init=(h_0, c_0),
            xs=x_transposed
        )
        
        if self.return_sequences:
            # Transpose back to (batch, seq_len, hidden_size)
            return jnp.transpose(all_outputs, (1, 0, 2))
        else:
            # Return only final hidden state (not cell state)
            return final_h


# ============================================================================
# GRU LAYER
# ============================================================================

class GRU(nn.Module):
    """
    GRU Layer
    
    Processes entire sequences using GRUCell and lax.scan.
    
    GRU is a simplified LSTM with:
    - Only hidden state (no separate cell state)
    - Fewer parameters, faster training
    
    Attributes:
        hidden_size (int): Dimension of hidden state
        return_sequences (bool): Return all hidden states or just final
    
    Example:
        >>> gru = GRU(hidden_size=64, return_sequences=False)
        >>> x = jnp.ones((32, 50, 128))  # (batch, seq_len, input_dim)
        >>> params = gru.init(jax.random.PRNGKey(0), x)
        >>> output = gru.apply(params, x)  # (32, 64)
    """
    
    hidden_size: int
    return_sequences: bool = False
    
    def setup(self):
        """Initialize the GRU cell"""
        self.cell = GRUCell(self.hidden_size)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Process a batch of sequences.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
        
        Returns:
            If return_sequences=True: (batch, seq_len, hidden_size)
            If return_sequences=False: (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state to zeros
        h_0 = jnp.zeros((batch_size, self.hidden_size))
        
        # Transpose for lax.scan
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        # Apply GRU cell across all timesteps
        final_carry, all_outputs = lax.scan(
            self.cell,
            init=h_0,
            xs=x_transposed
        )
        
        if self.return_sequences:
            # Transpose back to (batch, seq_len, hidden_size)
            return jnp.transpose(all_outputs, (1, 0, 2))
        else:
            # Return only final hidden state
            return final_carry


# ============================================================================
# BIDIRECTIONAL WRAPPER
# ============================================================================

class Bidirectional(nn.Module):
    """
    Bidirectional RNN Wrapper
    
    Processes sequences in both forward and backward directions,
    then concatenates the outputs.
    
    Useful for tasks where future context helps (e.g., NER, POS tagging).
    
    Attributes:
        layer: RNN layer to wrap (RNN, LSTM, or GRU)
        merge_mode: How to combine forward and backward outputs
            - 'concat': Concatenate (output_size = 2 * hidden_size)
            - 'sum': Element-wise sum (output_size = hidden_size)
            - 'avg': Element-wise average (output_size = hidden_size)
    
    Example:
        >>> base_lstm = LSTM(hidden_size=64, return_sequences=True)
        >>> bi_lstm = Bidirectional(base_lstm, merge_mode='concat')
        >>> x = jnp.ones((32, 50, 128))
        >>> params = bi_lstm.init(jax.random.PRNGKey(0), x)
        >>> output = bi_lstm.apply(params, x)  # (32, 50, 128) - doubled!
    """
    
    layer: nn.Module
    merge_mode: str = 'concat'  # 'concat', 'sum', 'avg'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Process sequence bidirectionally.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
        
        Returns:
            Merged outputs (batch, seq_len, output_dim)
            output_dim depends on merge_mode
        """
        # Forward pass
        forward_out = self.layer(x)
        
        # Backward pass (reverse sequence)
        x_reversed = jnp.flip(x, axis=1)
        backward_out = self.layer(x_reversed)
        
        # Reverse backward output to align with forward
        backward_out = jnp.flip(backward_out, axis=1)
        
        # Merge outputs
        if self.merge_mode == 'concat':
            return jnp.concatenate([forward_out, backward_out], axis=-1)
        elif self.merge_mode == 'sum':
            return forward_out + backward_out
        elif self.merge_mode == 'avg':
            return (forward_out + backward_out) / 2
        else:
            raise ValueError(f"Unknown merge_mode: {self.merge_mode}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Simple RNN for Sentiment Classification
---------------------------------------------------
import jax
import jax.numpy as jnp
from flax import linen as nn
from modelling.rnn.rnn_layers import RNN

class SentimentRNN(nn.Module):
    vocab_size: int = 8000
    embed_dim: int = 128
    hidden_size: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len) - token IDs
        x = nn.Embed(self.vocab_size, self.embed_dim)(x)
        x = RNN(self.hidden_size, return_sequences=False)(x)
        x = nn.Dense(1)(x)
        return x

model = SentimentRNN()
x = jnp.ones((2, 50), dtype=jnp.int32)
params = model.init(jax.random.PRNGKey(0), x)
logits = model.apply(params, x)
print(logits.shape)  # (2, 1)


Example 2: Stacked LSTM
-----------------------
from modelling.rnn.rnn_layers import LSTM

class StackedLSTM(nn.Module):
    hidden_sizes: list = [64, 32]
    
    @nn.compact
    def __call__(self, x):
        # First LSTM: return all sequences for next layer
        x = LSTM(self.hidden_sizes[0], return_sequences=True)(x)
        
        # Second LSTM: return final state only
        x = LSTM(self.hidden_sizes[1], return_sequences=False)(x)
        
        return x

model = StackedLSTM()
x = jnp.ones((2, 50, 128))
params = model.init(jax.random.PRNGKey(0), x)
output = model.apply(params, x)
print(output.shape)  # (2, 32)


Example 3: Bidirectional GRU
-----------------------------
from modelling.rnn.rnn_layers import GRU, Bidirectional

gru = GRU(hidden_size=64, return_sequences=True)
bi_gru = Bidirectional(gru, merge_mode='concat')

x = jnp.ones((2, 50, 128))
params = bi_gru.init(jax.random.PRNGKey(0), x)
output = bi_gru.apply(params, x)
print(output.shape)  # (2, 50, 128) - 64*2 from concat
"""

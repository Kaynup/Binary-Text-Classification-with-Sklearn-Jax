"""
RNN Layers - High-level sequence processing

This module contains RNN layers that use the cells from rnn_cells.py
to process entire sequences efficiently.

Author: Your Name
Date: 2026-01-18
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Optional

from .rnn_cells import VanillaRNNCell


class RNN(nn.Module):
    """
    RNN Layer - Processes entire sequences using RNN cell
    
    This layer wraps the VanillaRNNCell and applies it across all timesteps
    in a sequence using JAX's efficient lax.scan operation.
    
    Attributes:
        hidden_size (int): Dimension of hidden state
        return_sequences (bool): If True, return all hidden states.
                                 If False, return only final hidden state.
    
    Example:
        >>> rnn = RNN(hidden_size=64, return_sequences=False)
        >>> x = jnp.ones((32, 50, 128))  # (batch, seq_len, input_dim)
        >>> 
        >>> params = rnn.init(jax.random.PRNGKey(0), x)
        >>> output = rnn.apply(params, x)
        >>> 
        >>> # output shape: (32, 64) if return_sequences=False
        >>> # output shape: (32, 50, 64) if return_sequences=True
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
            x: Input sequences, shape (batch, seq_len, input_dim)
        
        Returns:
            If return_sequences=True: (batch, seq_len, hidden_size)
            If return_sequences=False: (batch, hidden_size)
        
        How it works:
            1. Initialize hidden state to zeros
            2. Use lax.scan to apply cell to each timestep
            3. Return all states or just final state
        """
        
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # TODO: Initialize hidden state
        # GUIDANCE: Create a zero tensor of shape (batch_size, hidden_size)
        h_0 = None  # Replace with jnp.zeros((batch_size, self.hidden_size))
        
        # TODO: Transpose for lax.scan
        # GUIDANCE: lax.scan expects (seq_len, batch, features)
        # Current x shape: (batch, seq_len, input_dim)
        # Need to transpose to: (seq_len, batch, input_dim)
        x_transposed = None  # Replace with jnp.transpose(x, (1, 0, 2))
        
        # TODO: Apply RNN cell across all timesteps using lax.scan
        # GUIDANCE:
        # lax.scan signature: scan(f, init, xs)
        # - f: function with signature (carry, x) -> (new_carry, output)
        # - init: initial carry (h_0)
        # - xs: sequence to scan over (x_transposed)
        # Returns: (final_carry, all_outputs)
        
        final_carry, all_outputs = None, None  # Replace with lax.scan(...)
        
        # TODO: Return based on return_sequences flag
        if self.return_sequences:
            # Need to transpose back to (batch, seq_len, hidden_size)
            return None  # Replace with jnp.transpose(all_outputs, (1, 0, 2))
        else:
            # Return only final hidden state
            return None  # Replace with final_carry


# ============================================================================
# EXERCISE FOR YOU:
# ============================================================================
#
# 1. Complete the TODOs above to implement the RNN layer
#
# 2. Test your implementation in a notebook:
#    ```python
#    import jax
#    import jax.numpy as jnp
#    from RNN_component_modelling.rnn_layers import RNN
#    
#    # Create RNN layer
#    rnn = RNN(hidden_size=64, return_sequences=False)
#    
#    # Sample input: batch_size=2, seq_len=10, input_dim=128
#    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 128))
#    
#    # Initialize
#    params = rnn.init(jax.random.PRNGKey(1), x)
#    
#    # Forward pass
#    output = rnn.apply(params, x)
#    
#    print(f"Input shape: {x.shape}")
#    print(f"Output shape: {output.shape}")
#    
#    # Try with return_sequences=True
#    rnn_seq = RNN(hidden_size=64, return_sequences=True)
#    params_seq = rnn_seq.init(jax.random.PRNGKey(1), x)
#    output_seq = rnn_seq.apply(params_seq, x)
#    print(f"Output (all sequences) shape: {output_seq.shape}")
#    ```
#
# 3. Questions to think about:
#    - Why do we need to transpose the input for lax.scan?
#    - What's the difference between final_carry and all_outputs?
#    - When would you use return_sequences=True vs False?
#
# 4. Advanced challenge:
#    - How would you stack multiple RNN layers?
#    - Hint: Use return_sequences=True for all but the last layer
#
# ============================================================================

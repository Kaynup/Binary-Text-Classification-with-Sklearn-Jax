"""
RNN Cells - Core building blocks for recurrent neural networks

This module contains the basic RNN cell implementation that processes
one timestep at a time.

Author: Your Name
Date: 2026-01-18
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any

# Type aliases for clarity
Carry = jnp.ndarray  # Hidden state
Input = jnp.ndarray  # Current input
Output = jnp.ndarray  # Cell output


class VanillaRNNCell(nn.Module):
    """
    Basic RNN Cell (Elman RNN)
    
    Implements the recurrence relation:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    
    This is the SIMPLEST form of RNN. It processes one timestep at a time
    and maintains a hidden state that acts as memory.
    
    Attributes:
        hidden_size (int): Dimension of the hidden state
    
    Example:
        >>> cell = VanillaRNNCell(hidden_size=64)
        >>> batch_size, input_dim = 32, 128
        >>> 
        >>> # Initialize
        >>> h_0 = jnp.zeros((batch_size, 64))  # Initial hidden state
        >>> x_t = jnp.ones((batch_size, input_dim))  # Current input
        >>> 
        >>> # Single timestep
        >>> params = cell.init(jax.random.PRNGKey(0), h_0, x_t)
        >>> h_1, output = cell.apply(params, h_0, x_t)
    """
    
    hidden_size: int
    
    @nn.compact
    def __call__(self, carry: Carry, x: Input) -> Tuple[Carry, Output]:
        """
        Process one timestep of the sequence.
        
        Args:
            carry: Previous hidden state, shape (batch, hidden_size)
            x: Current input, shape (batch, input_dim)
        
        Returns:
            new_carry: Updated hidden state, shape (batch, hidden_size)
            output: Same as new_carry for vanilla RNN
        
        Mathematical Operation:
            h_new = tanh(Dense(x) + Dense(h_prev))
        
        Why this signature?
            JAX's lax.scan expects functions with signature:
            (carry, x) -> (new_carry, output)
            This allows efficient sequential processing!
        """
        
        # TODO: Implement the RNN cell logic here
        # 
        # GUIDANCE:
        # 1. Extract previous hidden state from carry
        # 2. Apply linear transformation to input: W_ih @ x
        # 3. Apply linear transformation to hidden: W_hh @ h_prev
        # 4. Add them together (with bias automatically included by Dense)
        # 5. Apply tanh activation
        # 6. Return (new_hidden, new_hidden) - output = hidden for vanilla RNN
        #
        # HINT: Use nn.Dense() for linear transformations
        # HINT: Use nn.tanh() for activation
        
        h_prev = carry
        
        # Step 1: Transform input (W_ih @ x + b_ih)
        # TODO: Add Dense layer for input
        input_contribution = None  # Replace with nn.Dense(self.hidden_size)(x)
        
        # Step 2: Transform previous hidden state (W_hh @ h_prev + b_hh)
        # TODO: Add Dense layer for hidden state
        hidden_contribution = None  # Replace with nn.Dense(self.hidden_size)(h_prev)
        
        # Step 3: Combine and activate
        # TODO: Add contributions and apply tanh
        h_new = None  # Replace with nn.tanh(input_contribution + hidden_contribution)
        
        # Step 4: Return new state and output
        # For vanilla RNN, output = hidden state
        return h_new, h_new


# ============================================================================
# EXERCISE FOR YOU:
# ============================================================================
# 
# 1. Complete the TODOs above to implement the RNN cell
# 
# 2. Test your implementation in a notebook:
#    ```python
#    import jax
#    import jax.numpy as jnp
#    from RNN_component_modelling.rnn_cells import VanillaRNNCell
#    
#    # Create cell
#    cell = VanillaRNNCell(hidden_size=64)
#    
#    # Initialize
#    h_0 = jnp.zeros((2, 64))  # batch_size=2
#    x_t = jnp.ones((2, 128))   # input_dim=128
#    
#    params = cell.init(jax.random.PRNGKey(0), h_0, x_t)
#    
#    # Forward pass
#    h_1, out = cell.apply(params, h_0, x_t)
#    
#    print(f"Input shape: {x_t.shape}")
#    print(f"Hidden shape: {h_1.shape}")
#    print(f"Output shape: {out.shape}")
#    ```
#
# 3. Questions to think about:
#    - Why do we need TWO Dense layers (one for input, one for hidden)?
#    - What would happen if we used ReLU instead of tanh?
#    - Why is the output the same as the hidden state?
#
# ============================================================================

"""
RNN Cells - Core building blocks for recurrent neural networks

This module contains implementations of three RNN cell types:
- VanillaRNNCell: Basic Elman RNN
- LSTMCell: Long Short-Term Memory (solves vanishing gradients)
- GRUCell: Gated Recurrent Unit (simplified LSTM)

Author: Your Name
Date: 2026-01-18
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Union

# Type aliases for clarity
Carry = Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]  # Hidden state (or (h, c) for LSTM)
Input = jnp.ndarray  # Current input
Output = jnp.ndarray  # Cell output


# ============================================================================
# VANILLA RNN CELL
# ============================================================================

class VanillaRNNCell(nn.Module):
    """
    Basic RNN Cell (Elman RNN)
    
    Formula:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    
    Pros:
        - Simple and fast
        - Good for short sequences
    
    Cons:
        - Vanishing gradient problem
        - Can't capture long-term dependencies
    
    Attributes:
        hidden_size (int): Dimension of the hidden state
    """
    
    hidden_size: int
    
    @nn.compact
    def __call__(self, carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process one timestep.
        
        Args:
            carry: Previous hidden state (batch, hidden_size)
            x: Current input (batch, input_dim)
        
        Returns:
            new_carry: Updated hidden state
            output: Same as new_carry
        """
        h_prev = carry
        
        # Linear transformations
        # W_ih @ x_t + b_ih
        input_contribution = nn.Dense(self.hidden_size, name='input_transform')(x)
        
        # W_hh @ h_{t-1} + b_hh
        hidden_contribution = nn.Dense(self.hidden_size, use_bias=False, name='hidden_transform')(h_prev)
        
        # Combine and activate
        h_new = nn.tanh(input_contribution + hidden_contribution)
        
        return h_new, h_new


# ============================================================================
# LSTM CELL
# ============================================================================

class LSTMCell(nn.Module):
    """
    Long Short-Term Memory Cell
    
    LSTM solves the vanishing gradient problem using:
    - Cell state (c_t): Long-term memory highway
    - Gates: Control information flow
    
    Gates:
        1. Forget gate (f_t): What to forget from cell state
        2. Input gate (i_t): What new info to add to cell state
        3. Output gate (o_t): What to output from cell state
    
    Formulas:
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)  # Forget gate
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)  # Input gate
        g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)     # Candidate values
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)  # Output gate
        
        c_t = f_t * c_{t-1} + i_t * g_t            # Update cell state
        h_t = o_t * tanh(c_t)                       # Update hidden state
    
    Pros:
        - Handles long-term dependencies
        - Mitigates vanishing gradients
    
    Cons:
        - More parameters (4x vanilla RNN)
        - Slower to train
    
    Attributes:
        hidden_size (int): Dimension of hidden and cell states
    """
    
    hidden_size: int
    
    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Process one timestep.
        
        Args:
            carry: Tuple of (h_prev, c_prev)
                h_prev: Previous hidden state (batch, hidden_size)
                c_prev: Previous cell state (batch, hidden_size)
            x: Current input (batch, input_dim)
        
        Returns:
            new_carry: Tuple of (h_new, c_new)
            output: h_new (hidden state)
        """
        h_prev, c_prev = carry
        
        # Concatenate hidden and input for all gates
        # Shape: (batch, hidden_size + input_dim)
        combined = jnp.concatenate([h_prev, x], axis=-1)
        
        # All four gates computed together for efficiency
        # Output shape: (batch, 4 * hidden_size)
        gates = nn.Dense(4 * self.hidden_size, name='gates')(combined)
        
        # Split into individual gates
        # Each has shape: (batch, hidden_size)
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        
        # Apply activations
        i = nn.sigmoid(i)  # Input gate: what to add
        f = nn.sigmoid(f)  # Forget gate: what to keep
        g = nn.tanh(g)     # Candidate values: what values to add
        o = nn.sigmoid(o)  # Output gate: what to output
        
        # Update cell state (long-term memory)
        # c_t = f_t * c_{t-1} + i_t * g_t
        c_new = f * c_prev + i * g
        
        # Update hidden state (short-term memory)
        # h_t = o_t * tanh(c_t)
        h_new = o * nn.tanh(c_new)
        
        return (h_new, c_new), h_new


# ============================================================================
# GRU CELL
# ============================================================================

class GRUCell(nn.Module):
    """
    Gated Recurrent Unit Cell
    
    GRU is a simplified version of LSTM with:
    - Only 2 gates (vs 3 in LSTM)
    - No separate cell state
    - Fewer parameters, faster training
    
    Gates:
        1. Reset gate (r_t): How much past info to forget
        2. Update gate (z_t): How much to update hidden state
    
    Formulas:
        r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)  # Reset gate
        z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)  # Update gate
        
        h_tilde = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # Candidate
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde         # New hidden
    
    Intuition:
        - z_t controls how much to keep old vs new
        - r_t controls how much past info to use for candidate
    
    Pros:
        - Fewer parameters than LSTM
        - Often performs similarly to LSTM
        - Faster training
    
    Cons:
        - Slightly less expressive than LSTM
    
    Attributes:
        hidden_size (int): Dimension of the hidden state
    """
    
    hidden_size: int
    
    @nn.compact
    def __call__(self, carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process one timestep.
        
        Args:
            carry: Previous hidden state (batch, hidden_size)
            x: Current input (batch, input_dim)
        
        Returns:
            new_carry: Updated hidden state
            output: Same as new_carry
        """
        h_prev = carry
        
        # Concatenate for gates
        combined = jnp.concatenate([h_prev, x], axis=-1)
        
        # Compute reset and update gates together
        # Output shape: (batch, 2 * hidden_size)
        gates = nn.Dense(2 * self.hidden_size, name='gates')(combined)
        
        # Split into reset and update gates
        r, z = jnp.split(gates, 2, axis=-1)
        
        # Apply sigmoid activation
        r = nn.sigmoid(r)  # Reset gate
        z = nn.sigmoid(z)  # Update gate
        
        # Compute candidate hidden state
        # Use reset gate to control how much past info to use
        reset_hidden = r * h_prev
        candidate_input = jnp.concatenate([reset_hidden, x], axis=-1)
        h_tilde = nn.tanh(nn.Dense(self.hidden_size, name='candidate')(candidate_input))
        
        # Update hidden state
        # z controls interpolation between old and new
        h_new = (1 - z) * h_prev + z * h_tilde
        
        return h_new, h_new


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Vanilla RNN Cell
----------------------------
import jax
import jax.numpy as jnp
from modelling.rnn.rnn_cells import VanillaRNNCell

cell = VanillaRNNCell(hidden_size=64)
h_0 = jnp.zeros((2, 64))  # batch_size=2
x_t = jnp.ones((2, 128))  # input_dim=128

params = cell.init(jax.random.PRNGKey(0), h_0, x_t)
h_1, out = cell.apply(params, h_0, x_t)

print(f"Hidden: {h_1.shape}")  # (2, 64)


Example 2: LSTM Cell
--------------------
from modelling.rnn.rnn_cells import LSTMCell

cell = LSTMCell(hidden_size=64)
h_0 = jnp.zeros((2, 64))
c_0 = jnp.zeros((2, 64))  # Cell state
x_t = jnp.ones((2, 128))

params = cell.init(jax.random.PRNGKey(0), (h_0, c_0), x_t)
(h_1, c_1), out = cell.apply(params, (h_0, c_0), x_t)

print(f"Hidden: {h_1.shape}, Cell: {c_1.shape}")  # (2, 64), (2, 64)


Example 3: GRU Cell
-------------------
from modelling.rnn.rnn_cells import GRUCell

cell = GRUCell(hidden_size=64)
h_0 = jnp.zeros((2, 64))
x_t = jnp.ones((2, 128))

params = cell.init(jax.random.PRNGKey(0), h_0, x_t)
h_1, out = cell.apply(params, h_0, x_t)

print(f"Hidden: {h_1.shape}")  # (2, 64)
"""

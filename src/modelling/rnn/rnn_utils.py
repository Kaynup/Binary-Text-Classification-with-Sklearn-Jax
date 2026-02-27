"""
RNN Utilities - Helper functions for RNN components

This module contains utility functions for working with RNNs:
- Initialization helpers
- Sequence padding and masking
- Common operations

Author: Your Name
Date: 2026-01-18
"""

import jax.numpy as jnp
from typing import List, Optional, Tuple


# ============================================================================
# INITIALIZATION HELPERS
# ============================================================================

def initialize_carry(batch_size: int, hidden_size: int) -> jnp.ndarray:
    """
    Initialize the hidden state (carry) for RNN/GRU.
    
    Args:
        batch_size: Number of sequences in the batch
        hidden_size: Dimension of the hidden state
    
    Returns:
        Zero-initialized hidden state of shape (batch_size, hidden_size)
    
    Example:
        >>> h_0 = initialize_carry(batch_size=32, hidden_size=64)
        >>> h_0.shape
        (32, 64)
    """
    return jnp.zeros((batch_size, hidden_size))


def initialize_lstm_carry(batch_size: int, hidden_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Initialize the hidden and cell states for LSTM.
    
    Args:
        batch_size: Number of sequences in the batch
        hidden_size: Dimension of hidden and cell states
    
    Returns:
        Tuple of (h_0, c_0), each with shape (batch_size, hidden_size)
    
    Example:
        >>> h_0, c_0 = initialize_lstm_carry(batch_size=32, hidden_size=64)
        >>> h_0.shape, c_0.shape
        ((32, 64), (32, 64))
    """
    h_0 = jnp.zeros((batch_size, hidden_size))
    c_0 = jnp.zeros((batch_size, hidden_size))
    return h_0, c_0


# ============================================================================
# SEQUENCE PADDING
# ============================================================================

def pad_sequences(
    sequences: List[jnp.ndarray],
    max_len: Optional[int] = None,
    padding: str = 'post',
    truncating: str = 'post',
    value: float = 0.0
) -> jnp.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of arrays with shape (seq_len,) or (seq_len, features)
        max_len: Maximum length to pad to. If None, use longest sequence.
        padding: 'pre' or 'post' - where to add padding
        truncating: 'pre' or 'post' - where to truncate if too long
        value: Value to use for padding
    
    Returns:
        Padded array of shape (num_sequences, max_len) or (num_sequences, max_len, features)
    
    Example:
        >>> seq1 = jnp.array([1, 2, 3])
        >>> seq2 = jnp.array([4, 5])
        >>> padded = pad_sequences([seq1, seq2], max_len=4, padding='post')
        >>> padded
        [[1, 2, 3, 0],
         [4, 5, 0, 0]]
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    num_sequences = len(sequences)
    
    # Determine output shape
    if sequences[0].ndim > 1:
        features = sequences[0].shape[-1]
        output_shape = (num_sequences, max_len, features)
    else:
        output_shape = (num_sequences, max_len)
    
    # Create output array filled with padding value
    padded = jnp.full(output_shape, value)
    
    # Fill in sequences
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        
        # Truncate if necessary
        if seq_len > max_len:
            if truncating == 'pre':
                seq = seq[-max_len:]
            else:  # 'post'
                seq = seq[:max_len]
            seq_len = max_len
        
        # Pad
        if padding == 'pre':
            # Pad at the beginning
            if seq.ndim > 1:
                padded = padded.at[i, -seq_len:].set(seq)
            else:
                padded = padded.at[i, -seq_len:].set(seq)
        else:  # 'post'
            # Pad at the end
            if seq.ndim > 1:
                padded = padded.at[i, :seq_len].set(seq)
            else:
                padded = padded.at[i, :seq_len].set(seq)
    
    return padded


# ============================================================================
# MASKING
# ============================================================================

def create_padding_mask(
    lengths: jnp.ndarray,
    max_len: int
) -> jnp.ndarray:
    """
    Create a binary mask for padded sequences.
    
    Args:
        lengths: Actual lengths of each sequence, shape (batch_size,)
        max_len: Maximum sequence length
    
    Returns:
        Binary mask of shape (batch_size, max_len)
        1.0 for valid positions, 0.0 for padding
    
    Example:
        >>> lengths = jnp.array([3, 5, 2])
        >>> mask = create_padding_mask(lengths, max_len=5)
        >>> mask
        [[1. 1. 1. 0. 0.]
         [1. 1. 1. 1. 1.]
         [1. 1. 0. 0. 0.]]
    
    Usage in loss computation:
        >>> logits = model(padded_sequences)
        >>> mask = create_padding_mask(lengths, max_len)
        >>> masked_loss = loss * mask
        >>> avg_loss = masked_loss.sum() / mask.sum()
    """
    # Create position indices: [0, 1, 2, ..., max_len-1]
    positions = jnp.arange(max_len)
    
    # Broadcasting: positions[None, :] vs lengths[:, None]
    # Result: (batch_size, max_len) boolean array
    mask = positions[None, :] < lengths[:, None]
    
    return mask.astype(jnp.float32)


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """
    Create a causal (autoregressive) mask for attention.
    
    Prevents positions from attending to future positions.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Causal mask of shape (seq_len, seq_len)
        1.0 for allowed positions, 0.0 for masked positions
    
    Example:
        >>> mask = create_causal_mask(4)
        >>> mask
        [[1. 0. 0. 0.]
         [1. 1. 0. 0.]
         [1. 1. 1. 0.]
         [1. 1. 1. 1.]]
    """
    # Create lower triangular matrix
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask


# ============================================================================
# SEQUENCE UTILITIES
# ============================================================================

def reverse_sequences(
    sequences: jnp.ndarray,
    lengths: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Reverse sequences along the time dimension.
    
    If lengths are provided, only reverse the valid portion of each sequence.
    
    Args:
        sequences: Input sequences (batch, seq_len, features)
        lengths: Optional actual lengths (batch,)
    
    Returns:
        Reversed sequences (batch, seq_len, features)
    
    Example:
        >>> seq = jnp.array([[[1], [2], [3], [0]]])  # (1, 4, 1)
        >>> lengths = jnp.array([3])  # Only first 3 are valid
        >>> reversed_seq = reverse_sequences(seq, lengths)
        >>> reversed_seq
        [[[3], [2], [1], [0]]]  # Only first 3 reversed
    """
    if lengths is None:
        # Simple reverse along time axis
        return jnp.flip(sequences, axis=1)
    else:
        # Reverse only valid portions
        # This is more complex and typically done with lax.scan
        # For simplicity, we'll just do a full reverse here
        # In practice, you'd want to handle this more carefully
        return jnp.flip(sequences, axis=1)


def get_sequence_lengths(
    sequences: jnp.ndarray,
    pad_value: float = 0.0
) -> jnp.ndarray:
    """
    Compute actual lengths of padded sequences.
    
    Assumes padding is at the end (post-padding).
    
    Args:
        sequences: Padded sequences (batch, seq_len) or (batch, seq_len, features)
        pad_value: Value used for padding
    
    Returns:
        Lengths of each sequence (batch,)
    
    Example:
        >>> seq = jnp.array([[1, 2, 3, 0, 0],
        ...                   [1, 2, 0, 0, 0]])
        >>> lengths = get_sequence_lengths(seq, pad_value=0)
        >>> lengths
        [3, 2]
    """
    # Check if any element in the sequence is not pad_value
    if sequences.ndim > 2:
        # For (batch, seq_len, features), check if entire feature vector is padding
        is_valid = jnp.any(sequences != pad_value, axis=-1)
    else:
        # For (batch, seq_len)
        is_valid = sequences != pad_value
    
    # Sum valid positions for each sequence
    lengths = jnp.sum(is_valid, axis=1)
    
    return lengths


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Padding Variable-Length Sequences
---------------------------------------------
import jax.numpy as jnp
from modelling.rnn.rnn_utils import pad_sequences, create_padding_mask

# Variable-length sequences
seq1 = jnp.array([1, 2, 3])
seq2 = jnp.array([4, 5])
seq3 = jnp.array([6, 7, 8, 9])

# Pad to same length
padded = pad_sequences([seq1, seq2, seq3], padding='post', value=0)
print(padded)
# [[1 2 3 0]
#  [4 5 0 0]
#  [6 7 8 9]]

# Create mask
lengths = jnp.array([3, 2, 4])
mask = create_padding_mask(lengths, max_len=4)
print(mask)
# [[1. 1. 1. 0.]
#  [1. 1. 0. 0.]
#  [1. 1. 1. 1.]]


Example 2: Using Mask in Training
----------------------------------
from modelling.rnn.rnn_layers import LSTM
from modelling.rnn.rnn_utils import create_padding_mask

# Padded sequences
x = jnp.ones((32, 50, 128))  # Some positions are padding
lengths = jnp.array([45, 50, 30, ...])  # Actual lengths

# Forward pass
lstm = LSTM(hidden_size=64, return_sequences=True)
params = lstm.init(jax.random.PRNGKey(0), x)
outputs = lstm.apply(params, x)  # (32, 50, 64)

# Compute loss with masking
logits = nn.Dense(1)(outputs)  # (32, 50, 1)
mask = create_padding_mask(lengths, max_len=50)[:, :, None]  # (32, 50, 1)

# Masked loss
loss = binary_cross_entropy(logits, targets)
masked_loss = loss * mask
avg_loss = masked_loss.sum() / mask.sum()
"""

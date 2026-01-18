"""
RNN Utilities - Helper functions for RNN components

This module contains utility functions for working with RNNs,
such as initialization, padding, and masking.

Author: Your Name
Date: 2026-01-18
"""

import jax.numpy as jnp
from typing import List, Optional, Tuple


def initialize_carry(batch_size: int, hidden_size: int) -> jnp.ndarray:
    """
    Initialize the hidden state (carry) for an RNN.
    
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


def pad_sequences(
    sequences: List[jnp.ndarray],
    max_len: Optional[int] = None,
    padding: str = 'post',
    value: float = 0.0
) -> jnp.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of arrays with shape (seq_len, features)
        max_len: Maximum length to pad to. If None, use longest sequence.
        padding: 'pre' or 'post' - where to add padding
        value: Value to use for padding
    
    Returns:
        Padded array of shape (num_sequences, max_len, features)
    
    Example:
        >>> seq1 = jnp.array([[1, 2], [3, 4]])  # length 2
        >>> seq2 = jnp.array([[5, 6], [7, 8], [9, 10]])  # length 3
        >>> padded = pad_sequences([seq1, seq2])
        >>> padded.shape
        (2, 3, 2)  # 2 sequences, max_len=3, 2 features
    """
    
    # TODO: Implement sequence padding
    # GUIDANCE:
    # 1. Find max_len if not provided
    # 2. Create output array filled with padding value
    # 3. Copy each sequence into the output array
    # 4. Handle 'pre' vs 'post' padding
    
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    num_sequences = len(sequences)
    features = sequences[0].shape[-1] if sequences[0].ndim > 1 else 1
    
    # Create output array
    if sequences[0].ndim > 1:
        output_shape = (num_sequences, max_len, features)
    else:
        output_shape = (num_sequences, max_len)
    
    padded = jnp.full(output_shape, value)
    
    # TODO: Fill in sequences
    # Hint: Use array slicing to place each sequence
    
    return padded


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
        >>> lengths = jnp.array([3, 5, 2])  # 3 sequences
        >>> mask = create_padding_mask(lengths, max_len=5)
        >>> mask
        [[1. 1. 1. 0. 0.]
         [1. 1. 1. 1. 1.]
         [1. 1. 0. 0. 0.]]
    """
    
    # TODO: Implement mask creation
    # GUIDANCE:
    # Create a range [0, 1, 2, ..., max_len-1]
    # Compare with lengths to get binary mask
    
    positions = jnp.arange(max_len)  # [0, 1, 2, ..., max_len-1]
    # TODO: Compare positions with lengths
    # Hint: Use broadcasting - positions[None, :] < lengths[:, None]
    
    mask = None  # Replace with comparison
    return mask.astype(jnp.float32)


# ============================================================================
# EXERCISE FOR YOU:
# ============================================================================
#
# 1. Complete the pad_sequences function
#
# 2. Complete the create_padding_mask function
#
# 3. Test in a notebook:
#    ```python
#    import jax.numpy as jnp
#    from RNN_component_modelling.rnn_utils import (
#        initialize_carry,
#        pad_sequences,
#        create_padding_mask
#    )
#    
#    # Test initialize_carry
#    h_0 = initialize_carry(batch_size=4, hidden_size=32)
#    print(f"Initial carry shape: {h_0.shape}")
#    
#    # Test padding mask
#    lengths = jnp.array([3, 5, 2, 4])
#    mask = create_padding_mask(lengths, max_len=5)
#    print(f"Mask:\n{mask}")
#    ```
#
# 4. Questions to think about:
#    - Why do we need padding for variable-length sequences?
#    - How would you use the mask during training?
#    - What's the difference between 'pre' and 'post' padding?
#
# ============================================================================

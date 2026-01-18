# Recurrent Neural Networks (RNN) - Component Library

## 📚 What is an RNN?

### The Core Idea

Imagine reading a sentence word by word. As you read each word, you **remember** what you've read before. This memory helps you understand the current word in context.

**RNNs work the same way:**
- Process sequences one element at a time
- Maintain a "hidden state" (memory) that updates with each step
- Use both current input AND previous memory to make decisions

### RNN vs Bag-of-Words (Your Current Approach)

| Bag-of-Words (BoW) | Recurrent Neural Network (RNN) |
|-------------------|-------------------------------|
| "I love this" → [1, 1, 1] | "I" → "love" → "this" (sequential) |
| Order doesn't matter | Order matters! |
| Fast, simple | Slower, but captures context |
| "not good" = "good not" | "not good" ≠ "good not" |

**Example:**
- Text: "This movie is **not** good"
- BoW: Sees "good" → might predict positive ❌
- RNN: Sees "not" before "good" → predicts negative ✅

---

## 🧠 How RNNs Work: Step-by-Step

### The RNN Cell (Single Timestep)

```
Input (x_t) ──┐
              ├──> [RNN Cell] ──> Output (h_t)
Previous (h_{t-1}) ──┘              │
                                    └──> (becomes next h_{t-1})
```

**Mathematical Formula:**
```
h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
```

**Breaking it down:**
- `x_t`: Current input (e.g., word embedding)
- `h_{t-1}`: Previous hidden state (memory)
- `W_ih`: Weight matrix for input
- `W_hh`: Weight matrix for hidden state (recurrence!)
- `b`: Bias term
- `tanh`: Activation function (squashes to [-1, 1])

### Processing a Sequence

**Example: "I love movies"**

```
Step 1: "I"
  h_0 = [0, 0, 0]  (initial state)
  x_1 = embed("I")
  h_1 = tanh(W_ih @ x_1 + W_hh @ h_0 + b)

Step 2: "love"
  h_1 = [0.3, -0.1, 0.5]  (from step 1)
  x_2 = embed("love")
  h_2 = tanh(W_ih @ x_2 + W_hh @ h_1 + b)  ← Uses h_1!

Step 3: "movies"
  h_2 = [0.7, 0.2, -0.3]  (from step 2)
  x_3 = embed("movies")
  h_3 = tanh(W_ih @ x_3 + W_hh @ h_2 + b)  ← Uses h_2!

Final: h_3 contains information about ALL words
```

**Key Insight:** Each hidden state `h_t` is influenced by ALL previous words!

---

## 🏗️ Architecture Components

### Component 1: RNN Cell (`rnn_cells.py`)

**Purpose:** Process ONE timestep

```python
from flax import linen as nn

class VanillaRNNCell(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, carry, x):
        """
        Args:
            carry: Previous hidden state (batch, hidden_size)
            x: Current input (batch, input_dim)
        
        Returns:
            new_carry: Updated hidden state
            output: Same as new_carry (for RNN)
        """
        h_prev = carry
        
        # Linear transformations
        h_new = nn.Dense(self.hidden_size)(x) + \
                nn.Dense(self.hidden_size)(h_prev)
        
        # Activation
        h_new = nn.tanh(h_new)
        
        return h_new, h_new
```

**Why this design?**
- `carry`: JAX's way of passing state through time
- Returns `(new_state, output)`: Standard JAX scan pattern
- `@nn.compact`: Flax decorator for inline layer definition

---

### Component 2: RNN Layer (`rnn_layers.py`)

**Purpose:** Process ENTIRE sequence using the cell

```python
import jax
from jax import lax

class RNN(nn.Module):
    hidden_size: int
    return_sequences: bool = False
    
    def setup(self):
        self.cell = VanillaRNNCell(self.hidden_size)
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input sequence (batch, seq_len, input_dim)
        
        Returns:
            If return_sequences=True: (batch, seq_len, hidden_size)
            If return_sequences=False: (batch, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state
        h_0 = jnp.zeros((batch_size, self.hidden_size))
        
        # Process sequence with scan
        # scan applies cell to each timestep efficiently
        final_carry, all_outputs = lax.scan(
            self.cell,
            init=h_0,
            xs=jnp.transpose(x, (1, 0, 2))  # (seq_len, batch, input_dim)
        )
        
        if self.return_sequences:
            return jnp.transpose(all_outputs, (1, 0, 2))  # (batch, seq_len, hidden)
        else:
            return final_carry  # (batch, hidden)
```

**Key Concepts:**
- `lax.scan`: Efficient loop in JAX (compiled, fast!)
- `return_sequences`: Get all hidden states vs just final
- Transpose: JAX scan expects (time, batch, features)

---

### Component 3: Utilities (`rnn_utils.py`)

**Helper functions for common tasks:**

```python
def initialize_carry(batch_size: int, hidden_size: int):
    """Create initial hidden state (all zeros)"""
    return jnp.zeros((batch_size, hidden_size))

def pad_sequences(sequences, max_len=None, padding='post'):
    """Pad variable-length sequences to same length"""
    # Implementation here
    pass
```

---

## 🎯 Using Components in Notebooks

### Example: Sentiment Classification

```python
# In your notebook:
import sys
sys.path.append('../src')

from RNN_component_modelling.rnn_layers import RNN
from RNN_component_modelling.rnn_cells import VanillaRNNCell
import jax.numpy as jnp
from flax import linen as nn

# Build a sentiment classifier
class SentimentRNN(nn.Module):
    vocab_size: int = 8000
    embed_dim: int = 128
    hidden_size: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len) - token IDs
        
        # 1. Embed tokens
        x = nn.Embed(self.vocab_size, self.embed_dim)(x)
        # x: (batch, seq_len, embed_dim)
        
        # 2. Process with RNN (get final state only)
        x = RNN(self.hidden_size, return_sequences=False)(x)
        # x: (batch, hidden_size)
        
        # 3. Classification head
        x = nn.Dense(1)(x)  # (batch, 1)
        
        return x

# Initialize and use
model = SentimentRNN()
params = model.init(jax.random.PRNGKey(0), jnp.ones((2, 50), dtype=jnp.int32))

# Predict
sample_text_ids = jnp.array([[1, 45, 234, 12, ...]])  # Your tokenized text
logits = model.apply(params, sample_text_ids)
prediction = nn.sigmoid(logits)
```

---

## 📊 Training Process

### How Gradients Flow Through Time

**Forward Pass:**
```
x_1 → h_1 → h_2 → h_3 → output → loss
```

**Backward Pass (Backpropagation Through Time - BPTT):**
```
loss → ∂output → ∂h_3 → ∂h_2 → ∂h_1 → ∂W
```

**Key Challenge:** Gradients can **vanish** or **explode** over long sequences
- **Vanishing:** Gradients become tiny → early words don't learn
- **Exploding:** Gradients become huge → unstable training

**Solutions:**
- Gradient clipping (limit max gradient)
- LSTM/GRU cells (better memory mechanisms)
- Shorter sequences during training

---

## 🔬 Comparison: RNN vs Your BoW Model

### When to Use RNN:
✅ Order matters ("not good" vs "good not")  
✅ Context is important ("This movie is not bad" - double negative)  
✅ Longer-range dependencies  

### When to Use BoW:
✅ Fast training and inference  
✅ Simple sentiment (clear positive/negative words)  
✅ Limited compute resources  
✅ Baseline model  

### Expected Performance:
- **BoW (your baseline):** ~82% F1
- **Basic RNN:** ~83-85% F1 (slight improvement)
- **LSTM/GRU:** ~86-88% F1 (better long-term memory)

---

## 🚀 Next Steps

1. **Understand the theory** (you're here!)
2. **Build `rnn_cells.py`** - Implement the basic cell
3. **Build `rnn_layers.py`** - Wrap cell for sequences
4. **Test in notebook** - Import and verify
5. **Train on your data** - Compare with BoW baseline
6. **Experiment** - Try different hidden sizes, layers, etc.

---

## 📖 Additional Resources

- **JAX Documentation:** https://jax.readthedocs.io/
- **Flax Documentation:** https://flax.readthedocs.io/
- **Understanding LSTMs:** http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Your BoW Implementation:** `notebooks/Jax-workstation-basic.ipynb`

---

## 🤔 Common Questions

**Q: Why use `lax.scan` instead of a for loop?**  
A: `lax.scan` is JIT-compiled and much faster. Regular Python loops don't work well with JAX's transformations.

**Q: What's the difference between RNN cell and RNN layer?**  
A: Cell processes ONE timestep. Layer processes ENTIRE sequence by repeatedly calling the cell.

**Q: Can I stack multiple RNN layers?**  
A: Yes! Set `return_sequences=True` on all but the last layer.

**Q: How do I handle variable-length sequences?**  
A: Pad to max length and use masking (we'll add this in `rnn_utils.py`).

---

**Ready to start building? Let's create the components!** 🎉

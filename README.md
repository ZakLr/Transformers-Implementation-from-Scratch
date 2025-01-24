# Transformer from Scratch

A custom implementation of the Transformer architecture for sequence-to-sequence tasks in Natural Language Processing (NLP).

## Preparing Repository

1. Create an `images` directory in your repository:
```bash
mkdir images
```

2. Download these images and save them in the `images` directory:
- [Transformer Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Transformer_architecture.png/1200px-Transformer_architecture.png)
- [Transformer Block Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Transformer_Model_Architecture.png/1200px-Transformer_Model_Architecture.png)
- [Attention Mechanism](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Scaled_Dot_Product_Attention.png/1200px-Scaled_Dot_Product_Attention.png)
- [Transformer Components](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Transformer_components.png/1200px-Transformer_components.png)

## Background

The Transformer was introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. in 2017.

![Transformer Architecture](images/Architecture_d'un_Transformeur.png)

### Key Advantages

- Parallel Processing of sequences
- Global Context Capture
- High Scalability
- Reduced Training Time

## Architecture Overview

The Transformer comprises:

1. Encoder
2. Decoder
3. Multi-Head Attention
4. Positional Encoding

### Detailed Architecture

![Transformer Block Diagram](images/transformer_block_architecture.png)

## Mathematical Foundations

### Multi-Head Self-Attention Formula

The core attention mechanism is defined by:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- Q: Query matrices
- K: Key matrices
- V: Value matrices
- $d_k$: Dimensionality scaling factor

## Visualization of Components

![Attention Mechanism](images/Self_attention_mechanism.jpg)

![Transformer Components](images/transformer_components.svg)

## Code Example

```python
import torch

# Transformer usage example
src = torch.tensor([[1, 5, 6, 4, 3, 9, 2, 0, 0], 
                    [1, 8, 7, 3, 9, 2, 0, 0, 0]]).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], 
                    [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

out = transformer(src, trg[:, :-1])
print(out.shape)
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/transformer-from-scratch.git
pip install torch numpy
```

## Contributing

- Open issues for bugs
- Submit pull requests
- Follow code style guidelines

## Resources

- [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- [Wikipedia: Transformer Model](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

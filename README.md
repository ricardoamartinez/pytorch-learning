# PyTorch Learning Path: From Basics to World Models

A comprehensive, hands-on curriculum for learning PyTorch, progressing from fundamentals to implementing world models. Designed for ML practitioners who understand the theory but are new to coding.

## Overview

This repository contains 15 progressive lessons that take you from PyTorch basics to implementing a complete Dreamer-style world model.

```
Observation ──► [Encoder/VAE] ──► Latent z
                                     │
                                     ▼
                              [RSSM Dynamics] ◄── Action
                                     │
                                     ▼
                               Next Latent z'
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                    [Reward Head]         [Decoder]
                          │                     │
                          ▼                     ▼
                    Predicted r          Predicted obs
```

## Curriculum

### Part 1: PyTorch Foundations
| # | File | Topics |
|---|------|--------|
| 01 | `01_python_basics.py` | Variables, loops, functions, conditionals |
| 02 | `02_tensors_intro.py` | Tensor creation, shapes, operations, GPU |
| 03 | `03_autograd.py` | Automatic differentiation, computational graphs |
| 04 | `04_neural_network.py` | nn.Module, layers, Sequential vs custom |
| 05 | `05_training_loop.py` | DataLoader, training loop, saving/loading |
| 06 | `06_cnn_example.py` | MNIST CNN classification end-to-end |

### Part 2: Sequence Modeling
| # | File | Topics |
|---|------|--------|
| 07 | `07_rnn_lstm.py` | RNN, LSTM, GRU, bidirectional, stacked |
| 08 | `08_sequence_prediction.py` | Forecasting, autoregressive, encoder-decoder |

### Part 3: Representation Learning
| # | File | Topics |
|---|------|--------|
| 09 | `09_autoencoders.py` | Autoencoders, latent space, denoising |
| 10 | `10_vae.py` | VAE, reparameterization, KL divergence, ELBO |

### Part 4: Attention & Transformers
| # | File | Topics |
|---|------|--------|
| 11 | `11_attention.py` | Self-attention, multi-head, positional encoding |
| 12 | `12_transformers.py` | Full Transformer, GPT-style decoder-only |

### Part 5: Reinforcement Learning
| # | File | Topics |
|---|------|--------|
| 13 | `13_reinforcement_learning.py` | Policy/value networks, REINFORCE, actor-critic |

### Part 6: World Models
| # | File | Topics |
|---|------|--------|
| 14 | `14_latent_dynamics.py` | RSSM, prior/posterior, imagination rollouts |
| 15 | `15_world_model.py` | Complete Dreamer-style world model |

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pytorch-learning.git
cd pytorch-learning

# Install dependencies
pip install torch torchvision numpy

# Optional: for RL environments
pip install gymnasium
```

### Running Lessons

```bash
# Run any lesson
python 01_python_basics.py
python 07_rnn_lstm.py
python 15_world_model.py
```

Each file is self-contained and heavily commented. Run them sequentially for the best learning experience.

## Key Concepts Covered

- **Tensors & Autograd**: PyTorch's computational foundation
- **Neural Networks**: Building blocks with nn.Module
- **CNNs**: Convolutional networks for images
- **RNNs/LSTMs**: Sequence modeling with memory
- **Autoencoders & VAEs**: Learning latent representations
- **Attention & Transformers**: Modern sequence architectures
- **Reinforcement Learning**: Policies, values, and actor-critic
- **World Models**: RSSM, imagination, Dreamer architecture

## Famous World Models Referenced

- **Dreamer (2020-2023)**: RSSM + Actor-Critic
- **MuZero (2019)**: Learned model + MCTS
- **IRIS (2023)**: Transformer dynamics with discrete tokens
- **Genie (2024)**: Generative world model from video
- **GAIA-1 (2023)**: World model for autonomous driving

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [World Models Paper](https://worldmodels.github.io/)
- [Dreamer Papers](https://danijar.com/project/dreamer/)
- [Dive into Deep Learning](https://d2l.ai/)

## License

MIT License - feel free to use for learning and projects.

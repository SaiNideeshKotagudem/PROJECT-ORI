# 🧬 RNA 3D Folding Prediction: Multi-Modal Deep Learning Architecture

> A modular PyTorch-based pipeline to predict RNA tertiary structure (x, y, z atomic coordinates) from primary sequence, secondary structure, base pairing probabilities, and text-based annotations.
>
> This architecture is designed for the **Stanford Ribonanza Challenge**, integrating GNNs, Transformers, multi-modal fusion, and DeepSpeed acceleration.

---

## 📁 Repository Structure

```
├── model/
│   ├── sequence_encoder.py            # Transformer-based encoder for nucleotide sequences
│   ├── description_encoder.py         # Transformer-based encoder for description strings
│   ├── secondary_structure_encoder.py # Message-passing GNN for base-pair graphs
│   ├── bpp_graph_encoder.py           # Attention-based encoder for BPP graph
│   ├── feature_fusion.py              # Cross-attention + projection-based fusion
│   ├── final_model.py                 # End-to-end architecture integrating all components
├── data/
│   ├── dataset.py                     # Custom dataset loader for sequence/label CSVs
│   ├── graph_utils.py                 # Utilities for edge_index, masking, etc.
├── train.py                           # Training script (DeepSpeed integrated)
|___predict.py # Evaluation and prediction on test set
├── utils.py                           # General utility functions
├── config.py                          # Model and training configuration parameters
└── README.md                          # This file
```

---

## 🧠 Model Architecture Overview

The architecture predicts the 3D coordinates of RNA residues given multiple input modalities:

1. **Primary Sequence** (A, U, G, C) → Transformer encoder
2. **Secondary Structure** → Message-Passing GNN
3. **Base Pairing Probability Matrix (BPP)** → Graph Attention Encoder
4. **Freeform Description Text** → Transformer text encoder
5. **Fusion** → Cross-attention followed by projection
6. **Output Head** → Fully connected network to regress 3D coordinates

---

## 🔢 Input Modalities

### 🧬 1. Primary Sequence

* Format: One-hot encoding of RNA sequence (A, U, G, C) → shape `[B, L, 4]`
* Embedded via a learnable linear projection to `d_model=128`
* Positional encoding added using sinusoidal formula
* Passed through `N=3` layers of `nn.TransformerEncoderLayer` (PyTorch-native)

  * `d_model=128`, `nhead=8`, `dim_feedforward=512`, dropout=0.1

### 📓 2. Description Encoder (Text Metadata)

* Input: Freeform strings (e.g., experimental conditions, tags)
* Tokenized as ASCII byte-level character IDs ∈ \[0, 127]
* Embedded using `nn.Embedding(128, 128)`
* Positional encoding (sinusoidal) added
* Two-layer Transformer (`nhead=4`, `ff=256`)
* Final representation is globally pooled (mean) and projected to `[B, 128]`

### 🔗 3. Secondary Structure Encoder

* Graph-based input: RNA secondary structure edges → `edge_index` format
* Node features: One-hot sequence embedding
* Implements a 2-layer GNN with degree-normalized message passing:

  ```python
  m_i = Σ_j ∈ N(i) (W1 · x_j) / deg(i)
  x_i = ReLU(W2 · m_i)
  ```
* Activation: ReLU; Output dim: `[B, L, 128]`
* Edge dropout applied for robustness

### 🔁 4. Base Pairing Probability (BPP) Graph Encoder

* Input: Sparse upper-triangular matrix with BPP weights
* Encoded as graph: nodes are nucleotides, edges carry BPP weights
* Uses a **Graph Attention Mechanism**:

  * Keys, Queries, Values computed per node
  * Attention weights scaled by edge BPP probability
  * Final node representations refined by attention-weighted sum over neighbors
  * Multi-head (optional), LayerNorm, dropout used
* Output: `[B, L, 128]`

---

## 🧬 Feature Fusion: Cross-Modality Attention

The sequence embeddings are contextually fused with the description vector:

```python
desc_expanded = repeat(desc_vec, L, axis=1)
attn_output, _ = MultiHeadAttention(query=seq_embed,
                                    key=desc_expanded,
                                    value=desc_expanded)
```

* Cross-attention produces sequence-aware conditioning on description.
* Attention output is concatenated with original embeddings → `[B, L, 256]`
* Final fusion: `LayerNorm → Linear(256 → 256) → GELU`

---

## 🎯 Prediction Head

```python
Fused Features → MLP Head → Regress (x, y, z)
```

* Head:

  * Linear(256 → 512) → ReLU
  * Linear(512 → 256) → ReLU
  * Linear(256 → 3)

Optionally, auxiliary heads can be added for:

* Residue identity prediction
* Base-pair classification
* Structure confidence estimation

---

## 🧪 Loss Function

Multi-objective loss with configurable weighting:

| Component       | Description                              | Default Weight |
| --------------- | ---------------------------------------- | -------------- |
| `MSE_Loss`      | Mean squared error over predicted coords | `1.0`          |
| `Resid_ID_Loss` | MSE or BCE over residue ID positions     | `0.5`          |
| `Resname_Loss`  | CrossEntropy on predicted base labels    | `0.5`          |
| `Coverage_Loss` | % of residues with valid prediction      | `1.0`          |

Total loss:

```python
loss = w1 * coord_loss + w2 * resid_loss + w3 * resname_loss + w4 * coverage_loss
```

---

## ⚡ Training Setup (DeepSpeed Enabled)

```bash
deepspeed train.py --deepspeed config/ds_config.json
```

### ✅ DeepSpeed Features:

* ZeRO Stage 1 support for optimizer state sharding
* Automatic FP16 mixed-precision
* FusedAdam optimizer (CUDA-native)

### 🔧 Key Training Config:

```python
batch_size      = 16
learning_rate   = 1e-3
optimizer       = FusedAdam
epochs          = 25+
warmup_steps    = 500
grad_clip       = 1.0
```

---

## 🔬 Evaluation

* Coordinate MAE, RMSE, Pearson r per-axis
* Optional: contact map agreement, torsion angle distance
* 3D structure visualization tools with PyMOL or Matplotlib

---

## 🧪 Dataset Format

### `train_sequences.csv`

\| id     | sequence     | description    | ...
\|--------|--------------|----------------|

### `train_labels.csv`

| id | resid | x | y | z |
| -- | ----- | - | - | - |

* Join on `(id, resid)`
* All coordinates in Angstroms (Å)
* Sequences are padded to fixed maximum length (`max_len=512`)

---

## 🛠️ Future Work

* Integrate **SE(3)-equivariant GNNs** (e.g. EGN, SE3-Transformer)
* Use **graph pooling** for long sequences
* Add **attention visualizations** for interpretability
* Pretrain with self-supervised contrastive learning

---

## 📚 References

* [SE(3)-Transformer: Equivariant Attention for 3D Data](https://arxiv.org/abs/2006.10503)
* [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903)
* [Transformer Architecture (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)

---

## 🧑‍💻 Author

**Sai Nideesh Kotagudem**
Email: [sainideeshkotagudem@gmail.com](mailto:sainideeshkotagudem@gmail.com)
GitHub: [github.com/SaiNideeshKotagudem](https://github.com/SaiNideeshKotagudem)

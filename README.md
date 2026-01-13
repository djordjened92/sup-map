This project is GNN-based clustering solution, designed to optimize node embeddings for community detection by minimizing the description length of information flow within a k-Nearest Neighbor (k-NN) graph.

---

## Methodology

The following formulas are used within the implementation for loss calculation and sampling. You can copy-paste these directly into documentation or code comments.

### 1. Map Equation Loss

The model minimizes the codelength of a random walk. The simplified loss used in `map_eq_loss` is:

`L = H(q_sum) - 2 * sum(H(q_i)) + sum(H(q_i + p_i))`

Where:

* `H(x) = x * log2(x)` (Entropy component)
* `q_sum`: Total exit probability from all modules.
* `q_i`: Exit probability of module `i`.
* `p_i`: Visit probability (steady-state) of module `i`.

### 2. Shannon Entropy (Graph Stability)

Used to calculate the information density of the local neighborhood:

`H = -sum(p * log(p))`

Where:

* `p = w_ij / S_i`
* `w_ij`: Edge weight between node `i` and `j`.
* `S_i`: Total out-degree weight of node `i`.

### 3. Entropy Annealing (Hard Negative Mining)

The threshold for admitting hard negatives decreases over time to sharpen the cluster boundaries:

`delta(t) = delta_start * (1 - t) + delta_end * t`
`H_max = H_reference * (1.0 + delta)`

Where:

* `t`: Training progress (current_epoch / total_epochs).
* `delta`: The current entropy "slack."

### 4. Geometric Regularization

The total loss function is defined as:

`Total_Loss = Map_Equation_Loss + Internal_Variance + Variance_of_Means`

* **Internal_Variance:** Minimizes the squared difference of intra-cluster similarities: `sum((w_internal - mean_w)**2) / (count - 1)`.
* **Variance_of_Means:** Maximizes the separation between different cluster centers.

---

## Model Architecture: Residual GraphSAGE

The GNN architecture uses a residual pattern to preserve features across layers and prevent over-smoothing.

`x_layer = SiLU(SAGEConv(Normalize(x_in), edge_index))`
`x_out = x_layer + x_in`

**Key Layers:**

1. **Normalization:** L2-normalization is applied before every convolution to ensure stability on the unit hypersphere.
2. **SiLU Activation:** A smooth, non-monotonic activation function ().
3. **Residual Addition:** The initial features are added back to the output of each layer to maintain structural identity.

---

## Hyperparameter Reference

### Training & Graph Setup

| Parameter | Value | Description |
| --- | --- | --- |
| `K` | `40` | Neighbors used to build the initial affinity graph. |
| `TAU` | `0.6` | Teleportation probability (restarts) for the random walker. |
| `BATCH_SIZE` | Config-driven | Managed by `RandomNodeLoader` (80 partitions). |
| `LR` | Config-driven | Initial learning rate for Adam (CycleLR scheduler). |

### Entropy Negative Sampler

| Parameter | Value | Description |
| --- | --- | --- |
| `neg_multiplier` | `5.0` | Ratio of negative edges to real edges per batch. |
| `pool_factor` | `50.0` | Multiplier for sampling the initial random pair pool. |
| `delta_start` | `10.0` | Initial entropy allowance for hard negatives. |
| `delta_end` | `3.0` | Final strict entropy threshold for hard negatives. |
| `anneal_frac` | `0.8` | Training percentage where annealing occurs. |

---

## Implementation Details

* **Negative Sampling:** The `EntropyNegativeSampler` filters for pairs where `labels[i] != labels[j]` and selects pairs with the highest cosine similarity (hardest to distinguish) that still satisfy the entropy constraint.
* **Sparse Flow:** `smart_teleportation_sparse` computes the steady-state distribution using power iteration on the sparse edge index to handle large graphs without  density.
* **Evaluation:** Performance is tracked via **Pairwise F-score** (measuring if pairs are correctly grouped) and **B-Cubed** metrics (measuring precision/recall per node).
import torch
from torch_scatter import scatter
from torch_cluster import random_walk
from torch_geometric.utils import subgraph

def smart_teleportation_sparse(edge_index, weights, num_nodes, tau, iterations=10):
    """Sparse version of teleportation using edge_index."""
    row, col = edge_index
    # Initial visit frequency 1/n
    p = torch.ones(num_nodes, device=edge_index.device) / num_nodes

    for _ in range(iterations):
        # Probability update: flow from row to col
        # p[row] is the prob at source, weights is the transition prob
        p_flow = p[row] * weights * (1 - tau)
        
        # Aggregate flow into destination nodes
        w_in = scatter(p_flow, col, dim=0, dim_size=num_nodes, reduce='sum')
        
        # Add teleportation part
        p_new = w_in + (tau / num_nodes)
        p = p_new
        
    return p

def module_exit_prob_sparse(edge_index, weights, num_nodes, labels, tau):
    """Exit probability logic adapted for sparse edge_index."""
    row, col = edge_index
    
    # 1. Steady state visit distribution
    p = smart_teleportation_sparse(edge_index, weights, num_nodes, tau)
    
    # 2. Module visit rates (sum of p for all nodes in a label)
    # unique labels mapped to 0...C-1
    unique_labels, l_indices, module_size = torch.unique(labels, return_inverse=True, return_counts=True)
    num_modules = unique_labels.size(0)
    
    module_visit_rate = scatter(p, l_indices, dim=0, dim_size=num_modules, reduce='sum')
    
    # 3. Exit probabilities
    # Teleportation part of exit prob
    teleport_part = module_visit_rate * tau * (num_nodes - module_size) / num_nodes
    
    # Out-of-module flow: edge row->col where label[row] != label[col]
    is_out_flow = (l_indices[row] != l_indices[col]).float()
    out_of_module_flow = p[row] * weights * (1 - tau) * is_out_flow
    
    # Aggregate exit flow per module (from the source node's module)
    mep = scatter(out_of_module_flow, l_indices[row], dim=0, dim_size=num_modules, reduce='sum')
    mep = mep + teleport_part

    return mep, module_visit_rate

def H(x, eps=1e-10):
    """Helper for the Entropy component x*log2(x)"""
    # Entropy must be 0 if x is 0. 
    # We use relu to ensure no tiny negative noise enters the log
    x = torch.clamp(x, min=eps)
    return x * torch.log2(x)

def map_eq_loss(features, edge_index, labels, tau=0.1):
    num_nodes = features.size(0)
    device = features.device
    
    # 1. Similarity and Transition Probabilities
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    row, col = edge_index
    
    # Cosine-like similarity: (u·v)
    # Ensure non-negative similarities
    out_sims = torch.clamp((features[row] * features[col]).sum(dim=-1), min=0.0)
    
    # STRICT NORMALIZATION: Ensure row-stochastic weights
    out_degree = scatter(out_sims, row, dim=0, dim_size=num_nodes, reduce='sum')
    weights = out_sims / (out_degree[row] + 1e-9)

    # 2. Steady State and Flow
    # mep (q_i): Exit probability of module i
    # mvr (p_i): Visit probability of module i
    mep, mvr = module_exit_prob_sparse(edge_index, weights, num_nodes, labels, tau)

    # 3. Secure Codelength Calculation
    # Normalize probabilities so they sum to 1 (or close to it via tau)
    # This prevents the log terms from shifting into the wrong signs
    q_sum = mep.sum()
    
    # Equation components:
    # Term 1: Entropy of the total exit probability
    term1 = H(q_sum)
    
    # Term 2: Sum of entropies of exit probabilities per module
    term2 = H(mep).sum()
    
    # Term 3: Sum of entropies of (exit + visit) probabilities per module
    term3 = H(mep + mvr).sum()

    # Full Equation: L = Term1 - 2*Term2 + Term3 (simplified version)
    # Or more standard: L = Term1 - Term2 + Term3-of-per-node (but your version is fine)
    codelength = term1 - 2 * term2 + term3
    
    # Force non-negative result (numerical safety)
    return torch.relu(codelength)

def avg_internal_variance(edge_index: torch.Tensor, 
                          features: torch.Tensor, 
                          labels: torch.Tensor):
    """
    Calculates the mean of internal cluster variance and the variance of cluster means
    using the edges and features of the filtered subgraph (or full graph), 
    explicitly removing self-loops and correctly applying the cluster size threshold 
    for variance calculation.

    Args:
        edge_index: Edge index (2, E). Assumed to be the affinity graph (KNN).
        features: Feature matrix (N_sub, D).
        labels: Labels of the nodes (N_sub).
    
    Returns:
        (avg_variance_per_class, variance_of_means)
    """

    epsilon = torch.finfo(torch.float32).eps
    
    # --- 1. Filter Self-Loops ---
    row_all, col_all = edge_index
    
    # Create a mask to identify non-self-loop edges (row != col)
    non_self_loop_mask = row_all != col_all
    
    row = row_all[non_self_loop_mask]
    col = col_all[non_self_loop_mask]
    
    # If no edges remain after filtering self-loops, return zero variance
    if row.numel() == 0:
        device = features.device
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
    # --- 2. Compute Edge Weights (Similarity W) ---
    # W[i] = feature[row[i]] @ feature[col[i]]
    W = (features[row] * features[col]).sum(dim=1)
    W = torch.clamp(W, min=0.) 

    # --- 3. Identify Internal Edges and Class Info ---

    # Get class membership for source and destination nodes
    label_row = labels[row]
    label_col = labels[col]
    
    # int_mask: 1 if edge is internal
    int_mask = (label_row == label_col).float() 
    
    # Get unique labels and map them (needed for scatter)
    unique_labels, labels_inv, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    
    # scatter_idx: map source node labels (row) to the scatter index (labels_inv)
    scatter_idx = labels_inv[row]

    # --- 4. Calculate Internal Weight Sums and RAW Means ---
    
    # Sum of internal edges (count of internal edges) per class
    num_edges_per_class = scatter(int_mask, scatter_idx, dim=0, dim_size=len(unique_labels), reduce='sum')

    # Total weight of internal edges per class
    int_w = W * int_mask # Only weights of internal edges
    int_per_class = scatter(int_w, scatter_idx, dim=0, dim_size=len(unique_labels), reduce='sum')
    
    # RAW Mean internal similarity per class (used for subtraction)
    int_mean_raw = int_per_class / (num_edges_per_class + epsilon)
    
    # Filter for clusters with enough edges (> 100)
    is_valid_cluster = (num_edges_per_class > 100)
    
    # --- 5. Calculate Variance of Internal Similarity (Avg Internal Variance) ---
    
    # Get the RAW mean corresponding to each internal edge 
    # This correctly computes the mean for subtraction for ALL clusters
    int_mean_for_edge = int_mean_raw[labels_inv[row]]
    
    # Squared difference: (internal_weight - class_mean_raw)**2
    # Only calculate for internal edges (where int_mask is 1)
    sq_diff = (int_w - int_mean_for_edge * int_mask)**2
    
    # Sum of squared differences per class
    sum_sq_diff_per_class = scatter(sq_diff, scatter_idx, dim=0, dim_size=len(unique_labels), reduce='sum')
    
    # Variance = Sum of Sq Diff / (Count - 1). Apply the zero-out mask.
    variance_per_class = torch.where(is_valid_cluster,
                                     sum_sq_diff_per_class / (num_edges_per_class - 1 + epsilon),
                                     torch.zeros_like(sum_sq_diff_per_class))
    
    # Average variance over all unique classes 
    avg_variance_per_class = variance_per_class.mean()

    # --- 6. Calculate Variance of Cluster Means ---

    if int_mean_raw.numel() > 1:
        # Variance of means: Var(X) = sum((X - mean(X))^2) / (N - 1)
        variance_of_means = ((int_mean_raw - int_mean_raw.mean())**2).sum() / (int_mean_raw.shape[0] - 1)
    else:
        variance_of_means = torch.tensor(0.0, device=features.device)
        
    return avg_variance_per_class, variance_of_means

def randomwalk_nodes(edge_index: torch.Tensor, B: int, walk_length: int = 2, num_walks: int = 2) -> torch.Tensor:
    
    # 1. Determine the starting nodes (roots)
    start_nodes = torch.arange(B, device=edge_index.device)
    
    # Extract source (row) and destination (col) tensors
    row, col = edge_index[0], edge_index[1]

    # 2. Perform the Random Walks
    walk_nodes = random_walk(
        row, # Positional argument 1: source nodes
        col, # Positional argument 2: destination nodes
        start_nodes.repeat(num_walks), 
        walk_length,
    )
    # walk_nodes shape is [num_walks * B, walk_length + 1]

    # 3. Flatten and get unique reachable nodes within the initial batch B
    rw_nodes = walk_nodes.flatten().unique()
    rw_nodes = rw_nodes[rw_nodes < B] 
    
    return rw_nodes

def create_rw_mask(knn_idx: torch.Tensor, rw_nodes: torch.Tensor, B: int) -> torch.Tensor:
    knn_idx = knn_idx[:, 1:]
    allowed = torch.zeros(B, dtype=torch.bool, device=knn_idx.device)
    allowed[rw_nodes] = True
    mask = allowed[knn_idx]
    return mask

def filter_graph_by_rw(original_edge_index: torch.Tensor, original_labels: torch.Tensor, rw_nodes: torch.Tensor):
    """
    Filters the edge_index and labels to only include nodes present in rw_nodes.
    
    Args:
        original_edge_index: The full edge index (2, E).
        original_labels: The full labels tensor (N).
        rw_nodes: The indices of nodes (0 to B-1) visited by the random walk.
        
    Returns:
        (filtered_edge_index, filtered_labels, node_map_mask)
    """
    
    # 1. Filter the edge_index
    # The 'subgraph' utility keeps edges whose source and destination are in 'rw_nodes'.
    # It also re-maps the node indices in the edge_index to be 0-based based on the order in rw_nodes.
    filtered_edge_index, _ = subgraph(
        subset=rw_nodes,
        edge_index=original_edge_index,
        relabel_nodes=True,
        num_nodes=original_labels.size(0) # Ensure the utility knows the total number of nodes (B)
    )

    # 2. Filter the labels
    filtered_labels = original_labels[rw_nodes]
    
    # The node_map_mask is a boolean mask of size N, which is True for the nodes in rw_nodes.
    # It's not strictly necessary for the variance function but is returned for completeness
    # if you needed to map back to the original index.
    
    return filtered_edge_index, filtered_labels
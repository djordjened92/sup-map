import torch
import numpy as np
import infomap
from time import time
from tqdm import tqdm

def face_cluster(data,
                 batch_size=500_000, 
                 flow_model='undirected',
                 no_self_links=True,
                 silent=False):
    t = time()
    
    device = data.x.device
    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # Pre-normalize features to speed up cosine similarity
    x_norm = torch.nn.functional.normalize(data.x, p=2, dim=1)

    # Initialize Infomap
    # Note: ensure your infomap version supports these arguments
    info = infomap.Infomap(
        "--two-level --seed 100",
        flow_model=flow_model,
        no_self_links=no_self_links,
        silent=silent
    )

    # Batched Similarity Calculation
    for i in tqdm(range(0, num_edges, batch_size), desc="Adding links", disable=silent):
        end = min(i + batch_size, num_edges)
        batch_edges = edge_index[:, i:end]
        
        src, dst = batch_edges[0], batch_edges[1]
        
        # Calculate cosine similarity on GPU
        sim_batch = (x_norm[src] * x_norm[dst]).sum(dim=-1)
        
        # Move to CPU for Infomap ingestion
        src_cpu = src.cpu().numpy()
        dst_cpu = dst.cpu().numpy()
        sim_cpu = sim_batch.cpu().numpy()

        for idx in range(len(src_cpu)):
            u, v, w = int(src_cpu[idx]), int(dst_cpu[idx]), float(sim_cpu[idx])
            # Infomap requires weights > 0
            if w > 0:
                info.add_link(u, v, round(w, 8))

    print("Running Infomap solver (this may take a while)...")
    info.run()

    # Create label array
    pred_labels = np.zeros(num_nodes, dtype=int) - 1
    
    # Iterate through the tree and extract leaf nodes
    for node in info.iterTree():
        if node.is_leaf:
            # node.physical_id is the original index provided in add_link
            # node.module_id is the cluster assignment
            pred_labels[node.physicalId] = node.moduleIndex()

    # Handle nodes that were not part of any edges (isolated nodes)
    unassigned = np.where(pred_labels == -1)[0]
    if len(unassigned) > 0:
        # Start new IDs from the highest existing cluster ID + 1
        current_max = pred_labels.max() if pred_labels.max() >= 0 else 0
        for i, node_idx in enumerate(unassigned):
            pred_labels[node_idx] = current_max + 1 + i

    print(f'Done. Total time: {time() - t:.2f}s')
    return pred_labels
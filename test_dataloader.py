import dgl
import torch
import numpy as np
from utils import l2norm, knn_faiss, knns2spmat, sparse_mx_to_indices_values, to_dgl
from dataset import load_feat, load_labels, FeatureDataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

feat_dim = 256
k = 30
device = torch.device("cuda")

val_feat_path = '/home/djordje/Documents/Projects/pyg/learn-to-cluster/data/features/part1_test.bin'
val_feat = load_feat(val_feat_path, feat_dim)
val_features = l2norm(val_feat)
print('Test:')
print('features shape:', val_features.shape)

val_label_path = '/home/djordje/Documents/Projects/pyg/learn-to-cluster/data/labels/part1_test.meta'
val_labels = load_labels(val_label_path)
print(f'num of labels: {val_labels.shape[0]}')
print(f'#cls: {len(np.unique(val_labels))}')

with torch.no_grad():
    sims, nbrs = knn_faiss(val_features, k=k)
    spmat = knns2spmat(torch.from_numpy(sims).to(device), torch.from_numpy(nbrs).to(device))
    indices, _ = sparse_mx_to_indices_values(spmat)

    data = Data(x=torch.from_numpy(val_features).to(device), edge_index=indices.contiguous()).to(device)

    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=1024
    )

    total_len = 0
    for batch in loader:
        total_len += batch.num_sampled_nodes[0]
    print(total_len)
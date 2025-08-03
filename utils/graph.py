import torch
import torch_geometric as pyg
from torch_scatter import scatter_max, scatter_mean
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from utils import knn_faiss, \
                  z_outlier_detection, \
                  jaccard_index, \
                  knns2spmat, \
                  sparse_mx_to_indices_values

def gen_graph(features, config, K, device, z_score=False):
    sims, nbrs = knn_faiss(features, k=K)
    sims = sims / sims.sum(dim=1, keepdims=True)

    switch_points = []
    if z_score:
        nbrs, switch_points = z_outlier_detection(sims, nbrs, config['WINDOW_SIZE'])

    spmat = knns2spmat(sims, nbrs)
    indices, _ = sparse_mx_to_indices_values(spmat)

    data = Data(x=features, edge_index=indices.contiguous()).to(device)
    data['nbrs_bounds'] = switch_points

    return data, nbrs

def out_degrees(data):
    return pyg.utils.degree(data.edge_index[0], data.num_nodes)

def in_degrees(data):
    return pyg.utils.degree(data.edge_index[1], data.num_nodes)

def filter_edges(data, tau):
    # Filter by edge connection score
    edge_conn_filter = data['raw_affine'] > tau

    # Keep edges with direction from node of lower to higher density
    src_win = data['w_in'][data.edge_index[0]]
    dst_win = data['w_in'][data.edge_index[1]]
    win_filter = src_win < dst_win

    edge_mask = torch.logical_and(edge_conn_filter, win_filter)
    return edge_mask

def tree_generation(data, device, edge_conn='raw_affine'):
    # Keep edges with highest probability
    src = data.edge_index[0]
    uniq = src.unique()
    basis = SparseTensor(row=uniq.long(),
                         col=torch.zeros_like(uniq).to(uniq.device).long(),
                         value=torch.arange(len(uniq)).to(uniq.device),
                         sparse_sizes=(data.x.size(0), 1))
    src_index = basis[src].to_dense().squeeze()
    out, argmax = scatter_max(data[edge_conn], src_index, dim=-1) # for each src node find edge with highest prob
    filtered_edges = data.edge_index[:, argmax]
    treeg = Data(x=data.x, edge_index=filtered_edges).to(device)
    return treeg
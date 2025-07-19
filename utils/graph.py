import dgl
import torch
import random
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

def to_dgl(data, device):
    '''
    Current pyg version (2.5.3) has a bug for converting the graph
    with isolated nodes.
    This is simplified implementation with the bug fix.
    '''
    row, col = data.edge_index
    g = dgl.graph((row, col), num_nodes=data.num_nodes)

    for attr in data.node_attrs():
        g.ndata[attr] = data[attr].to(device)
    for attr in data.edge_attrs():
        if attr in ['edge_index', 'adj_t']:
            continue
        g.edata[attr] = data[attr].to(device)
    return g

def peak_propagation(data):
    pred_labels = torch.zeros(data.num_nodes).long() - 1 # initialize pred_labels to -1
    peaks = torch.where(out_degrees(data) == 0)[0] # filter for nodes with out degree 0
    pred_labels[peaks] = torch.arange(peaks.shape[0]) # assign peak labels as indices of the list
    
    rev_edge_index = data.edge_index.flip([0]) # reverse edge direction to propagate peak labels
    rev_data = Data(x=pred_labels, edge_index=rev_edge_index) # create temporary, reversed pyg graph
    rev_data.num_nodes = data.num_nodes

    # Convert pyg to dgl graph, to be able to propagate peak label sequentially
    g = to_dgl(rev_data, data.x.device)

    def message_func(edges):
        return {'mlb': edges.src['x']}

    def reduce_func(nodes):
        return {'x': nodes.mailbox['mlb'][:, 0]}

    node_order = [nids.to(g.device) for nids in dgl.traversal.topological_nodes_generator(g, reverse=False)]
    g.prop_nodes(node_order, message_func, reduce_func) # propagate labels
    pred_labels = g.ndata['x']

    return peaks, pred_labels

def peak_agg(cluster_feats, pred_labels, config, tau, K):
    clst_labels, counts = pred_labels.unique(sorted=True, return_counts=True)
    assert torch.equal(clst_labels,
                       torch.arange(cluster_feats.shape[0], device=clst_labels.device))
    print(f'Peaks before: {cluster_feats.shape[0]}')
    # Create graph
    sims, nbrs = knn_faiss(cluster_feats, K)
    ji = jaccard_index(nbrs,
                       torch.zeros_like(sims, device=sims.device).float(),
                       1.)
    sims = (1 - config['SIM_LAMBDA']) * ji + config['SIM_LAMBDA'] * sims
    spmat = knns2spmat(sims, nbrs)
    indices, values = sparse_mx_to_indices_values(spmat)
    peaks_g = Data(x=clst_labels, edge_index=indices.contiguous()).to(cluster_feats.device)
    peaks_g['counts'] = counts
    peaks_g['raw_affine'] = values

    # Filter by edge connection score and by cluster count
    edge_conn_filter = peaks_g['raw_affine'] > tau
    src_cnt = peaks_g['counts'][peaks_g.edge_index[0]]
    dst_cnt = peaks_g['counts'][peaks_g.edge_index[1]]
    clst_cnt_filter = src_cnt < dst_cnt

    edge_mask = torch.logical_and(edge_conn_filter, clst_cnt_filter)
    peaks_g.edge_index = peaks_g.edge_index[:, edge_mask]
    peaks_g['raw_affine'] = peaks_g['raw_affine'][edge_mask]

    # Tree generation
    tree_peaks_g = tree_generation(peaks_g, cluster_feats.device)

    # Propagate cluster labels from the most populated
    rev_g = Data(x=peaks_g.x, edge_index=tree_peaks_g.edge_index.flip([0]))
    g = to_dgl(rev_g, cluster_feats.device)

    def message_func(edges):
        return {'mlb': edges.src['x']}

    def reduce_func(nodes):
        return {'x': nodes.mailbox['mlb'][:, 0]}

    node_order = [nids.to(g.device) for nids in dgl.traversal.topological_nodes_generator(g, reverse=False)]
    g.prop_nodes(node_order, message_func, reduce_func) # propagate labels

    # Get final cluster labels
    final_clst_labels = g.ndata['x']
    print(f'Peaks after: {len(final_clst_labels.unique())}')
    return final_clst_labels[pred_labels]
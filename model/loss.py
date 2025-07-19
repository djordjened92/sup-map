import torch
import torch.nn.functional as F
from utils import knns2spmat, jaccard_index, z_outlier_detection
from torch_sparse import SparseTensor
from torch_scatter import scatter

def mse_adj_loss(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    # Find adjacency matrix
    nbrs = nbrs[:, 1:].long()
    adj_mat = (labels[nbrs] == labels[:, None]).long()

    # Calculate similarities
    out_sims = inf_result[nbrs]@inf_result[:, :, None]
    out_sims = out_sims.squeeze()

    aux_loss = (torch.pow(1 - out_sims, 2) * adj_mat).sum() / adj_mat.sum()

    return aux_loss

def modularity(inf_result, nbrs, labels, switch_points, nbrs_mask=None):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    # Find adjacency matrix
    nbrs = nbrs[:, 1:].long()
    adj_mat = (labels[nbrs] == labels[:, None]).long()

    # Calculate similarities
    out_sims = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    out_sims = out_sims.squeeze()

    if nbrs_mask is not None:
        adj_mat = adj_mat * nbrs_mask

    if switch_points is not None:
        for i, sp in enumerate(switch_points):
            adj_mat[i, sp:] = 0
            out_sims[i, sp:] = 0

    # calculate input and output degree for each node
    w_out = out_sims.sum(dim=1)
    w_in = torch.zeros(nbrs.shape[0], device=inf_result.device)
    w_in_scat = scatter(out_sims.reshape(-1), nbrs.reshape(-1))
    w_in[:w_in_scat.shape[0]] = w_in_scat # if some of nodes doesn't appear as neighbour
    w_agg = (w_in[nbrs] * w_out[:, None]) / out_sims.sum()

    modularity = (out_sims - w_agg) * adj_mat # filter positive samples
    modularity = modularity / out_sims.sum()

    return modularity

def modularity2(inf_result, nbrs):
    '''
    DGClustering inspired loss
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''

    # Find adjacency matrix
    nbrs = nbrs[:, 1:].long()
    adj_mat = torch.zeros((nbrs.shape[0], nbrs.shape[0]), device=inf_result.device)
    adj_mat = adj_mat.scatter_(1, nbrs, 1)

    # calculate input and output degree for each node
    w_out = adj_mat.sum(dim=1)
    w_in = adj_mat.sum(dim=0)
    w_agg = (w_out[:, None] * w_in[None]) / adj_mat.sum()

    modularity = (adj_mat - w_agg) * adj_mat # filter positive samples
    modularity = modularity@(inf_result@inf_result.T)
    modularity = torch.trace(modularity) / adj_mat.sum()

    return modularity

def mod_contrast_loss(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    # Find adjacency matrix
    nbrs = nbrs[:, 1:].long()
    adj_mat = (labels[nbrs] == labels[:, None]).long()

    # Calculate similarities
    temperature = 0.1
    out_sims = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    out_sims /= temperature
    out_sims = out_sims.squeeze()

    # calculate input and output degree for each node
    exp_sims = torch.exp(out_sims)
    w_out = exp_sims.sum(dim=1)
    w_in = torch.zeros(nbrs.shape[0], device=inf_result.device)
    w_in_scat = scatter(exp_sims.reshape(-1), nbrs.reshape(-1))
    w_in[:w_in_scat.shape[0]] = w_in_scat
    log_prob = out_sims + torch.log(1 / w_out[:, None] + 1 / w_in[nbrs])

    return (log_prob * adj_mat) / adj_mat.sum()

def sup_con_loss(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    # Find adjacency matrix
    nbrs = nbrs[:, 1:].long()
    adj_mat = (labels[nbrs] == labels[:, None]).long()

    # Calculate similarities
    temperature = 0.1
    out_sims = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    out_sims /= temperature
    out_sims = out_sims.squeeze()

    # calculate input and output degree for each node
    exp_sims = torch.exp(out_sims)
    log_prob = out_sims - torch.log(exp_sims.sum(dim=1, keepdim=True))

    loss = - (log_prob * adj_mat).sum() / adj_mat.sum()

    return loss

def avg_internal_degree(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    nbrs = nbrs[:, 1:].long()
    W = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    W = W.squeeze()

    _, labels, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    L = labels[nbrs]
    int_mask = (L == labels[:, None]).int()

    internal_edges = (W * int_mask).mean(dim=-1, keepdim=True)
    int_agg = scatter(internal_edges, labels, dim=0, reduce='mean')

    return int_agg

def internal_degree_mask(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    nbrs = nbrs[:, 1:].long()
    W = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    W = W.squeeze()

    # Remap labels to the range [0, len(labels) - 1]
    _, labels, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    L = labels[nbrs]
    int_mask = (L == labels[:, None]).int()

    # Calculate internal degree per label
    W_int = W * int_mask
    out_edges_w = W_int.sum(dim=-1)
    in_edges_w = torch.zeros(nbrs.shape[0], device=inf_result.device)
    in_edges_w_scat = scatter(W_int.reshape(-1), nbrs.reshape(-1), reduce='sum')
    in_edges_w[:in_edges_w_scat.shape[0]] = in_edges_w_scat
    total_w = out_edges_w + in_edges_w

    # Find total (in and out) edges count per node
    nbr_counts_out = torch.zeros(nbrs.shape[0], device=inf_result.device, dtype=nbrs.dtype)
    nbr_values, nbr_counts = torch.unique(nbrs, return_counts=True)
    # torch.unique only is not enough for the case when some node doesn't appear as nbr
    nbr_counts_out.scatter_(0, nbr_values, nbr_counts) # count in edges
    nbr_counts_out += nbrs.shape[1] # add out edges
    avg_per_node = total_w / nbr_counts_out
    avg_per_cluster = scatter(avg_per_node, labels, reduce='mean')
    int_deg_exp = avg_per_cluster[labels] # expand avg values to all labels

    # Caluclate std per cluster
    squared_diff = (avg_per_node - int_deg_exp) ** 2
    var_per_cluster = scatter(squared_diff, labels, reduce='mean')
    std_per_cluster = torch.sqrt(var_per_cluster)

    # Create mask, criteria that node internal degree is lower than average for the cluster
    labels_mask = avg_per_node < int_deg_exp  - std_per_cluster[labels]# dim=len(labels)
    nbrs_mask = labels_mask[nbrs] # shape=nbrs.shape
    nbrs_mask = torch.logical_or(nbrs_mask, labels_mask[:, None])
    return nbrs_mask

def avg_max_ext_weight(init_feats, inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    nbrs = nbrs[:, 1:].long()
    W_init = torch.clamp(init_feats[nbrs]@init_feats[:, :, None], 0.)
    W = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    W_diff = (torch.clamp(W - W_init, 0.)).squeeze()

    _, labels, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    L = labels[nbrs]
    ext_mask = (L != labels[:, None]).int()

    ext_w, _ = (W_diff * ext_mask).max(dim=-1, keepdim=True)
    ext_min_avg = scatter(ext_w, labels, dim=0, reduce='max').mean()

    return ext_min_avg

def avg_max_int_weight(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    nbrs = nbrs[:, 1:].long()
    W = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    W = W.squeeze()

    _, labels, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    L = labels[nbrs]
    int_mask = (L == labels[:, None]).int()

    int_w, _ = (W * int_mask).max(dim=-1, keepdim=True)
    int_max_avg = scatter(int_w, labels, dim=0, reduce='max').mean()

    return int_max_avg

def avg_max_int_degree(inf_result, nbrs, labels):
    '''
    inf_result - features matrix of shape Nxk
    nbrs - indices matrix of shape Nxk
    labels - labels array of len N
    '''
    nbrs = nbrs[:, 1:].long()
    W = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    W = W.squeeze()

    _, labels, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    L = labels[nbrs]
    int_mask = (L == labels[:, None]).int()

    int_w = (W * int_mask).sum(dim=-1, keepdim=True) / (int_mask.sum(dim=-1, keepdim=True) + 1e-9)
    int_max_avg = scatter(int_w, labels, dim=0, reduce='max').mean()

    return int_max_avg

def node_density_loss(inf_result, nbrs, labels):
    nbrs = nbrs[:, 1:].long()
    A = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    A = A.squeeze()

    _, labels, clust_size = torch.unique(labels, return_inverse=True, return_counts=True)
    L = labels[nbrs]
    int_mask = (L == labels[:, None]).int()
    ext_mask = 1 - int_mask

    int_agg = A * int_mask
    ext_agg = A * ext_mask

    contrast_loss = torch.clamp(0.5 - int_agg, 0.).mean(dim=-1) + torch.clamp(ext_agg - 0.5, 0.).mean(dim=-1)
    contrast_loss = scatter(contrast_loss, labels, dim=0, reduce='mean').mean()

    return contrast_loss
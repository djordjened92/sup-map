import gc
import faiss
import torch
import random
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec

def batch_search(index, query, k, bs, verbose=False):
    n = len(query)
    sims = torch.zeros((n, k), dtype=torch.float32).to(query.device)
    nbrs = torch.zeros((n, k), dtype=torch.int32).to(query.device)

    for sid in tqdm(range(0, n, bs), desc="faiss searching...", disable=not verbose):
        eid = min(n, sid + bs)
        sims[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], k)

    return sims, nbrs

def knn_faiss(
    feats,
    k
):
    feats = feats.float()
    size, dim = feats.shape
    index = faiss.IndexFlatIP(dim)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(feats)
    
    sims, nbrs = batch_search(index, feats, k, bs=1024)

    # release the resources
    index.reset()
    del index
    del res
    gc.collect()

    return sims, nbrs

def z_outlier_detection(sims, nbrs, window_size):
    # detect outlier z-score
    delta_p = sims[:, :-1] - sims[:, 1:]
    z = torch.zeros_like(sims, device=sims.device)
    for j in range(delta_p.shape[1] - window_size, -1, -1):
        mu_test = delta_p[:, j:j + window_size].mean(dim=1)
        mu_ref = delta_p[:, j:].mean(dim=1)
        sigma_ref = delta_p[:, j:].std(dim=1)
        q = j + (window_size + 1) // 2
        z[:, q] = torch.abs(mu_test - mu_ref) / sigma_ref
    q_star = torch.argmax(z, dim=1)

    # Add self loops after the switch point
    # In knn2spmat they will be ignored on graph creation
    for i, switch_point in enumerate(q_star):
        nbrs[i, switch_point:] = i

    return nbrs, q_star

def knns2spmat(sims, nbrs, th_sim=-np.Inf):
    # convert knns to symmetric sparse matrix
    row, col = torch.where(sims >= th_sim)

    # remove the self-loop
    idxs = torch.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    n = sims.shape[0]
    
    spmat = SparseTensor(row=row.long(), col=col.long(), value=data,
                        sparse_sizes=(n, n))
    return spmat

def sparse_mx_to_indices_values(sparse_mx):
    row, col, val = sparse_mx.coo()
    indices = torch.cat((row[None], col[None]), dim=0)
    return indices, val

def adjacency_matrix(labels, device):
    n = labels.shape[0]
    self_conn = torch.eye(n, device=device).long()

    # Create pure adjacency matrix
    adj_matrix = (labels[:, None] == labels).long() - self_conn

    return adj_matrix

def community2spmat(labels, device):
    n = labels.shape[0]
    self_conn = torch.eye(n, device=device).long()

    # Create pure adjacency matrix
    community_matrix = (labels[:, None] == labels).long() - self_conn
    row, col = torch.where(community_matrix)
    val = torch.ones((community_matrix.sum()), device=device)
    spmat = SparseTensor(row=row.long(), col=col.long(), value=val,
                        sparse_sizes=(n, n))

    # Create indicator matrix of negative neighbours
    neg_community_matrix = (labels[:, None] != labels).long()
    row, col = torch.where(neg_community_matrix)
    val = torch.ones((neg_community_matrix.sum()), device=device)
    neg_spmat = SparseTensor(row=row.long(), col=col.long(), value=val,
                        sparse_sizes=(n, n))
    return spmat, neg_spmat

def filter_neighbours(nbrs_2, nbrs):
    # Inspired by the Ada-NETS implementation of struct space projection
    # https://github.com/damo-cv/Ada-NETS/blob/main/tool/struct_space.py,
    # keep neighbours which contain each other in the neighbourhood
    N = nbrs_2.shape[0]
    nbr_ids = torch.arange(N, device=nbrs_2.device)
    nbrs_2_eq = nbrs_2 == nbr_ids[:, None, None]
    mask = torch.any(nbrs_2_eq, dim=-1)

    # free up the memory
    del nbrs_2_eq
    del nbr_ids

    fill_nulls = torch.arange(nbrs.numel(), device=nbrs.device).view(nbrs.shape) + N
    nbrs = torch.where(mask, nbrs, fill_nulls)
    return nbrs

def jaccard_index(nbrs, sims, ji_lambda, nbrs_bounds=None):
    N, k = nbrs.shape
    s = torch.zeros_like(nbrs, dtype=torch.float32, device=nbrs.device)
    for i, curr_nbrs_orig in enumerate(nbrs):
        curr_nbrs = curr_nbrs_orig.clone()

        # Calculate intersection
        curr_nbrs_2 = nbrs[curr_nbrs] # copy of tensor, since curr_nbrs is torch.tensor!

        if nbrs_bounds is not None:
            curr_nbrs_2[nbrs_bounds[i]:, :] = -1
            curr_nbrs[nbrs_bounds[i]:] = N
            for j, valid_nbrs in enumerate(curr_nbrs_2[:nbrs_bounds[i]]):
                valid_nbrs[nbrs_bounds[curr_nbrs[j]]:] = N + j + 1

        curr_nbrs_eq = curr_nbrs_2[..., None] == curr_nbrs[None, None, :]
        sims_diff = torch.abs(sims[curr_nbrs_orig][..., None] - sims[i][None, None, :])
        sims_diff *= curr_nbrs_eq.float()
        sims_diff = sims_diff.sum(dim=(-2, -1))

        nbrs_intersection = torch.any(curr_nbrs_eq, dim=-2).sum(dim=-1) - 1

        # Calculate union
        if nbrs_bounds is not None:
            nbrs_union = (nbrs_bounds[curr_nbrs_orig] + nbrs_bounds[i]).to(nbrs.device)
        else:
            nbrs_union = torch.fill(torch.empty(nbrs_intersection.shape, device=nbrs.device),
                                    2 * k)
        nbrs_union -= nbrs_intersection
        ji = nbrs_intersection / nbrs_union
        ji = ji.nan_to_num()

        d = 1 - sims_diff / nbrs_intersection
        d = d.nan_to_num()

        s[i, :] = ji_lambda * ji + (1 - ji_lambda) * d

        # free up the gpu memory
        del curr_nbrs_2
        del curr_nbrs_eq
        del nbrs_intersection
        del nbrs_union

    return s
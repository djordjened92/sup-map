import torch
from torch.nn.functional import one_hot

# create smart teleportation flow matrix and flow distribution
def mkSmartTeleportationFlow(A, T, e_v, w_tot, alpha = 0.15, iter = 1000):
    # calculate the flow distribution with a power iteration
    p = e_v
    for _ in range(iter):
        p = alpha * e_v + (1-alpha) * p @ T
    
    # make the flow matrix for minimising the map equation
    F = alpha * A / w_tot + (1-alpha) * (p.T * T)
    
    return F, p

def map_eq_pooling(inf_result, nbrs, labels):
    eps = 1e-10

    # One-hot labels
    s = one_hot(labels.argsort()).float()

    # Create adjacancy matrix
    nbrs = nbrs[:, 1:].long()
    adj = torch.zeros((nbrs.shape[0], nbrs.shape[0]), device=inf_result.device)
    adj = adj.scatter_(1, nbrs, 1)

    # Calculate Transition matrix
    out_sims = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    out_sims = out_sims.squeeze()
    w = torch.zeros((nbrs.shape[0], nbrs.shape[0]), device=inf_result.device)
    w = w.scatter_(1, nbrs, out_sims)
    w_tot = w.sum()
    T = torch.nan_to_num(w / w.sum(dim=1, keepdim=True), nan = 0.0)

    # distribution according to nodes' in-degrees
    e_v = w.sum(dim=0, keepdim=True) / w_tot

    F, p = mkSmartTeleportationFlow(adj, T, e_v, w_tot)

    # this term is constant, so only calculate it once
    p_log_p = torch.sum(p * torch.log2(p + eps))

    C = s.T @ F @ s
    diag_C = torch.diag(C)

    q   = 1.0 - torch.trace(C)
    q_m = torch.sum(C, dim = 1) - diag_C
    m_exit = torch.sum(C, dim = 0) - diag_C
    p_m = q_m + torch.sum(C, dim = 0)

    codelength = torch.sum(q * torch.log2(q + eps)) \
                - torch.sum(q_m * torch.log2(q_m + eps)) \
                - torch.sum(m_exit * torch.log2(m_exit + eps)) \
                - p_log_p \
                + torch.sum(p_m * torch.log2(p_m + eps))
    return codelength
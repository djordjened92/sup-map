import torch
from torch_scatter import scatter

def smart_teleportation(weights, num_of_nodes, nbrs, tau, iterations=1000, prob_delta=5e-6):
    # The initial visit frequency estimation is 1/n
    p = torch.ones(num_of_nodes, device=nbrs.device) * (1 / num_of_nodes)

    for i in range(iterations):
        prob_update = p[:, None] * weights * (1 - tau)
        w_in = torch.zeros(num_of_nodes, device=nbrs.device)
        w_in_scat = scatter(prob_update.reshape(-1), nbrs.reshape(-1), reduce='sum')
        w_in[:w_in_scat.shape[0]] = w_in_scat # if some of nodes doesn't appear as neighbour

        # Teleportation of probability
        p_new = w_in + tau / num_of_nodes

        # Check successive probability delta
        delta = (p_new - p).abs().sum().item()
        if delta < prob_delta:
            break

        p = p_new
        # print(p.sum())

    # print(f'Teleportation iterations: {i}')

    return p

def module_exit_prob(weights, num_of_nodes, nbrs, labels, tau=0.1):
    '''
    The implementation of exit probability for modules
    according to the section B. Directed weighted networks
    in https://arxiv.org/pdf/0906.1405
    '''
    p = smart_teleportation(weights, num_of_nodes, nbrs, tau)

    _, l_indices, module_size = torch.unique(labels, return_inverse=True, return_counts=True) # count nodes per label and map it to [0, len-1]
    module_visit_rate = scatter(p, l_indices, reduce='sum')
    teleport_part = module_visit_rate * tau * (num_of_nodes - module_size) / num_of_nodes

    L = l_indices[nbrs]
    out_module_mask = (L != l_indices[:, None]).int()
    out_of_module = p[:, None] * weights * out_module_mask
    out_of_module = out_of_module.sum(dim=1) * (1 - tau)
    out_of_module = scatter(out_of_module, l_indices, reduce='sum')

    me_prob = teleport_part + out_of_module

    return me_prob, module_visit_rate, p

def map_eq_loss(inf_result, nbrs, labels):
    '''
    The implementation of The Map Equation(https://arxiv.org/pdf/0906.1405)
    according to Equation 4. We implement version without
    the third term which is total entropy of random node
    visit independent of the partitioning. 
    '''
    if(torch.any(torch.isnan(inf_result))):
        print(inf_result)

    num_of_nodes = nbrs.shape[0]
    nbrs = nbrs[:, 1:].long() # remove self links

    # Calculate similarities
    out_sims = torch.clamp(inf_result[nbrs]@inf_result[:, :, None], 0.)
    out_sims = out_sims.squeeze()
    out_sims /= out_sims.sum(dim=1, keepdim=True) + torch.finfo().eps

    # mep - vector of exit probabilities per module, the length is number of modules
    # mvr - vector of visit probabilities per module, the length is number of modules
    mep, mvr, p = module_exit_prob(out_sims, num_of_nodes, nbrs, labels)

    # Loss calculation
    codelength = mep.sum() * torch.log2(mep.sum() + torch.finfo().eps) \
                 - 2 * (mep * torch.log2(mep + torch.finfo().eps)).sum() \
                 + ((mep + mvr) * torch.log2(mep + mvr + torch.finfo().eps)).sum()
    return codelength
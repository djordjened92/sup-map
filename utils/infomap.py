from time import time
import numpy as np
from tqdm import tqdm
import infomap

def face_cluster(nbrs,
                 sims,
                 flow_model='undirected',
                 regularized=False,
                 regularization_strength=0.,
                 no_self_links=False,
                 silent=True):
    sims = sims / np.sum(sims, axis=1, keepdims=True)
    t = time()
    single, links, weights = [], [], []
    for i, nbrs_i in enumerate(nbrs):
        count = 0
        for idx, j in enumerate(nbrs_i):
            if i == j:
                pass
            else:
                count += 1
                links.append((i, j))
                weights.append(sims[i, idx])
        if count == 0:
            single.append(i)

    info = infomap.Infomap("--two-level",
                           flow_model=flow_model,
                           regularized=regularized,
                           regularization_strength=regularization_strength,
                           no_self_links=no_self_links,
                           silent=silent)
    for (i, j), sim in tqdm(zip(links, weights)):
        _ = info.addLink(i, j, sim)
    del links
    del weights

    info.run(seed=100)

    lb2idx = {}
    idx2lb = {}
    for node in info.iterTree():
        if node.moduleIndex() not in lb2idx:
            lb2idx[node.moduleIndex()] = []
        lb2idx[node.moduleIndex()].append(node.physicalId)

    for k, v in lb2idx.items():
        if k == 0:
            lb2idx[k] = v[2:]
            for u in v[2:]:
                idx2lb[u] = k
        else:
            lb2idx[k] = v[1:]
            for u in v[1:]:
                idx2lb[u] = k

    lb_len = len(lb2idx)
    if len(single) > 0:
        for k in single:
            if k in idx2lb:
                continue
            idx2lb[k] = lb_len
            lb2idx[lb_len] = [k]
            lb_len += 1
    print('time cost of FaceMap: {:.2f}s'.format(time() - t))

    idx_len = max(list(idx2lb.keys()))
    pred_labels = np.zeros(idx_len + 1) - 1
    for k, v in idx2lb.items():
        pred_labels[k] = v

    return pred_labels
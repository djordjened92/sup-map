import torch
from torch_scatter import scatter
import math

class EntropyNegativeSampler:
    def __init__(
        self,
        tb_writer,
        neg_multiplier=3.0,
        pool_factor=10,
        alpha=0.2,
        delta_start=0.15,
        delta_end=0.0,
        anneal_frac=0.7,
        eps=1e-6,
    ):
        """
        neg_multiplier : max negatives as fraction of real edges
        pool_factor    : how many random pairs to sample (relative to E)
        alpha          : negative edge weight scale
        delta_start    : initial entropy slack
        delta_end      : final entropy slack
        anneal_frac    : fraction of training used for annealing
        """
        self.neg_multiplier = neg_multiplier
        self.pool_factor = pool_factor
        self.alpha = alpha
        self.delta_start = delta_start
        self.delta_end = delta_end
        self.anneal_frac = anneal_frac
        self.eps = eps
        self.tb_writer = tb_writer

        self.H_ref = None  # initialized lazily

    def _current_delta(self, progress):
        """progress ∈ [0, 1]"""
        if progress >= self.anneal_frac:
            return self.delta_end
        t = progress / self.anneal_frac
        return self.delta_start * (1 - t) + self.delta_end * t

    @torch.no_grad()
    def _compute_entropy(self, out, edge_index):
        row, col = edge_index
        w = (out[row] * out[col]).sum(dim=-1).clamp(min=self.eps)

        S = scatter(w, row, dim=0, dim_size=out.size(0))
        p = w / S[row]

        H = scatter(-p * torch.log(p), row, dim=0, dim_size=out.size(0))
        return H, S, w

    def sample(
        self,
        out,
        edge_index,
        labels,
        epoch,
        num_epochs
    ):
        """
        out        : (N, D) normalized embeddings
        edge_index: (2, E) base edges
        labels     : (N,)
        """
        device = out.device
        N = out.size(0)
        E = edge_index.size(1)

        # ---- entropy on base graph ----
        H, S, w_pos = self._compute_entropy(out, edge_index)

        self.tb_writer.add_scalar("Train/mean_entropy", H.mean().item(), epoch)
        self.tb_writer.add_scalar("Train/max_entropy", H.max().item(), epoch)

        if self.H_ref is None:
            self.H_ref = H.mean().item()

        progress = epoch / max(1, num_epochs - 1)
        delta = self._current_delta(progress)
        H_max = self.H_ref * (1.0 + delta)

        # ---- candidate pool ----
        num_pool = int(E * self.pool_factor)
        pool = torch.randint(0, N, (2, num_pool), device=device)

        is_neg = labels[pool[0]] != labels[pool[1]]
        pool = pool[:, is_neg]

        if pool.numel() == 0:
            return None

        i, j = pool

        # cosine similarity (ranking only)
        sim = (out[i] * out[j]).sum(dim=-1)

        S_i = S[i]
        if i.numel() == 0:
            return None

        # ---- negative edge weight ----
        out_deg = scatter(
            torch.ones_like(w_pos),
            edge_index[0],
            dim=0,
            dim_size=N
        )[i]

        mean_pos = S_i / (out_deg + self.eps)
        mean_pos[out_deg == 0] = self.H_ref
        w_neg = torch.clamp(self.alpha * mean_pos, min=self.eps)

        # ---- entropy update test (NUMERICALLY STABLE) ----
        S_new = S_i + w_neg

        ratio = (S_i / S_new).clamp(min=self.eps, max=1.0)
        p_new = (w_neg / S_new).clamp(min=self.eps)

        H_new = (
            ratio * H[i]
            - ratio * torch.log(ratio)
            - p_new * torch.log(p_new)
        )

        # ---- FINAL NaN / INF GUARD ----
        finite = torch.isfinite(H_new)
        H_new = H_new[finite]
        i = i[finite]
        j = j[finite]
        sim = sim[finite]

        if i.numel() == 0:
            return None

        valid = H_new <= H_max
        if valid.sum() == 0:
            return None

        i = i[valid]
        j = j[valid]
        sim = sim[valid]

        # ---- hardness ranking ----
        max_neg = int(E * self.neg_multiplier)
        k = min(max_neg, sim.numel())

        _, idx = torch.topk(sim, k=k)
        neg_edge_index = torch.stack([i[idx], j[idx]], dim=0)

        return neg_edge_index

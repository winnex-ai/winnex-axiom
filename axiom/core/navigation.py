"""
AxiomNavigation — PiPrime π-Based Candidate Navigation
=======================================================
Explores threat space using K orthonormal anchors indexed by π and primes.
Powered by the same von Neumann entropy estimation as AxiomCore.

Navigation is deterministic — same query always returns same anchors.
"""

import math, numpy as np
from numpy.linalg import qr


def _sieve(n):
    is_p = [True] * (n * 10)
    is_p[:2] = [False, False]
    for i in range(2, int((n * 10) ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i:n * 10:i] = [False] * ((n * 10 - 1 - i * i) // i + 1)
    return [i for i in range(2, n * 10) if is_p[i]][:n]


class AxiomNavigation:
    """
    PiPrime π-based navigation for threat candidate exploration.

    Parameters:
      n_anchors: number of orthogonal anchors (default: 8, from π × primes)
      d_model: embedding dimension (default: 384)

    The navigation explores the search space and generates candidates
    for AxiomCore to score. Fully deterministic.
    """

    def __init__(self, n_anchors=8, d_model=384, seed=42):
        self.K = n_anchors
        self.d_model = d_model
        self.rng = np.random.RandomState(seed)
        self.primes = _sieve(n_anchors)
        self.pi_primes = [math.pi * p for p in self.primes]
        self.anchors = None
        self.ap = None
        self.anchor_scores = np.zeros(n_anchors)
        self.anchor_counts = np.zeros(n_anchors, dtype=np.int32)
        self.D_int = None
        self._built = False

    def build(self, embeddings):
        n, d = embeddings.shape
        self.d_model = d
        sample = embeddings[:min(n, 10000)]

        # SVD for intrinsic dimension
        from . import estimate_intrinsic_dim
        self.D_int = estimate_intrinsic_dim(sample)

        # Principal components as seeds
        U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(0), full_matrices=False)
        Vt = Vt.astype(np.float32)

        seeds = [embeddings.mean(0).astype(np.float32)]
        for i in range(1, min(self.K, len(Vt))):
            pi_factor = self.pi_primes[i] / self.pi_primes[1]
            v = Vt[i].copy().astype(np.float32)
            v += self.rng.randn(d).astype(np.float32) * (1e-4 * pi_factor)
            seeds.append(v)
        while len(seeds) < self.K:
            seeds.append(self.rng.randn(d).astype(np.float32))

        # Gram-Schmidt orthogonalization
        anchors = []
        for v in seeds[:self.K]:
            v = v.astype(np.float64).copy()
            for a in anchors:
                v -= np.dot(v, a.astype(np.float64)) * a
            nrm = np.linalg.norm(v)
            if nrm > 1e-9:
                anchors.append((v / nrm).astype(np.float32))
            else:
                anchors.append(self.rng.randn(d).astype(np.float32))
                anchors[-1] /= max(np.linalg.norm(anchors[-1]), 1e-9)

        self.anchors = np.array(anchors[:self.K], dtype=np.float32)

        # π-weighted anchor potentials
        self.ap = np.array([
            1.0 + 0.1 * max(0, self.D_int - 1.0) * math.log(i + 2)
            for i in range(self.K)
        ], dtype=np.float64)
        self.ap /= max(self.ap.sum(), 1e-10)
        self._built = True
        return self

    def potential(self, anchor_idx, query, temperature=1.0):
        """U(a) = 0.7·sim(a,q)/T + 0.3·(-0.1·repulsion(a, anchors))"""
        a = self.anchors[anchor_idx]
        q = query.astype(np.float32).flatten()
        q /= max(np.linalg.norm(q), 1e-10)
        sim = float(np.dot(a, q))
        rep = 0.0
        for i in range(self.K):
            if i == anchor_idx:
                continue
            d = float(np.linalg.norm(a - self.anchors[i]))
            rep += self.ap[i] * math.log(1.0 + 1.0 / max(d, 0.01))
        return 0.7 * (sim / temperature) + 0.3 * (-0.1 * rep)

    def navigate(self, query, top_k=3):
        """Deterministic navigation — same query always same anchors."""
        scores = [(i, self.potential(i, query)) for i in range(self.K)]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def explore(self, query, n_candidates=10):
        """Explore: expand from top anchors, generate diverse candidates."""
        top = self.navigate(query, top_k=min(n_candidates, self.K))
        result = []
        for idx, _ in top:
            result.append(self.anchors[idx].copy())
            for j in range(max(1, n_candidates // self.K)):
                p = self.ap[idx] * 0.1 * self.rng.randn(self.d_model).astype(np.float32)
                v = self.anchors[idx] + p
                v /= max(np.linalg.norm(v), 1e-10)
                result.append(v)
                if len(result) >= n_candidates:
                    break
            if len(result) >= n_candidates:
                break
        return np.array(result[:n_candidates], dtype=np.float32)

    def update(self, anchor_idx, feedback):
        """Reinforcement: update anchor score with feedback."""
        old = self.anchor_scores[anchor_idx]
        n = self.anchor_counts[anchor_idx]
        self.anchor_scores[anchor_idx] = (old * n + feedback) / (n + 1)
        self.anchor_counts[anchor_idx] = n + 1

    def stats(self):
        return {"K": self.K, "D_int": round(self.D_int, 1) if self.D_int else None,
                "built": self._built, "orthogonality_error":
                float(np.abs(self.anchors @ self.anchors.T - np.eye(self.K)).max())
                if self._built else None}

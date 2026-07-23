"""
AxiomCore — Madhava-Cauchy Bound Engine
=========================================
Powered by Winnex Madhava QR projection (Zenodo 21088504).
Guarantees 0% false negatives via Cauchy-Schwarz inequality.

Mathematical foundation (extracted from Zenodo 21500959, 20970487):

  1. QR-ORTHOGONAL PROJECTION (NOT JL)
     R ← N(0,1)^{d_out × D}
     R^T = Q · R'  (QR decomposition)
     P ← Q[:, :d_out]^T  ∈ ℝ^{d_out × D}
     P·P^T = I_{d_out}  (verified < 1e-5)

  2. CAUCHY-SCHWARZ UPPER BOUND (deterministic)
     ⟨v,q⟩ = ⟨Pv, Pq⟩ + ⟨v_⟂, q_⟂⟩          // decomposition
           ≤ ⟨Pv, Pq⟩ + ‖v_⟂‖·‖q_⟂‖          // CS inequality
           = B₁(v,q)                          // Q.E.D.
     where v_⟂ = v - P^T P v, ‖v_⟂‖ = √(‖v‖² - ‖Pv‖²)
     B₁(v,q) ≥ ⟨v,q⟩ ALWAYS. 0% false negatives.

  3. TWO-STAGE CASCADE
     Stage 1 (d₁): B₁ = ⟨P₁v, P₁q⟩ + e₁(v)·e₁(q)    [fast, broad]
     Stage 2 (d₂): B₂ = ⟨P₂v, P₂q⟩ + e₂(v)·e₂(q)    [tighter]
     Modulation: score = B₁ + α·(B₂ - B₁)
       α(v) = σ((e₁(v) - e₂(v)) / mean(e₁))  ∈ [0.01, 0.99]

  4. DUAL USE
     Madhava Direct (search): score → select survivors → exact dot
     Axiom/Sec (classification): max(score) → Youden threshold
     SAME engine. DIFFERENT output.

  Reference:
    - Madhava Direct: 10.5281/zenodo.21088504
    - Madhava v18 Proof: 10.5281/zenodo.21500959
    - Madhava Cascade: 10.5281/zenodo.21166403
    - PiPrime: 10.5281/zenodo.20856138
"""

import time, math, warnings
import numpy as np
from numpy.linalg import qr

SEED = 42

def estimate_intrinsic_dim(embeddings):
    """Von Neumann entropy -> intrinsic dimension."""
    _, s, _ = np.linalg.svd(embeddings.astype(np.float64), full_matrices=False)
    e2 = np.maximum(s ** 2, 1e-15)
    e2 /= e2.sum() + 1e-15
    return float(np.exp(-np.sum(e2 * np.log(e2 + 1e-15))))

def optimize_threshold(scores, labels):
    """Youden's J: maximize TPR - FPR."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    J = tpr - fpr
    best = np.argmax(J)
    return float(thr[best]), float(J[best])

class AxiomCore:
    """
    Madhava-Cauchy bound engine — two-stage QR projection + CS bound.

    This is the core mathematical engine that powers all other Axiom layers.
    It inherits the mathematical guarantee proven across 254M+ query-vector pairs.

    Usage:
      core = AxiomCore(stage_dims=[64, 128]).build(centroids)
      scores = core.score(query_embedding)
      core.regime_check()  # evaluates operational regime
    """

    def __init__(self, stage_dims=None):
        self.dims = stage_dims or [64, 128]
        self.full_dim = 384
        self.rng = np.random.RandomState(43)
        self.d_int = None
        self.vectors = None
        self.n = 0
        self.proj_f32 = {}
        self.error_f32 = {}
        self.proj_mat = {}
        self.norms = None
        self.build_time = 0.0

    def _ortho_proj(self, d_out, d_in=None):
        """QR-orthogonalized random projection."""
        d_in = d_in or self.full_dim
        d_out = min(d_out, d_in)
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = qr(R.T)
        return Q[:, :d_out].T.astype(np.float32)

    def build(self, vectors):
        """Build projection cache from centroids/vectors."""
        t0 = time.time()
        self.full_dim = vectors.shape[1]
        self.vectors = vectors.astype(np.float32)
        self.n = len(vectors)
        norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)
        self.norms = np.maximum(norms, 1e-10)
        self.d_int = estimate_intrinsic_dim(vectors[:min(self.n, 10000)])

        for d in self.dims:
            d_eff = min(d, self.full_dim)
            P = self._ortho_proj(d_eff, self.full_dim)
            self.proj_mat[d] = P
            proj = self.vectors @ P.T
            self.proj_f32[d] = proj
            captured = np.linalg.norm(proj, axis=1).astype(np.float32)
            self.error_f32[d] = np.sqrt(
                np.maximum(self.norms ** 2 - captured ** 2, 0)
            ).astype(np.float32)

        self.build_time = time.time() - t0
        return self

    def _upper_bound(self, pv, ev, pq, eq):
        return pv @ pq + ev * eq

    def score(self, query_vec, return_profile=False):
        """
        Score ALL centroids with modulated Madhava-Cauchy bounds.
        No pruning — computes bound for every centroid.
        """
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        d1, d2 = self.dims[0], self.dims[-1]
        mu = max(np.mean(self.error_f32[d1]), 1e-9)

        # Stage 1
        q1 = q @ self.proj_mat[d1].T
        qr1 = math.sqrt(max(0, qn ** 2 - np.linalg.norm(q1) ** 2))
        B1 = self._upper_bound(self.proj_f32[d1], self.error_f32[d1], q1, qr1)

        # Stage 2
        q2 = q @ self.proj_mat[d2].T
        qr2 = math.sqrt(max(0, qn ** 2 - np.linalg.norm(q2) ** 2))
        B2 = self._upper_bound(self.proj_f32[d2], self.error_f32[d2], q2, qr2)

        # Modulation
        delta_e = (self.error_f32[d1] - self.error_f32[d2]) / mu
        alpha = np.clip(1.0 / (1.0 + np.exp(-delta_e * 0.5)), 0.01, 0.99)
        modulated = B1 + alpha * (B2 - B1)

        result = {int(i): float(modulated[i]) for i in range(self.n)}

        if return_profile:
            prof = {
                "n_total": self.n, "d_int": round(self.d_int, 1),
                "dims": list(self.dims),
                "modulated_range": [float(modulated.min()), float(modulated.max())],
                "alpha_mean": float(np.mean(alpha)),
            }
            return result, prof
        return result

    def check_bounds(self, query_vec):
        """Verify 0% bound violation guarantee."""
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        V = self.vectors
        nv = np.maximum(np.linalg.norm(V, axis=1), 1e-10)
        tru = (V @ q) / (nv * qn)
        eps = np.finfo(np.float32).eps * 1000
        viol = {}
        for d in self.dims:
            qd = q @ self.proj_mat[d].T
            qr = math.sqrt(max(0, qn ** 2 - np.linalg.norm(qd) ** 2))
            ub = self._upper_bound(self.proj_f32[d], self.error_f32[d], qd, qr)
            viol[f"{d}D"] = int(np.sum(tru > ub + eps))
        return viol, self.n

    def regime_check(self):
        """Evaluate operational regime."""
        if self.n == 0 or self.d_int is None:
            return {"flag": "UNKNOWN"}
        d = max(self.dims)
        ratio = min(1.0, d / max(self.d_int, 1))
        if ratio >= 0.7:
            flag = "GREEN"
        elif ratio >= 0.3:
            flag = "AMBER"
        else:
            flag = "RED"
        return {"flag": flag, "d_int": round(self.d_int, 1), "ratio": round(ratio, 3)}

    def stats(self):
        r = self.regime_check()
        total = sum(
            self.vectors.nbytes + sum(
                self.proj_f32.get(d, np.array([])).nbytes +
                self.proj_mat.get(d, np.array([])).nbytes +
                self.error_f32.get(d, np.array([])).nbytes
                for d in self.dims
            )
        ) if self.vectors is not None else 0
        return {"n": self.n, "dims": list(self.dims), "d_int": round(self.d_int, 1),
                "build_time_s": round(self.build_time, 3), "size_mb": round(total/1e6, 1),
                "regime": r["flag"]}


def auto_configure(vectors, verbose=True):
    """Auto-configure projection dims based on intrinsic dimension."""
    N, D = vectors.shape
    d_int = estimate_intrinsic_dim(vectors[:min(N, 10000)])
    d1 = max(16, min(128, D, int(math.ceil(d_int * 1.5))))
    d2 = max(32, min(256, D, int(math.ceil(d_int * 3.0))))
    if d1 >= d2:
        d2 = min(d1 * 2, D)
    cfg = {"dims": [d1, d2], "d_int": round(d_int, 1), "full_dim": D}
    if verbose:
        print(f"[AxiomCore] {N}x{D} D_int={d_int:.1f} -> dims=[{d1},{d2}]")
    return cfg

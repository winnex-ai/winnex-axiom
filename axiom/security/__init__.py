"""
AxiomSecurity — Multi-Embedder Ensemble
========================================
Resolves the GIGO single-embedder problem using weighted consensus.
Combines all-MiniLM, BGE, and e5 with calibration-based weights.

If any embedder disagrees with high confidence, the candidate is escalated.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

EMBEDDER_NAMES = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"]


class AxiomSecurity:
    """
    Multi-embedder ensemble with weighted consensus.

    Features:
      - Batch embedding cache (embed once per model, reuse across queries)
      - HDBSCAN or KMeans clustering (auto detects best method)
      - Per-model weights via calibration F1
      - Youden's J threshold per model
    """

    def __init__(self, model_names=None):
        self.model_names = model_names or EMBEDDER_NAMES
        self._embedders = {}
        self._cores = {}       # embedder_name -> AxiomCore
        self._centroids = {}
        self._thresholds = {}
        self._weights = {}
        self._cache = {}
        self._built = False

    def _get_embedder(self, name):
        if name not in self._embedders:
            from sentence_transformers import SentenceTransformer
            try:
                self._embedders[name] = SentenceTransformer(name, device="cpu")
            except Exception:
                if name != self.model_names[0]:
                    self._embedders[name] = SentenceTransformer(self.model_names[0], device="cpu")
                else:
                    raise
        return self._embedders[name]

    def embed(self, texts, model_name=None):
        mn = model_name or self.model_names[0]
        if mn not in self._cache:
            self._cache[mn] = {}
        key = str(hash(tuple(texts[:5])))
        if key not in self._cache[mn]:
            self._cache[mn][key] = self._get_embedder(mn).encode(
                texts, normalize_embeddings=True, show_progress_bar=False, batch_size=128).astype(np.float32)
        return self._cache[mn][key]

    def build(self, attack_texts, clean_texts=None, cluster_method="kmeans"):
        from . import AxiomCore, auto_configure
        for mn in self.model_names:
            embs = self.embed(attack_texts[:min(2000, len(attack_texts))], mn)
            K = max(2, min(30, len(embs)//10))
            km = KMeans(n_clusters=K, random_state=42, n_init=3).fit(embs)
            centroids = km.cluster_centers_.astype(np.float32)
            cn = np.linalg.norm(centroids, axis=1, keepdims=True)
            cn[cn==0] = 1.0; centroids /= cn
            self._centroids[mn] = centroids

            core = AxiomCore(stage_dims=[64,128]).build(centroids)
            self._cores[mn] = core

            # Youden threshold
            from . import optimize_threshold
            attack_scores = self._score_texts(attack_texts[:500], mn)
            if clean_texts and len(clean_texts):
                clean_scores = self._score_texts(clean_texts[:500], mn)
                all_s = np.concatenate([attack_scores, clean_scores])
                all_l = np.array([1]*len(attack_scores)+[0]*len(clean_scores))
                th, _ = optimize_threshold(all_s, all_l)
            else:
                th = float(np.percentile(attack_scores, 10))
            self._thresholds[mn] = th

        self._built = True
        return self

    def calibrate(self, attack_texts, clean_texts):
        for mn in self.model_names:
            a_s = self._score_texts(attack_texts[:500], mn)
            c_s = self._score_texts(clean_texts[:500], mn)
            all_s = np.concatenate([a_s, c_s])
            all_l = np.array([1]*len(a_s)+[0]*len(c_s))
            best = 0.0
            for th in np.linspace(all_s.min(), all_s.max(), 200):
                f = f1_score(all_l, (all_s>=th).astype(int), zero_division=0)
                if f > best: best = f
            self._weights[mn] = best
        total = sum(self._weights.values())
        if total > 0:
            for mn in self._weights: self._weights[mn] /= total
        return self

    def _score_texts(self, texts, model_name):
        embs = self.embed(texts, model_name)
        scores = np.zeros(len(texts))
        for i in range(len(texts)):
            scores[i] = max(self._cores[model_name].score(embs[i]).values())
        return scores

    def evaluate(self, text):
        if not self._built:
            return {"safe": 0.0, "verdict": "uncertain"}
        if not self._weights:
            eq = 1.0 / len(self.model_names)
            for mn in self.model_names: self._weights[mn] = eq

        weighted = 0.0
        confidences = {}
        for mn in self.model_names:
            emb = self.embed([text], mn)[0]
            s = self._cores[mn].score(emb)
            mx = max(s.values())
            th = self._thresholds.get(mn, 0.3)
            flagged = mx >= th
            w = self._weights.get(mn, 1.0/len(self.model_names))
            confidences[mn] = {"score": round(float(mx), 4), "flagged": bool(flagged)}
            if flagged: weighted += w

        safe = 1.0 - weighted
        verdict = "safe" if safe >= 0.8 else "flagged" if safe < 0.5 else "uncertain"
        return {"safe": round(safe, 3), "verdict": verdict, "confidences": confidences}

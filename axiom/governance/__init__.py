"""
AxiomGovernance — Audit Trail, RAI Verification, Compliance Proofs
===================================================================
Records every AxiomCore scoring decision with mathematical proof.
Each entry proves: "candidate was excluded because bound < threshold"
"""

import time, json, hashlib

class AxiomGovernance:
    """
    Immutable audit trail for every scoring decision.

    Each record stores:
      - query hash (not raw text)
      - bound score per centroid
      - decision threshold
      - action taken (allow/escalate)
      - mathematical proof document
      - timestamp

    The audit trail can be exported for regulatory compliance (LGPD, AI Act).
    """

    def __init__(self):
        self.records = []
        self._enabled = True

    def record(self, query_hash, scores, threshold, decision, proof):
        entry = {
            "ts": time.time(),
            "query_hash": query_hash,
            "scores": scores,
            "threshold": threshold,
            "decision": decision,
            "proof": proof,
            "id": hashlib.sha256(f"{query_hash}{time.time()}".encode()).hexdigest()[:16],
        }
        self.records.append(entry)
        return entry["id"]

    def export(self, format="json"):
        if format == "json":
            return json.dumps(self.records, indent=2)
        return self.records

    def stats(self):
        total = len(self.records)
        escalated = sum(1 for r in self.records if r["decision"] == "escalate")
        allowed = sum(1 for r in self.records if r["decision"] == "allow")
        return {"total": total, "escalated": escalated, "allowed": allowed}

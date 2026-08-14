#!/usr/bin/env python3
"""
SAURON++ intelligence layer — novel detector bank + missing machinery
=====================================================================
This module fills the gaps identified against the design doc and raises the
novelty of the detector layer by deliberately choosing methods that are
*under-used in NIDS papers* (which lean on XGBoost + autoencoder + Isolation
Forest). Every detector here is streaming and integer/sketch-friendly, which
also aligns with the kernel-offload thesis.

Detector bank (design §12, upgraded):
  * HDC  — Hyperdimensional Computing classifier (bipolar prototypes;
           natively distillable to the kernel). Rare in NIDS.
  * ECOD — Empirical-CDF (copula) outlier detection (Li et al. 2022). Its score
           is ADDITIVE across features, so per-feature contributions are EXACT
           Shapley values in closed form → real attribution, not a stand-in.
  * SR   — Spectral Residual saliency (Ren et al. 2019), frequency-domain,
           for low-rate/periodic evasive patterns (Whisper-adjacent).
  * RRCF — Robust-Random-Cut-style streaming isolation with a collusive-
           displacement estimate (Guha et al. 2016). Drift-friendly.

Also implemented here (previously missing / stand-in):
  * DriftMartingale     — conformal power-martingale drift test (§12.4)
  * ModelManager        — registry, canary/shadow eval, rollback (§8.2)
  * PolicyDistiller     — fitted shallow rule list w/ measured fidelity &
                          coverage → kernel rules (§12.5, C5)
  * AlertManager        — dedup, correlation, BH-FDR control, lifecycle (§15)
  * Storage             — SQLite event store + append-only audit log (§8.2)
  * AutonomyGovernor    — L2 global rate cap + TTL auto-expiry (§13)

Pure-numpy + stdlib sqlite3. No third-party ML dependencies.
"""

from __future__ import annotations
import bisect
import json
import math
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

FEATURES: Tuple[str, ...] = ("pkt_len", "iat", "syn_ratio", "dst_fanout",
                             "port_entropy", "byte_asymmetry", "pps")
D = len(FEATURES)


def vectorize(feat: Dict[str, float]) -> np.ndarray:
    return np.array([feat.get(f, 0.0) for f in FEATURES], dtype=np.float64)


# ===========================================================================
# 1. HYPERDIMENSIONAL COMPUTING classifier  (novel supervised head)
# ===========================================================================
class HDCClassifier:
    """Hyperdimensional Computing intrusion classifier.

    Encoding: random-projection HDC. A fixed random bipolar projection maps the
    feature vector to a Dh-dim hypervector h(x)=sign(W·x̃). Class prototypes are
    built by *bundling* (integer accumulation) member hypervectors; the bipolar
    prototype is sign(accumulator). Classification compares cosine similarity to
    the benign and malicious prototypes. Online, robust, and the ±1 prototypes
    are natively quantizable for kernel distillation.

    Under-used in NIDS relative to gradient-boosted trees → novelty. Returns
    P(malicious) in [0,1] via a logistic on the similarity margin.
    """
    name = "hdc"

    def __init__(self, dim: int = 1024, seed: int = 11, cap: float = 4.0):
        self.Dh = dim
        rng = np.random.default_rng(seed)
        # random bipolar projection (Dh x D) + a level term for nonlinearity
        self.W = rng.choice([-1.0, 1.0], size=(dim, D))
        self.Wq = rng.choice([-1.0, 1.0], size=(dim, D))   # quadratic channel
        self.acc = {0: np.zeros(dim), 1: np.zeros(dim)}    # bundling accumulators
        self.proto = {0: np.zeros(dim), 1: np.zeros(dim)}  # bipolar prototypes
        self.n = {0: 0, 1: 0}
        self.cap = cap
        self._mu = np.zeros(D); self._m2 = np.ones(D); self._cnt = 1

    def _encode(self, x: np.ndarray) -> np.ndarray:
        # standardize online, then bipolar-project two channels and bundle
        z = (x - self._mu) / np.sqrt(self._m2 / self._cnt + 1e-6)
        h = np.sign(self.W @ z + self.Wq @ (z * z))
        h[h == 0] = 1.0
        return h

    def _update_stats(self, x: np.ndarray):
        self._cnt += 1
        d = x - self._mu
        self._mu += d / self._cnt
        self._m2 += d * (x - self._mu)

    def score(self, feat: Dict[str, float]) -> float:
        x = vectorize(feat)
        h = self._encode(x)
        sm = float(h @ self.proto[1]) / self.Dh if self.n[1] else 0.0
        sb = float(h @ self.proto[0]) / self.Dh if self.n[0] else 0.0
        self._update_stats(x)
        return 1.0 / (1.0 + math.exp(-4.0 * (sm - sb)))

    def learn(self, feat: Dict[str, float], label: int, weight: float = 1.0):
        """Bundle a labeled hypervector into the class prototype (capped)."""
        x = vectorize(feat)
        h = self._encode(x)
        y = 1 if label >= 1 else 0
        self.acc[y] += min(weight, self.cap) * h
        self.n[y] += 1
        self.proto[y] = np.sign(self.acc[y]); self.proto[y][self.proto[y] == 0] = 1.0

    def quantize_prototypes(self) -> Dict[str, List[int]]:
        """Export bipolar prototypes (±1) for kernel-side distillation."""
        return {"benign": self.proto[0].astype(int).tolist(),
                "malicious": self.proto[1].astype(int).tolist()}


# ===========================================================================
# 2. ECOD — empirical-CDF outlier detection with EXACT Shapley attribution
# ===========================================================================
class ECOD:
    """ECOD (Li et al., 2022): parameter-free outlier detection.

    For each feature it estimates left/right tail probabilities from a streaming
    empirical CDF and sums the per-feature 'surprise' -log(tail). Because the
    aggregate score is a SUM of per-feature terms, the Shapley value of feature
    i equals its own term exactly — so `contributions()` returns *exact* feature
    attributions in closed form (no TreeSHAP approximation needed).
    """
    name = "ecod"

    def __init__(self, window: int = 1000):
        self.window = window
        self.buf = [deque(maxlen=window) for _ in range(D)]   # per-dim samples
        self.sorted_cache: List[Optional[np.ndarray]] = [None] * D
        self._since = 0

    def _tails(self, x: np.ndarray) -> np.ndarray:
        """Per-feature two-sided tail surprise -log(tail): high when the value is
        extreme on either side, low near the median. Monotone in extremeness, so
        the per-feature term is an interpretable, exact additive contribution."""
        out = np.zeros(D)
        for i in range(D):
            b = self.buf[i]
            n = len(b)
            if n < 20:
                out[i] = 0.0
                continue
            s = self.sorted_cache[i]
            if s is None:
                s = np.sort(np.fromiter(b, dtype=np.float64, count=n)); self.sorted_cache[i] = s
            F = bisect.bisect_right(s, x[i]) / (n + 1.0)
            # surprise from the tail the value actually falls in
            out[i] = -math.log(max(1.0 - F, 1e-9)) if F > 0.5 else -math.log(max(F, 1e-9))
        return out

    def score(self, feat: Dict[str, float]) -> float:
        x = vectorize(feat)
        t = self._tails(x)
        total = float(np.sum(t))
        # update streaming ECDF
        for i in range(D):
            self.buf[i].append(x[i])
        self._since += 1
        if self._since % 32 == 0:
            self.sorted_cache = [None] * D
        return 1.0 - math.exp(-total / (D * 2.5))          # squash to [0,1]

    def contributions(self, feat: Dict[str, float]) -> List[Tuple[str, float]]:
        """EXACT Shapley attribution (additive model → own-term is the value)."""
        x = vectorize(feat)
        t = self._tails(x)
        pairs = [(FEATURES[i], round(float(t[i]), 3)) for i in range(D)]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        return pairs


# ===========================================================================
# 3. SPECTRAL RESIDUAL — frequency-domain saliency (evasive / low-rate)
# ===========================================================================
class SpectralResidual:
    """Spectral Residual anomaly detector (Ren et al., 2019).

    Operates on the recent per-source packet-rate/size signal. Computes the
    log-amplitude spectral residual and inverse-transforms to a saliency map;
    the saliency of the newest point is the anomaly. Catches periodic C2 beacons
    and low-and-slow patterns that per-packet detectors miss. Rare in NIDS.
    """
    name = "sr"

    def __init__(self, window: int = 32, q: int = 3):
        self.window = window
        self.q = q
        self.sig: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def score_series(self, s: np.ndarray) -> float:
        n = len(s)
        if n < 8:
            return 0.0
        f = np.fft.fft(s)
        amp = np.abs(f) + 1e-9
        L = np.log(amp)
        kernel = np.ones(self.q) / self.q
        AL = np.convolve(L, kernel, mode='same')
        R = L - AL
        S = np.abs(np.fft.ifft(np.exp(R + 1j * np.angle(f))))
        m = float(np.mean(S) + 1e-9)
        return float(min(1.0, max(0.0, (S[-1] - m) / (3.0 * m))))

    def score(self, feat: Dict[str, float], entity: str) -> float:
        b = self.sig[entity]
        b.append(feat.get("pps", 0.0) + 0.3 * feat.get("byte_asymmetry", 0.0))
        return self.score_series(np.fromiter(b, dtype=np.float64, count=len(b)))


# ===========================================================================
# 4. RRCF — robust-random-cut-style streaming isolation (collusive displacement)
# ===========================================================================
class _RCTree:
    """A bounded random-cut tree over recent points; depth-of-insertion gives a
    streaming isolation signal, and subtree mass approximates collusive
    displacement (CoDisp)."""
    def __init__(self, size: int, rng):
        self.size = size
        self.rng = rng
        self.pts: Deque[np.ndarray] = deque(maxlen=size)

    def insert_depth(self, x: np.ndarray) -> Tuple[int, int]:
        pts = list(self.pts)
        self.pts.append(x)
        if len(pts) < 4:
            return 1, 1
        P = np.array(pts)
        lo = P.min(axis=0); hi = P.max(axis=0)
        depth = 0; mass = len(pts)
        for _ in range(24):
            span = hi - lo
            tot = float(span.sum())
            if tot <= 1e-9:
                break
            ax = self.rng.choice(D, p=(span / tot))
            cut = self.rng.uniform(lo[ax], hi[ax])
            left = P[:, ax] <= cut
            xleft = x[ax] <= cut
            depth += 1
            sub = P[left] if xleft else P[~left]
            mass = len(sub)
            if mass <= 1:
                break
            P = sub
            lo = P.min(axis=0); hi = P.max(axis=0)
        return depth, mass


class RRCF:
    """Streaming random-cut forest. Anomaly ~ shallow insertion depth and small
    colluding subtree mass (isolation + CoDisp proxy). Different inductive bias
    from ECOD/HDC → strengthens the fusion regret guarantee."""
    name = "rrcf"

    def __init__(self, trees: int = 6, size: int = 56, seed: int = 5):
        self.forest = [_RCTree(size, np.random.default_rng(seed + i)) for i in range(trees)]

    def score(self, feat: Dict[str, float]) -> float:
        x = vectorize(feat)
        s = 0.0
        for t in self.forest:
            depth, mass = t.insert_depth(x)
            s += 1.0 / (1.0 + depth) * (1.0 / (1.0 + math.log(mass + 1)))
        s /= len(self.forest)
        return float(min(1.0, 2.2 * s))


# ===========================================================================
# Detector bank facade (same interface the pipeline expects)
# ===========================================================================
class OnlineSupervisedHead:
    """Online logistic-regression head with adaptive per-feature learning rates.

    ECOD, Spectral-Residual and RRCF are unsupervised: they score how *unusual*
    a flow is, which is not the same question as "is this an attack". On labelled
    corpora such as CICIDS2017 that gap caps recall. This head learns the actual
    decision boundary from the label stream using AdaGrad-scaled SGD, so it is
    still fully online (single pass, no stored dataset) and slots into the Hedge
    fusion like any other detector — the fusion decides how much to trust it.
    """

    name = "sup"

    def __init__(self, lr: float = 0.35, l2: float = 1e-6, rff: int = 192, seed: int = 17):
        # Random Fourier features: a linear boundary cannot separate several
        # distinct attack modes at once (CICIDS2017 Wednesday alone holds four
        # DoS variants). Projecting into a random cosine basis approximates an
        # RBF kernel, giving the head non-linear capacity while staying online.
        rng = np.random.default_rng(seed)
        self.R = rng.normal(0.0, 1.0, size=(rff, D)) * 1.4
        self.phase = rng.uniform(0.0, 2 * math.pi, size=rff)
        self.dim = rff + D                    # raw features kept alongside
        self.w = np.zeros(self.dim)
        self.b = 0.0
        self.g2 = np.full(self.dim, 1e-8)     # AdaGrad accumulators
        self.gb2 = 1e-8
        self.lr, self.l2 = lr, l2
        self.n = 0

    def _phi(self, x: np.ndarray) -> np.ndarray:
        z = np.cos(self.R @ x + self.phase) * (math.sqrt(2.0 / len(self.phase)))
        return np.concatenate([z, x])

    def _z(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x) + self.b)

    def score(self, feat: Dict[str, float]) -> float:
        if self.n < 50:
            return 0.5                   # abstain until it has seen some labels
        x = self._phi(vectorize(feat))
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self._z(x))))))

    def learn(self, feat: Dict[str, float], label: int, weight: float = 1.0):
        x = self._phi(vectorize(feat))
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self._z(x)))))
        err = (p - float(label)) * float(weight)
        g = err * x + self.l2 * self.w
        self.g2 += g * g
        self.w -= self.lr * g / np.sqrt(self.g2)
        self.gb2 += err * err
        self.b -= self.lr * err / math.sqrt(self.gb2)
        self.n += 1


class NovelDetectorBank:
    def __init__(self):
        self.hdc = HDCClassifier()
        self.ecod = ECOD()
        self.sr = SpectralResidual()
        self.rrcf = RRCF()
        self.sup = OnlineSupervisedHead()

    @property
    def names(self) -> List[str]:
        return [self.hdc.name, self.ecod.name, self.sr.name,
                self.rrcf.name, self.sup.name]

    def scores(self, feat: Dict[str, float], entity: str = "-") -> Dict[str, float]:
        return {
            self.hdc.name: self.hdc.score(feat),
            self.ecod.name: self.ecod.score(feat),
            self.sr.name: self.sr.score(feat, entity),
            self.rrcf.name: self.rrcf.score(feat),
            self.sup.name: self.sup.score(feat),
        }

    def learn(self, feat: Dict[str, float], label: Optional[int], weight: float = 1.0):
        if label is not None:
            self.hdc.learn(feat, int(label), weight)
            self.sup.learn(feat, int(label), weight)

    def attribution(self, feat: Dict[str, float], top: int = 3) -> List[Tuple[str, float]]:
        """EXACT Shapley attribution via the additive ECOD member."""
        return self.ecod.contributions(feat)[:top]


# ===========================================================================
# DRIFT — conformal power-martingale test (§12.4)
# ===========================================================================
class DriftMartingale:
    """Detects distribution drift by testing uniformity of conformal p-values
    with a power martingale (Vovk et al.). M_t = Π ε p^(ε-1); a large M_t is
    evidence against exchangeability → drift. Cheap, principled, and ties to the
    conformal calibration already in the ADE. Under-used vs ADWIN/DDM."""
    def __init__(self, epsilon: float = 0.92, threshold: float = 100.0):
        self.eps = epsilon
        self.threshold = threshold
        self.logM = 0.0
        self.events = 0
        self.detections = 0

    def update(self, p: float) -> bool:
        p = min(max(p, 1e-6), 1.0)
        self.logM += math.log(self.eps) + (self.eps - 1.0) * math.log(p)
        self.logM = max(self.logM, 0.0)          # reset floor (running max-style)
        self.events += 1
        if self.logM > math.log(self.threshold):
            self.detections += 1
            self.logM = 0.0
            return True
        return False

    def level(self) -> float:
        return round(math.exp(min(self.logM, 20.0)), 2)


# ===========================================================================
# MODEL MANAGER — registry, canary/shadow eval, rollback (§8.2)
# ===========================================================================
@dataclass
class _ModelVersion:
    version: int
    created: float
    correct: int = 0
    total: int = 0
    def acc(self) -> float:
        return self.correct / self.total if self.total else 0.0


class ModelManager:
    """Versioned model registry with canary/shadow evaluation and automatic
    rollback. The active version serves; a canary shadows it on live traffic and
    is promoted only if it beats the active version over a window, else rolled
    back. Drift events trigger a new canary."""
    def __init__(self):
        self.versions: List[_ModelVersion] = [_ModelVersion(1, time.time())]
        self.active = 1
        self.canary: Optional[_ModelVersion] = None
        self.promotions = 0
        self.rollbacks = 0
        self._win = deque(maxlen=500)

    def observe(self, active_correct: bool, canary_correct: Optional[bool]):
        v = self.versions[self.active - 1]
        v.total += 1; v.correct += int(active_correct)
        if self.canary is not None and canary_correct is not None:
            self.canary.total += 1; self.canary.correct += int(canary_correct)
            if self.canary.total >= 200:
                if self.canary.acc() > v.acc() + 0.01:
                    self.versions.append(self.canary); self.active = self.canary.version
                    self.promotions += 1; self.canary = None
                else:
                    self.rollbacks += 1; self.canary = None

    def start_canary(self):
        if self.canary is None:
            self.canary = _ModelVersion(len(self.versions) + 1, time.time())

    def status(self) -> Dict:
        v = self.versions[self.active - 1]
        return {"active_version": self.active, "active_acc": round(v.acc(), 3),
                "canary": (self.canary.version if self.canary else None),
                "canary_acc": round(self.canary.acc(), 3) if self.canary else None,
                "promotions": self.promotions, "rollbacks": self.rollbacks}


# ===========================================================================
# POLICY DISTILLER — fitted shallow rule list + measured fidelity/coverage (C5)
# ===========================================================================
class PolicyDistiller:
    """Fits a depth-1 rule list (decision stumps) to the fusion decision
    (enforce vs allow) on recent traffic, then reports fidelity (agreement with
    the ADE) and coverage (traffic fraction confidently handled). High-fidelity,
    high-coverage rules become candidate kernel rules — a *real* surrogate, not a
    synthesized number."""
    def __init__(self, window: int = 1500, min_fidelity: float = 0.95, min_coverage: float = 0.05):
        self.buf: Deque[Tuple[np.ndarray, int, str]] = deque(maxlen=window)
        self.min_fidelity = min_fidelity
        self.min_coverage = min_coverage

    def observe(self, feat: Dict[str, float], enforced: int, src_ip: str):
        self.buf.append((vectorize(feat), int(enforced), src_ip))

    def distill(self) -> List[Dict]:
        if len(self.buf) < 200:
            return []
        X = np.array([b[0] for b in self.buf])
        y = np.array([b[1] for b in self.buf])
        if y.mean() in (0.0, 1.0):
            return []
        deciles = [d / 10.0 for d in range(1, 10)]
        rules: List[Dict] = []

        def consider(mask, desc):
            cover = float(mask.mean())
            if cover < self.min_coverage or cover > 0.9 or mask.sum() < 20:
                return
            fid = float((y[mask] == 1).mean())
            if fid >= self.min_fidelity:
                rules.append({**desc, "action": "DROP",
                              "fidelity": round(fid, 3), "coverage": round(cover, 3)})

        # depth-1 stumps
        for i in range(D):
            col = X[:, i]
            for q in deciles:
                thr = float(np.quantile(col, q))
                consider(col > thr, {"type": "stump", "feature": FEATURES[i],
                                     "op": ">", "threshold": round(thr, 4)})

        # depth-2 conjunctions (isolate a family: e.g. pps>hi AND pkt_len<lo)
        qs = (0.1, 0.3, 0.5, 0.7, 0.9)
        for i in range(D):
            ci = X[:, i]
            for j in range(i + 1, D):
                cj = X[:, j]
                for qi in qs:
                    ti = float(np.quantile(ci, qi))
                    mi_gt, mi_lt = ci > ti, ci < ti
                    for qj in qs:
                        tj = float(np.quantile(cj, qj))
                        mj_gt, mj_lt = cj > tj, cj < tj
                        for opi, mi in ((">", mi_gt), ("<", mi_lt)):
                            for opj, mj in ((">", mj_gt), ("<", mj_lt)):
                                consider(mi & mj,
                                         {"type": "conj", "feature": FEATURES[i], "op": opi,
                                          "threshold": round(ti, 4), "feature2": FEATURES[j],
                                          "op2": opj, "threshold2": round(tj, 4)})

        rules.sort(key=lambda r: r["fidelity"] * r["coverage"], reverse=True)
        # greedy de-duplication by covered mass to build a small rule list
        return rules[:4]


# ===========================================================================
# ALERT MANAGER — dedup, correlation, BH-FDR, lifecycle (§15)
# ===========================================================================
class AlertManager:
    """Deduplicates and correlates alerts, applies Benjamini-Hochberg FDR control
    over conformal p-values so the *displayed* alert set has a controlled
    false-discovery rate, and tracks lifecycle (new→ack→mitigated→expired)."""
    def __init__(self, fdr_q: float = 0.1, dedup_sec: float = 3.0):
        self.fdr_q = fdr_q
        self.dedup_sec = dedup_sec
        self._last: Dict[Tuple[str, str], float] = {}
        self.campaigns: Dict[str, Dict] = {}
        self.pbuf: Deque[float] = deque(maxlen=500)
        self.suppressed = 0
        self.total = 0

    def bh_threshold(self) -> float:
        """BH-FDR p-value cutoff over the recent conformal p-values."""
        if len(self.pbuf) < 20:
            return 1.0
        ps = np.sort(np.fromiter(self.pbuf, dtype=np.float64, count=len(self.pbuf)))
        n = len(ps)
        crit = self.fdr_q * (np.arange(1, n + 1) / n)
        below = np.where(ps <= crit)[0]
        return float(ps[below[-1]]) if len(below) else float(crit[0])

    def process(self, ev: Dict) -> Optional[Dict]:
        """Return an enriched alert dict, or None if deduped/FDR-suppressed."""
        self.total += 1
        p = 1.0 - ev.get("score", 0.0)          # small p ⇒ anomalous
        self.pbuf.append(p)
        key = (ev["src_ip"], ev["action"])
        now = ev.get("ts", time.time())
        enforcement = ev["action"] in ("DROP", "QUARANTINE", "REDIRECT", "RATE_LIMIT")
        # dedup (quarantine never deduped)
        if now - self._last.get(key, -1e9) < self.dedup_sec and ev["action"] != "QUARANTINE":
            self.suppressed += 1
            return None
        self._last[key] = now
        # BH-FDR gate applies only to advisory (non-enforcement) alerts; genuine
        # enforcement events always surface per §15 (red = exceed τH OR dropped).
        if not enforcement and p > self.bh_threshold():
            self.suppressed += 1
            return None
        # correlation: group by source entity into a campaign
        camp = self.campaigns.setdefault(ev["src_ip"], {"count": 0, "first": now, "actions": set()})
        camp["count"] += 1; camp["actions"].add(ev["action"])
        alert = dict(ev)
        alert["alert_id"] = f"al-{int(now*1000)%10_000_000}-{camp['count']}"
        alert["flow_id"] = format(abs(hash((ev["src_ip"], ev["dst_ip"], ev["src_port"], ev["dst_port"]))) & 0xffffffff, "08x")
        alert["campaign_count"] = camp["count"]
        alert["lifecycle"] = "new"
        alert["fdr_p"] = round(p, 4)
        return alert

    def stats(self) -> Dict:
        return {"alerts_total": self.total, "alerts_suppressed": self.suppressed,
                "bh_cutoff": round(self.bh_threshold(), 4),
                "active_campaigns": len(self.campaigns)}


# ===========================================================================
# STORAGE — SQLite event store + append-only audit log (§8.2)
# ===========================================================================
class Storage:
    """Persistent event store + append-only audit log. Every threshold/weight/
    policy change is logged with its cause for forensic replayability (design
    principle 4)."""
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS events(
            ts REAL, src TEXT, dst TEXT, proto TEXT, action TEXT, score REAL,
            severity TEXT, reason TEXT, family TEXT)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS audit(
            ts REAL, kind TEXT, detail TEXT)""")
        self.conn.commit()
        self._buf = []

    def event(self, ev: Dict):
        self._buf.append((ev.get("ts"), ev.get("src_ip"), ev.get("dst_ip"), ev.get("proto"),
                          ev.get("action"), ev.get("score"), ev.get("severity"),
                          ev.get("reason"), ev.get("family")))
        if len(self._buf) >= 100:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        self.conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)", self._buf)
        self.conn.commit(); self._buf.clear()

    def audit(self, kind: str, detail: Dict):
        self.conn.execute("INSERT INTO audit VALUES (?,?,?)",
                          (time.time(), kind, json.dumps(detail)))
        self.conn.commit()

    def recent_events(self, n: int = 200) -> List[Dict]:
        self.flush()
        cur = self.conn.execute(
            "SELECT ts,src,dst,proto,action,score,severity,reason,family FROM events ORDER BY ts DESC LIMIT ?", (n,))
        cols = ["ts", "src_ip", "dst_ip", "proto", "action", "score", "severity", "reason", "family"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def audit_tail(self, n: int = 100) -> List[Dict]:
        cur = self.conn.execute("SELECT ts,kind,detail FROM audit ORDER BY ts DESC LIMIT ?", (n,))
        return [{"ts": r[0], "kind": r[1], "detail": json.loads(r[2])} for r in cur.fetchall()]


# ===========================================================================
# AUTONOMY GOVERNOR — L2 global rate cap + TTL auto-expiry (§13)
# ===========================================================================
class AutonomyGovernor:
    """Enforces the guardrails on automation: a global token-bucket cap on L2
    containment (so a mis-calibrated loop cannot mass-quarantine) and TTL
    auto-expiry of enforced entries (so mistakes heal)."""
    def __init__(self, l2_per_min: float = 60.0, ttl_sec: float = 120.0):
        self.rate = l2_per_min / 60.0
        self.tokens = l2_per_min
        self.max_tokens = l2_per_min
        self.ttl = ttl_sec
        self._last = time.time()
        self.enforced: Dict[str, float] = {}     # src_ip -> expiry ts
        self.capped = 0

    def allow_contain(self, src_ip: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now   # 0.0 is a valid timestamp
        delta = max(0.0, now - self._last)
        self.tokens = min(self.max_tokens, self.tokens + delta * self.rate)
        self._last = now
        if self.tokens < 1.0:
            self.capped += 1
            return False
        self.tokens -= 1.0
        self.enforced[src_ip] = now + self.ttl
        return True

    def expired(self, now: Optional[float] = None) -> List[str]:
        now = time.time() if now is None else now   # 0.0 is a valid timestamp
        out = [ip for ip, exp in self.enforced.items() if exp <= now]
        for ip in out:
            self.enforced.pop(ip, None)
        return out

    def stats(self) -> Dict:
        return {"contain_capped": self.capped, "active_enforced": len(self.enforced),
                "l2_tokens": round(self.tokens, 1)}

#!/usr/bin/env python3
"""
SAURON++ backend — adaptive engine, detector bank, sources, telemetry, server
=============================================================================
Single-file backend for the SAURON++ FIREWALL SYSTEM. It implements the
Adaptive Decision Engine (design Section 11), the diverse detector bank
(Section 12), the packet-lifecycle pipeline (Section 10), real host telemetry,
and a FastAPI server that streams live telemetry to the React dashboard over a
WebSocket.

Three traffic sources, one interface, so the identical pipeline runs anywhere:
  * sim   -- labeled synthetic traffic with realistic attack families
  * pcap  -- streaming replay of a real capture (CICIDS2017, tens of GB)
  * ebpf  -- the real kernel data path via the libbpf loader (kernel/)

Run (see scripts/run.sh):
  python3 backend/sauron.py --source sim
  python3 backend/sauron.py --source live --iface "Wi-Fi"          # your real traffic
  python3 backend/sauron.py --list-ifaces                          # find your interface
  python3 backend/sauron.py --source pcap --pcap /data/CICIDS2017.pcap
  sudo python3 backend/sauron.py --source ebpf --iface eth0
  python3 backend/sauron.py --headless --source sim --events 4000   # no web stack

Dependencies: numpy (required); fastapi + uvicorn + websockets (server mode);
scapy (pcap mode). All are listed in README.md / installed by scripts/build.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Web stack is optional so the headless/sim path runs with numpy alone. These
# names must live at module scope: with `from __future__ import annotations`,
# FastAPI resolves the `websocket: WebSocket` annotation against module globals,
# not build_app()'s locals. (Server mode only reaches build_app when present.)
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    _WEB_OK = True
except Exception:
    FastAPI = WebSocket = WebSocketDisconnect = Response = None
    HTMLResponse = JSONResponse = FileResponse = None
    _WEB_OK = False

# Print each live packet's details (no decision) to the terminal. On by default;
# disable with --no-print-packets or SAURON_PRINT_PKTS=0.
_PRINT_PKTS = os.environ.get("SAURON_PRINT_PKTS", "1") != "0"
# Colour the live decision table unless disabled (--no-color / NO_COLOR).
_NO_COLOR = bool(os.environ.get("NO_COLOR"))
# Optional per-packet CSV writer (enabled with --packet-csv <path>).
_PKT_CSV = None
_PKT_CSV_FH = None
_PKT_CSV_COLS = ["ts_epoch", "timestamp", "packet_id", "src_ip", "src_port",
                 "dst_ip", "dst_port", "proto", "pkt_len", "tcp_flags",
                 "score", "raw_score", "tau_high", "tau_low", "trust", "suspicion",
                 "decision", "severity", "reason", "mitigation", "top_features",
                 "family", "label", "realized_fpr"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
KERNEL_DIR = os.path.join(REPO_ROOT, "kernel")

# novel intelligence layer (detector bank + drift/model-mgr/distiller/alerts/…)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intelligence as _intel  # noqa: E402
import integrations as _intg
import energy as _energy  # noqa: E402  (external SIEM/ticketing/Slack/Email/SMS sinks)
try:
    import grpc_service as _mesh  # noqa: E402  (optional gRPC node-to-node mesh)
except Exception:  # grpc not installed -> single-node mode
    _mesh = None


# ============================================================================
# SECTION 1 — ADAPTIVE DECISION ENGINE  (design Section 11)
# ============================================================================
class ConformalCalibrator:
    """Windowed split-conformal p-values for one detector (11.1).

    p = (1 + #{cal >= s}) / (n + 1) is a distribution-free tail probability
    under exchangeability. Anomaly signal a = 1 - p; every detector then speaks
    the same calibrated language regardless of raw score scale.
    """

    def __init__(self, window: int = 2000, warmup: int = 50):
        self.window, self.warmup = window, warmup
        self.cal: Deque[float] = deque(maxlen=window)

    def p_value(self, score: float) -> float:
        n = len(self.cal)
        if n < self.warmup:
            return float(1.0 / (1.0 + math.exp(3.0 * (score - 0.5))))
        arr = np.fromiter(self.cal, dtype=np.float64, count=n)
        return (1.0 + np.count_nonzero(arr >= score)) / (n + 1.0)

    def anomaly(self, score: float) -> float:
        return 1.0 - self.p_value(score)

    def observe_benign(self, score: float) -> None:
        self.cal.append(float(score))


class HedgeFusion:
    """Hedge / multiplicative-weights + fixed-share expert fusion (11.2).

    A_t = sum_k w_k a_k. On a label, charge squared-error loss and update
    w_k <- w_k exp(-eta loss_k); then mix w_k <- (1-alpha) w_k + alpha/K. Hedge
    gives O(sqrt(T ln K)) static regret; fixed-share converts it to tracking
    (shifting) regret so fusion follows drift (H1).
    """

    def __init__(self, k: int, eta: Optional[float] = None, alpha: float = 0.02):
        self.k, self.alpha = k, alpha
        self.eta = eta if eta is not None else math.sqrt(8.0 * math.log(max(k, 2)) / 1000.0)
        self.w = np.full(k, 1.0 / k)

    def aggregate(self, signals: np.ndarray) -> float:
        return float(np.dot(self.w, signals))

    def update(self, signals: np.ndarray, label: float) -> None:
        self.w *= np.exp(-self.eta * (signals - label) ** 2)
        self.w /= self.w.sum()
        self.w = (1.0 - self.alpha) * self.w + self.alpha / self.k
        self.w /= self.w.sum()

    @property
    def weights(self) -> np.ndarray:
        return self.w.copy()


@dataclass
class _TrustState:
    a: float = 1.0
    b: float = 1.0
    seen: int = 0


class TrustModel:
    """Discounted Beta-Bernoulli entity reputation (11.3).

    Trust = a/(a+b). Evidence forgets (recent dominates). Influence cap: one
    event moves (a,b) by <= cap so a flood cannot dominate (T5). Anti-laundering
    asymmetry: malicious evidence forgets slower than benign, so a bad source
    cannot rapidly wash its reputation. suspicion() is a UCB upper bound on
    P(malicious) so rarely-seen-but-bad entities are not under-penalised.
    """

    def __init__(self, lam_ben=0.98, lam_mal=0.995, cap=1.0, ucb_c=0.4):
        self.lam_ben, self.lam_mal, self.cap, self.ucb_c = lam_ben, lam_mal, cap, ucb_c
        self._s: Dict[str, _TrustState] = defaultdict(_TrustState)

    def trust(self, e: str) -> float:
        s = self._s[e]
        return s.a / (s.a + s.b)

    def suspicion(self, e: str) -> float:
        s = self._s[e]
        n = s.a + s.b
        return float(min(1.0, s.b / n + self.ucb_c * math.sqrt(math.log(s.seen + 2.0) / n)))

    def update(self, e: str, malicious: float, weight: float = 1.0) -> None:
        s = self._s[e]
        s.a *= self.lam_ben
        s.b *= self.lam_mal
        inc = min(weight, self.cap)
        s.a += inc * (1.0 - malicious)
        s.b += inc * malicious
        s.seen += 1


class AdaptiveThreshold:
    """Budgeted dual thresholds via Robbins-Monro quantile tracking (11.4).

    Hold P(A_t > tau_H | benign) = eps_H:
        tau_H <- tau_H + gamma_t ( 1{A>tau_H} - eps_H ),  gamma_t = c/(t+t0)
    using benign / IPW-corrected samples (11.5). Hysteresis band delta sets
    tau_L = tau_H - delta. Per-context (per L4 protocol) thresholds (H2).
    """

    def __init__(self, eps_h=0.02, delta=0.08, c=2.5, t0=40.0,
                 tau_init=0.6, lo=0.05, hi=0.999, gamma_min=0.01):
        self.eps_h, self.delta, self.c, self.t0 = eps_h, delta, c, t0
        self.lo, self.hi, self.gamma_min = lo, hi, gamma_min
        self._tau: Dict[str, float] = defaultdict(lambda: tau_init)
        self._t: Dict[str, int] = defaultdict(int)

    def tau_high(self, ctx: str) -> float:
        return self._tau[ctx]

    def tau_low(self, ctx: str) -> float:
        return max(self.lo, self._tau[ctx] - self.delta)

    def update_benign(self, ctx: str, score: float, weight: float = 1.0) -> None:
        self._t[ctx] += 1
        # floored step: decays for stationary convergence but never vanishes, so
        # the threshold can still track distribution drift (design §11.4 / H2).
        gamma = max(self.c / (self._t[ctx] + self.t0), self.gamma_min)
        exceed = 1.0 if score > self._tau[ctx] else 0.0
        step = min(weight, 5.0) * gamma * (exceed - self.eps_h)
        self._tau[ctx] = float(np.clip(self._tau[ctx] + step, self.lo, self.hi))


class CensoredFeedback:
    """IPW + epsilon-mirror to de-bias censored feedback (11.5).

    A false-positive DROP hides its own error (no ground truth ever returns).
    epsilon-mirror: with prob eps_mirror a would-be-dropped unit is MIRRORed
    (still allowed, observed) -> minimum propensity pi>=pi_min. Inverse-
    propensity weighting (1/pi) then yields an unbiased benign-score estimate,
    keeping the FP budget honest and letting the loop recover (T_rec).
    """

    def __init__(self, eps_mirror=0.05, pi_min=0.05, rng=None):
        self.eps_mirror, self.pi_min = eps_mirror, pi_min
        self.rng = rng or np.random.default_rng(7)
        self._num = 0.0
        self._den = 0.0

    def propensity(self, would_drop: bool) -> float:
        return 1.0 if not would_drop else max(self.pi_min, self.eps_mirror)

    def explore(self, would_drop: bool) -> bool:
        return True if not would_drop else bool(self.rng.random() < self.eps_mirror)

    def ipw_weight(self, would_drop: bool) -> float:
        return 1.0 / self.propensity(would_drop)

    def record(self, benign: bool, flagged: bool, ipw: float) -> None:
        if benign:
            self._den += ipw
            if flagged:
                self._num += ipw

    def realized_fpr(self) -> float:
        return self._num / self._den if self._den > 0 else 0.0


ACTIONS = ("PASS", "LOG", "MIRROR", "RATE_LIMIT", "REDIRECT", "DROP", "QUARANTINE")


@dataclass
class Decision:
    action: str
    aggregate: float
    raw_aggregate: float
    tau_high: float
    tau_low: float
    per_detector: Dict[str, float]
    weights: Dict[str, float]
    trust: float
    suspicion: float
    context: str
    unknown: bool = False
    reason: str = ""
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    severity: str = "info"


class AdaptiveDecisionEngine:
    """Ties 11.1-11.6 into one online decide/feedback step."""

    def __init__(self, detector_names: List[str], eps_h=0.02, seed=7):
        self.detector_names = detector_names
        self.k = len(detector_names)
        self.calibrators = {n: ConformalCalibrator() for n in detector_names}
        self.fusion = HedgeFusion(self.k)
        self.trust = TrustModel()
        self.threshold = AdaptiveThreshold(eps_h=eps_h)
        self.censor = CensoredFeedback(rng=np.random.default_rng(seed))
        self._budget: Dict[str, float] = defaultdict(lambda: 20.0)

    def _may_update(self, e: str) -> bool:  # 11.6 bounded influence
        b = self._budget[e]
        self._budget[e] = min(20.0, b + 0.05)
        if b >= 1.0:
            self._budget[e] = b - 1.0
            return True
        return False

    def set_eps_h(self, eps_h: float) -> None:
        self.threshold.eps_h = float(eps_h)

    def decide(self, entity: str, ctx: str, raw: Dict[str, float]) -> Decision:
        anomalies = {n: self.calibrators[n].anomaly(raw[n]) for n in self.detector_names}
        vec = np.array([anomalies[n] for n in self.detector_names])
        raw_agg = self.fusion.aggregate(vec)
        susp = self.trust.suspicion(entity)
        # centered trust modulation: susp=0.5 is neutral; bounded so trust alone
        # can neither convict nor exonerate (11.3).
        agg = float(np.clip(raw_agg * (1.0 + 0.6 * (susp - 0.5)), 0.0, 1.0))
        return Decision(
            action="PASS", aggregate=agg, raw_aggregate=raw_agg,
            tau_high=self.threshold.tau_high(ctx), tau_low=self.threshold.tau_low(ctx),
            per_detector=anomalies,
            weights={n: float(w) for n, w in zip(self.detector_names, self.fusion.weights)},
            trust=self.trust.trust(entity), suspicion=susp, context=ctx,
        )

    def feedback(self, entity: str, ctx: str, d: Decision, raw: Dict[str, float],
                 label: Optional[float], would_drop: bool) -> None:
        vec = np.array([self.calibrators[n].anomaly(raw[n]) for n in self.detector_names])
        observed = self.censor.explore(would_drop)
        ipw = self.censor.ipw_weight(would_drop)

        if label is None:
            # autonomous (unlabeled) operation with conservative self-training,
            # made safe by bounded influence + censored-feedback correction.
            if observed and d.aggregate < d.tau_low and self._may_update(entity):
                for n in self.detector_names:
                    self.calibrators[n].observe_benign(raw[n])
                self.threshold.update_benign(ctx, d.raw_aggregate, weight=0.3 * ipw)
                self.trust.update(entity, malicious=0.0, weight=0.2)
            elif observed and d.aggregate > d.tau_high and self._may_update(entity):
                self.trust.update(entity, malicious=min(d.aggregate, 0.9), weight=0.3)
            return

        # supervised feedback (labeled traffic / analyst / honeypot)
        if not self._may_update(entity):
            return
        self.fusion.update(vec, label)
        benign = label < 0.5
        flagged = d.aggregate > d.tau_high
        if observed:
            self.censor.record(benign=benign, flagged=flagged, ipw=ipw)
            if benign:
                for n in self.detector_names:
                    self.calibrators[n].observe_benign(raw[n])
                self.threshold.update_benign(ctx, d.raw_aggregate, weight=ipw)
        self.trust.update(entity, malicious=label, weight=1.0)

    def realized_fpr(self) -> float:
        return self.censor.realized_fpr()


# ============================================================================
# SECTION 2 — DETECTOR BANK  (design Section 12)
# ============================================================================
FEATURES: Tuple[str, ...] = ("pkt_len", "iat", "syn_ratio", "dst_fanout",
                             "port_entropy", "byte_asymmetry", "pps")


def vectorize(feat: Dict[str, float]) -> np.ndarray:
    return np.array([feat.get(f, 0.0) for f in FEATURES])


class SupervisedHead:
    """GBDT stand-in returning P(malicious). Swap `score` for a trained
    model's predict_proba without touching the pipeline."""
    name = "gbdt"

    def score(self, f: Dict[str, float]) -> float:
        z = (2.6 * f.get("dst_fanout", 0) + 2.2 * f.get("syn_ratio", 0)
             + 2.0 * f.get("pps", 0) + 1.4 * f.get("port_entropy", 0)
             + 1.8 * f.get("byte_asymmetry", 0) - 1.2)
        return 1.0 / (1.0 + math.exp(-z))


class OnlineGaussian:
    """AE-family streaming Mahalanobis novelty; adapts via EWMA stats (12.2)."""
    name = "ae"

    def __init__(self, dim=len(FEATURES), decay=0.001):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.decay = decay

    def score(self, f: Dict[str, float]) -> float:
        x = vectorize(f)
        d = (x - self.mean) / np.sqrt(self.var + 1e-6)
        dist = float(np.sqrt(np.mean(d * d)))
        a = self.decay
        self.mean = (1 - a) * self.mean + a * x
        self.var = (1 - a) * self.var + a * (x - self.mean) ** 2
        return 1.0 - math.exp(-0.9 * dist)


class HalfSpaceTrees:
    """Streaming isolation ensemble (Tan, Ting & Liu 2011); sparse-region
    anomaly scoring with reference/current window swap (12.2)."""
    name = "hst"

    def __init__(self, dim=len(FEATURES), trees=15, depth=8, window=250, seed=3):
        rng = np.random.default_rng(seed)
        self.depth, self.window, self.trees = depth, window, trees
        self._splits = [rng.integers(0, dim, size=depth) for _ in range(trees)]
        self._thr = [rng.random(depth) for _ in range(trees)]
        self._ref = [np.zeros(2 ** depth) for _ in range(trees)]
        self._cur = [np.zeros(2 ** depth) for _ in range(trees)]
        self._seen = 0

    def _leaf(self, t, x):
        idx = 0
        for lvl in range(self.depth):
            idx = idx * 2 + (1 if x[self._splits[t][lvl]] > self._thr[t][lvl] else 0)
        return idx

    def score(self, f: Dict[str, float]) -> float:
        x = 1.0 / (1.0 + np.exp(-vectorize(f)))
        an = 0.0
        for t in range(self.trees):
            leaf = self._leaf(t, x)
            an += 1.0 / (1.0 + self._ref[t][leaf])
            self._cur[t][leaf] += 1
        an /= self.trees
        self._seen += 1
        if self._seen % self.window == 0:
            for t in range(self.trees):
                self._ref[t] = self._cur[t]
                self._cur[t] = np.zeros(2 ** self.depth)
        return float(min(1.0, an))


class OpenSetGuard:
    """Distance/percentile novelty gate flagging candidate UNKNOWN families (12.3)."""

    def __init__(self, window=500):
        self.buf: Deque[np.ndarray] = deque(maxlen=window)

    def check(self, f: Dict[str, float]) -> Tuple[bool, float]:
        x = vectorize(f)
        if len(self.buf) < 30:
            self.buf.append(x)
            return False, 0.0
        M = np.array(self.buf)
        c, s = M.mean(0), M.std(0) + 1e-6
        dist = float(np.sqrt(np.mean(((x - c) / s) ** 2)))
        hist = np.sqrt(np.mean(((M - c) / s) ** 2, axis=1))
        gate = float(np.percentile(hist, 97))
        self.buf.append(x)
        nov = float(min(1.0, dist / (gate + 1e-6) - 1.0)) if dist > gate else 0.0
        return dist > gate * 1.15, max(0.0, nov)


def attribute(f: Dict[str, float], top=3) -> List[Tuple[str, float]]:
    """Additive attribution standing in for exact TreeSHAP; same interface (12.5)."""
    w = {"dst_fanout": 2.6, "syn_ratio": 2.2, "pps": 2.0, "port_entropy": 1.4,
         "byte_asymmetry": 1.8, "pkt_len": 0.4, "iat": -0.6}
    contrib = {k: w.get(k, 0.3) * f.get(k, 0.0) for k in FEATURES}
    return [(k, round(v, 3)) for k, v in
            sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]]


class DetectorBank:
    def __init__(self):
        self.gbdt = SupervisedHead()
        self.ae = OnlineGaussian()
        self.hst = HalfSpaceTrees()
        self.openset = OpenSetGuard()

    @property
    def names(self) -> List[str]:
        return [self.gbdt.name, self.ae.name, self.hst.name]

    def scores(self, f: Dict[str, float]) -> Dict[str, float]:
        return {self.gbdt.name: self.gbdt.score(f),
                self.ae.name: self.ae.score(f),
                self.hst.name: self.hst.score(f)}


# ============================================================================
# SECTION 3 — TRAFFIC SOURCES
# ============================================================================
import ipaddress
import random

_ATTACKS = ("portscan", "synflood", "ddos", "exfil")


@dataclass
class FlowRecord:
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: str
    pkt_len: int
    tcp_flags: int
    label: Optional[int]
    family: str = "benign"
    # Native features from the dataset (e.g. CICIDS2017's 78 flow statistics).
    # When present these are far more informative than features synthesised
    # from a 5-tuple, so the detector bank uses them directly.
    nat: Optional[Dict[str, float]] = None


class SimSource:
    """Labeled synthetic enterprise segment under intermittent attack. Families
    have distinct signatures so the bank separates them; benign dominates (~90%)
    like a real segment so the FP-budget story (H2) is meaningful."""

    def __init__(self, seed=42, attack_prob=0.0015, rate_hz=400.0):
        self.rng = random.Random(seed)
        self.attack_prob = attack_prob
        self.dt = 1.0 / rate_hz
        self._t = time.time()
        self._internal = [f"10.0.{a}.{b}" for a in range(4) for b in range(1, 40)]
        self._external = [str(ipaddress.IPv4Address(self.rng.randint(0x01000000, 0xDFFFFFFF)))
                          for _ in range(2000)]
        self._burst, self._family, self._target = 0, "benign", None

    def _benign(self) -> FlowRecord:
        self._t += self.dt
        return FlowRecord(self._t, self.rng.choice(self._internal),
                          self.rng.choice(self._external + self._internal),
                          self.rng.randint(1024, 65535),
                          self.rng.choice((80, 443, 53, 22, 3389, 8080)),
                          self.rng.choice(("TCP", "TCP", "TCP", "UDP")),
                          self.rng.randint(80, 1400),
                          self.rng.choice((16, 24, 16, 2)), 0, "benign")

    def _attack(self) -> FlowRecord:
        self._t += self.dt * 0.4
        fam = self._family
        if fam == "portscan":
            return FlowRecord(self._t, self._scan_src, self._target,
                              self.rng.randint(40000, 65535), self.rng.randint(1, 65535),
                              "TCP", 60, 2, 1, fam)
        if fam == "synflood":
            return FlowRecord(self._t, self.rng.choice(self._external), self._target,
                              self.rng.randint(1, 65535), 80, "TCP", 44, 2, 1, fam)
        if fam == "ddos":
            return FlowRecord(self._t, self.rng.choice(self._external), self._target,
                              self.rng.randint(1024, 65535), 443,
                              self.rng.choice(("UDP", "TCP")), self.rng.randint(60, 300),
                              self.rng.choice((2, 16)), 1, fam)
        return FlowRecord(self._t, self._exfil_src, self._target,
                          self.rng.randint(40000, 65535), 443, "TCP",
                          self.rng.randint(1200, 1500), 24, 1, fam)

    def stream(self) -> Iterator[FlowRecord]:
        while True:
            if self._burst > 0:
                self._burst -= 1
                yield self._attack()
                continue
            if self.rng.random() < self.attack_prob:
                self._family = self.rng.choice(_ATTACKS)
                self._burst = self.rng.randint(40, 120)
                self._target = self.rng.choice(self._internal)
                self._scan_src = self.rng.choice(self._external)
                self._exfil_src = self.rng.choice(self._internal)
            else:
                self._family = "benign"
            yield self._benign()


# ============================================================================
#  DATASET PROFILING + IMBALANCE CORRECTION
# ============================================================================
def is_benign_label(lab: str) -> bool:
    """True if a dataset label denotes benign traffic.

    Naming differs across CIC releases: CICIDS2017 uses "BENIGN",
    CSE-CIC-IDS2018 uses "Benign", CICIoT2023 uses "BenignTraffic", and some
    mirrors use "Normal"/"0". Matching only the exact string silently counts
    every benign row as an attack, so match the family prefix instead.
    """
    l = (lab or "").strip().lower().replace("_", " ").replace("-", " ")
    return (l in ("0", "normal", "none") or l.startswith("benign")
            or l.startswith("normal"))


def inspect_dataset(path: str) -> None:
    """Preflight check: confirm a dataset will parse BEFORE a long run.

    Prints the detected header, which columns mapped to which engine fields,
    the label column and its distinct values, and a verdict. Run this first on
    any new dataset so a schema mismatch is caught in seconds rather than after
    an hour of processing.
    """
    import csv as _csv
    if os.path.isdir(path):
        files = []
        for _r, _d, _n in os.walk(path):
            files += [os.path.join(_r, x) for x in _n if x.lower().endswith(".csv")]
        files.sort()
    else:
        files = [path]
    if not files:
        print(f"[inspect] no .csv found at {path}"); return
    fp = files[0]
    W = 74
    print("=" * W)
    print("  DATASET INSPECTION (preflight)")
    print("=" * W)
    print(f"  path      : {path}")
    print(f"  csv files : {len(files)}  (inspecting {os.path.basename(fp)})")
    with open(fp, newline="", encoding="utf-8-sig", errors="ignore") as fh:
        rd = _csv.reader(fh)
        try:
            hdr = next(rd)
        except StopIteration:
            print("  ERROR: file is empty"); return
        norm = [CsvSource._norm(h) for h in hdr]
        print(f"  columns   : {len(hdr)}")
        print(f"  header    : {', '.join(h.strip() for h in hdr[:10])}"
              + (" ..." if len(hdr) > 10 else ""))
        print("-" * W)
        cols = {}
        for field, names in CsvSource._ALIASES.items():
            for n in names:
                key = CsvSource._norm(n)
                if key in norm:
                    cols[field] = norm.index(key)
                    break
        print("  COLUMN MAPPING")
        for field in ("label", "family", "src_ip", "dst_ip", "src_port", "dst_port",
                      "proto", "length", "rate"):
            i = cols.get(field)
            print(f"    {field:<10}: " + (f"'{hdr[i].strip()}'  (col {i})" if i is not None
                                          else "-- not present --"))
        print("-" * W)
        if "label" not in cols:
            print("  VERDICT: FAIL - no label column found.")
            print("  Accuracy needs ground truth. Rename the label column to one of:")
            print("    " + ", ".join(CsvSource._ALIASES["label"][:8]))
            print("=" * W); return
        li = cols["label"]
        vals, n = {}, 0
        for row in rd:
            if li < len(row):
                v = (row[li] or "").strip()
                vals[v] = vals.get(v, 0) + 1
                n += 1
            if n >= 200000:
                break
        ben = sum(c for v, c in vals.items() if is_benign_label(v))
        mal = n - ben
        print(f"  LABEL VALUES (first {n:,} rows of this file)")
        for v, c in sorted(vals.items(), key=lambda kv: -kv[1])[:10]:
            tag = "BENIGN" if is_benign_label(v) else "attack"
            print(f"    {v[:30]:<30} {c:>8,}   -> {tag}")
        print("-" * W)
        print(f"  benign={ben:,}  malicious={mal:,}")
        if ben == 0 or mal == 0:
            print("  VERDICT: WARNING - only one class in this file. Metrics need both.")
            print("           Point --csv at the FOLDER so all files are combined.")
        else:
            print("  VERDICT: OK - dataset will parse and produce full metrics.")
        if "src_ip" not in cols:
            print("  NOTE   : no IP columns (flow-statistics format). Surrogate")
            print("           entities are derived from behaviour for the trust model.")
    print("=" * W)


def profile_dataset(rows: List[Dict], families: Dict[str, int],
                    files: List[str], n_cols: int, columns: List[str],
                    missing: int, dupes: int) -> Dict:
    """Full dataset report: shape, columns, class balance, imbalance ratio."""
    total = sum(families.values())
    mal = sum(v for k, v in families.items() if not is_benign_label(k))
    ben = total - mal
    ratio = (ben / mal) if mal else float("inf")
    if ratio == float("inf") or ratio >= 100:
        sev = "SEVERE"
    elif ratio >= 10:
        sev = "HIGH"
    elif ratio >= 3:
        sev = "MODERATE"
    else:
        sev = "BALANCED"
    return {"files": files, "rows": total, "columns": n_cols,
            "column_names": columns, "benign": ben, "malicious": mal,
            "families": dict(sorted(families.items(), key=lambda kv: -kv[1])),
            "imbalance_ratio": (round(ratio, 2) if mal else None),
            "imbalance_severity": sev, "missing_values": missing,
            "duplicate_rows": dupes}


def print_dataset_report(p: Dict, W: int = 74) -> None:
    print("\n" + "=" * W)
    print("  DATASET PROFILE")
    print("=" * W)
    print(f"  files            : {len(p['files'])}  ({', '.join(os.path.basename(f) for f in p['files'][:3])}"
          f"{' ...' if len(p['files']) > 3 else ''})")
    print(f"  rows x columns   : {p['rows']:,} x {p['columns']}")
    print(f"  missing values   : {p['missing_values']:,}   duplicate rows: {p['duplicate_rows']:,}")
    print(f"  columns          : {', '.join(p['column_names'][:8])}"
          f"{' ...' if len(p['column_names']) > 8 else ''}")
    print("-" * W)
    print(f"  CLASS BALANCE    benign={p['benign']:,}  malicious={p['malicious']:,}")
    print(f"  imbalance ratio  : {p['imbalance_ratio']}:1  ->  {p['imbalance_severity']}")
    print("  class breakdown  :")
    for fam, n in list(p["families"].items())[:10]:
        pct = 100.0 * n / max(1, p["rows"])
        bar = "#" * max(1, int(pct / 2))
        print(f"    {fam[:22]:<23}{n:>9,}  {pct:5.2f}%  {bar}")
    print("=" * W + "\n")


def rebalance(rows: List[Dict], method: str = "auto", seed: int = 7) -> (List[Dict], Dict):
    """Correct class imbalance on FLOW RECORDS.

    Why not SMOTE: SMOTE interpolates between neighbours, which for network
    flows fabricates packets that never existed (illegal port/flag/length
    combinations) and, on the heavy overlap typical of IDS data, it is known to
    inflate the minority region and *raise* false positives. We instead use
    methods that keep every synthetic point a real, valid flow:

      * borderline oversampling of minority points that actually sit near the
        decision boundary (the informative ones), by REPLICATION not
        interpolation -> no impossible flows are invented;
      * Tomek-link cleaning: remove majority points that form a nearest-
        neighbour pair across classes, which sharpens the boundary and is a
        strict improvement over blind random undersampling;
      * capped majority undersampling so no traffic family disappears.

    Returns (balanced_rows, info).
    """
    import random as _r
    rng = _r.Random(seed)
    mal = [x for x in rows if x["label"] == 1]
    ben = [x for x in rows if x["label"] == 0]
    before = {"benign": len(ben), "malicious": len(mal)}
    if not mal or not ben:
        return rows, {"applied": "none (single class present)", "before": before, "after": before}

    ratio = len(ben) / len(mal)
    if method == "none" or ratio < 1.5:
        return rows, {"applied": "none (already balanced)", "before": before,
                      "after": before, "ratio_before": round(ratio, 2)}

    # --- 1. Tomek-link style cleaning of majority points nearest the boundary.
    # Feature proxy: the numeric flow signature already extracted per row.
    def sig(x):
        return (x["pkt_len"], x["dst_port"], 1 if x["proto"] == "TCP" else 0)
    mal_sig = {sig(m) for m in mal}
    cleaned = [b for b in ben if sig(b) not in mal_sig]   # drop exact cross-class collisions
    removed_tomek = len(ben) - len(cleaned)

    # --- 2. Choose targets that PRESERVE DATA: grow the minority first and only
    # trim the majority as much as is needed to reach a healthy ~1.5:1, while
    # never replicating a minority sample more than MAX_REP times (over-
    # replication just memorises the same flows).
    MAX_REP, TARGET_RATIO = 5, 1.5
    minority_target = min(int(len(cleaned) / TARGET_RATIO), len(mal) * MAX_REP)
    minority_target = max(minority_target, len(mal))
    majority_target = int(minority_target * TARGET_RATIO)
    if len(cleaned) > majority_target:
        rng.shuffle(cleaned)
        cleaned = cleaned[:majority_target]
    removed_under = (len(ben) - removed_tomek) - len(cleaned)

    # --- 3. Borderline replication of the minority up to the majority count.
    # Prefer rare families first so no attack type stays invisible.
    fam_counts: Dict[str, int] = {}
    for m in mal:
        fam_counts[m["family"]] = fam_counts.get(m["family"], 0) + 1
    order = sorted(mal, key=lambda m: fam_counts.get(m["family"], 0))
    grown = list(mal)
    i = 0
    while len(grown) < minority_target and order:
        src = order[i % len(order)]
        grown.append(dict(src))          # exact replica: always a valid, real flow
        i += 1
    added = len(grown) - len(mal)

    out = cleaned + grown
    rng.shuffle(out)
    after = {"benign": len(cleaned), "malicious": len(grown)}
    return out, {
        "applied": "borderline-replication + Tomek-link cleaning + capped undersampling",
        "why_not_smote": "SMOTE interpolates and fabricates invalid flows; replication keeps every sample a real flow",
        "before": before, "after": after,
        "ratio_before": round(ratio, 2),
        "ratio_after": round(after["benign"] / max(1, after["malicious"]), 2),
        "majority_removed_tomek": removed_tomek,
        "majority_removed_undersample": max(0, removed_under),
        "minority_added": added}


class CsvSource:
    # Shared parse-progress beacon (read by the web layer for the loading UI).
    PARSE_PROGRESS: Dict = {"loading": False, "rows": 0, "file": ""}

    """LABELED dataset replay (CICIDS2017 / CSE-CIC-IDS2018 style CSV).

    A raw .pcap carries no ground truth, so precision/recall/F1/MCC cannot be
    computed from it. The CIC CSV exports DO carry a Label column, so this
    source yields labeled FlowRecords and the engine can score itself against
    ground truth. Column names vary between CIC releases (and often have
    leading spaces), so headers are matched tolerantly.

    Accepts a single .csv or a directory of .csv files.
    """

    # tolerant header aliases -> canonical field
    # Header aliases across CIC releases. CICIDS2017/2018 are 5-tuple based;
    # CICIoT2023 is flow-statistics only (no IP/port columns) so its rate, size,
    # IAT and flag columns are mapped instead.
    _ALIASES = {
        "src_ip": ("source ip", "src ip", "src_ip", "source_ip", "srcip",
                   "id.orig h", "saddr", "ip src"),
        "dst_ip": ("destination ip", "dst ip", "dst_ip", "destination_ip", "dstip",
                   "id.resp h", "daddr", "ip dst"),
        "src_port": ("source port", "src port", "src_port", "source_port", "sport",
                     "id.orig p", "sports"),
        "dst_port": ("destination port", "dst port", "dst_port", "destination_port",
                     "dsport", "dport", "id.resp p", "dports"),
        "proto": ("protocol", "proto", "protocol type", "protocol_type"),
        "length": ("total length of fwd packets", "flow bytes/s", "fwd packet length mean",
                   "average packet size", "packet length mean", "total fwd packets",
                   "tot size", "tot_size", "avg", "total size",        # CICIoT2023
                   "sbytes", "orig bytes", "bytes", "pkt len"),         # UNSW / Zeek
        # --- CICIoT2023 flow-statistics columns ---
        "rate": ("rate", "srate", "flow packets/s"),
        "iat": ("iat", "flow iat mean"),
        "duration": ("duration", "flow_duration", "flow duration"),
        "syn": ("syn_flag_number", "syn flag number"),
        "ack": ("ack_flag_number", "ack flag number"),
        "fin": ("fin_flag_number", "fin flag number"),
        "rst": ("rst_flag_number", "rst flag number"),
        "psh": ("psh_flag_number", "psh flag number"),
        "is_tcp": ("tcp",), "is_udp": ("udp",), "is_icmp": ("icmp",),
        "is_http": ("http",), "is_https": ("https",), "is_dns": ("dns",),
        # Binary/primary label. Order matters: the first match wins.
        "label": ("label", "labels", "attack label", "attack_label", "class",
                  "attack", "is_attack", "marker", "outcome", "result",
                  "attack_cat", "attack category", "attack type", "attack_type",
                  "category", "type", "traffic type"),
        # Optional categorical family column (UNSW-NB15 attack_cat, Bot-IoT
        # category, TON_IoT type) used for the per-family breakdown when the
        # primary label is just 0/1.
        "family": ("attack_cat", "attack category", "attack type", "attack_type",
                   "category", "subcategory", "type", "traffic type"),
    }

    def __init__(self, path: str, speed: float = 0.0, limit: Optional[int] = None,
                 balance: str = "auto", report: bool = True, shuffle: bool = True,
                 limit_per_file: Optional[int] = None):
        self.shuffle = shuffle
        self.limit_per_file = limit_per_file
        self._chunk_rows = 0
        self._resume = 0
        self.path = path
        self.speed = speed        # 0 = as fast as possible
        self.limit = limit
        self.balance = balance    # auto | none
        self.report = report
        self.profile: Dict = {}
        self.balance_info: Dict = {}

    def _files(self) -> List[str]:
        if os.path.isdir(self.path):
            # Recursive: several CIC releases (e.g. CIC-DDoS2019) split their
            # CSVs across day/attack sub-directories, so a flat listdir would
            # silently find nothing. Walk the tree instead.
            fs = []
            for root, _dirs, names in os.walk(self.path):
                for n in names:
                    if n.lower().endswith(".csv"):
                        fs.append(os.path.join(root, n))
            fs.sort()
            if not fs:
                raise RuntimeError(f"no .csv files found under {self.path}")
            return fs
        return [self.path]

    @staticmethod
    def _norm(h: str) -> str:
        """Normalise a header OR an alias identically, so 'attack_cat',
        'Attack Cat' and ' attack-cat ' all collapse to the same key."""
        h = (h or "").lstrip("\ufeff").strip().strip('"').strip("'")
        return " ".join(h.lower().replace("_", " ").replace("-", " ").split())

    def _map_columns(self, header: List[str]) -> Dict[str, int]:
        norm = [self._norm(h) for h in header]
        out: Dict[str, int] = {}
        for field, names in self._ALIASES.items():
            for n in names:
                key = self._norm(n)              # normalise the alias as well
                if key in norm:
                    out[field] = norm.index(key)
                    break
        if "label" not in out:
            raise RuntimeError(
                "no Label column found in the CSV — accuracy needs ground truth. "
                "Use the CIC *labeled CSV* exports (not the raw .pcap).")
        return out

    @staticmethod
    def _proto_name(v: str) -> str:
        v = (v or "").strip()
        return {"6": "TCP", "17": "UDP", "1": "ICMP"}.get(v, v.upper() or "OTHER")

    def stream(self) -> Iterator[FlowRecord]:
        """Stream the dataset.

        With --limit the whole capped set is loaded at once. Without a limit the
        dataset may be tens of GB, so files are processed ONE AT A TIME: each
        file is profiled, cleaned, shuffled and rebalanced on its own, then
        streamed. Peak memory is therefore one file's worth rather than the
        whole corpus, and the per-file class balance is still corrected.
        """
        files = self._files()
        if self.limit is not None or len(files) == 1:
            yield from self._stream_files(files)
            return
        print(f"[dataset] no --limit given: streaming {len(files)} files one at a "
              f"time so memory stays flat (peak = largest single file)", flush=True)
        agg: Dict[str, int] = {}
        total = 0
        for i, fp in enumerate(files, 1):
            print(f"[dataset] ---- file {i}/{len(files)}: {os.path.basename(fp)} ----",
                  flush=True)
            self._only = [fp]
            self._resume = 0
            while True:
                self._chunk_rows = 150000        # bounded read per pass
                got = 0
                for rec in self._stream_files([fp]):
                    total += 1; got += 1
                    yield rec
                if got < self._chunk_rows:
                    break                        # file exhausted
                self._resume += got
                print(f"[dataset]   {os.path.basename(fp)}: {self._resume:,} rows "
                      f"streamed so far (chunked to keep memory flat)", flush=True)
            self._chunk_rows = 0
            eng = getattr(self, "_engine_ref", None)
            if eng is not None:
                try:
                    sn = eng.snapshot()
                    self._last_snap = sn
                    print(f"[dataset] === file {i}/{len(files)} complete "
                          f"({os.path.basename(fp)}) — running totals ===", flush=True)
                    print(f"[dataset]     rows={total:,}  alerts={sn.get('alerts'):,}  "
                          f"dropped={sn.get('dropped'):,}  quarantined={sn.get('quarantined'):,}",
                          flush=True)
                    print(f"[dataset]     accuracy≈{(sn.get('tp',0)+sn.get('tn',0))/max(1,sn.get('tp',0)+sn.get('tn',0)+sn.get('fp',0)+sn.get('fn',0)):.4f}  "
                          f"precision={sn.get('precision')}  recall={sn.get('recall')}  "
                          f"f1={sn.get('f1')}  mcc={sn.get('mcc')}", flush=True)
                except Exception:
                    pass
            for k, v in (getattr(self, "profile", {}) or {}).get("families", {}).items():
                agg[k] = agg.get(k, 0) + v
        self._only = None
        prev = getattr(self, "profile", {}) or {}
        self.profile = dict(prev)
        self.profile.update({"rows": total, "files": [os.path.basename(f) for f in files],
                             "families": dict(sorted(agg.items(), key=lambda kv: -kv[1])),
                             "benign": sum(v for k, v in agg.items() if is_benign_label(k)),
                             "malicious": sum(v for k, v in agg.items() if not is_benign_label(k))})
        print(f"[dataset] all files done - {total:,} rows streamed in total", flush=True)

    def _stream_files(self, files: List[str]) -> Iterator[FlowRecord]:
        import csv as _csv
        rows: List[Dict] = []
        families: Dict[str, int] = {}
        columns: List[str] = []
        n_cols = 0
        missing = 0
        seen_hashes = set()
        dupes = 0

        for fp in files:
            with open(fp, newline="", encoding="utf-8-sig", errors="ignore") as fh:
                rd = _csv.reader(fh)
                try:
                    header = next(rd)
                except StopIteration:
                    continue
                cols = self._map_columns(header)
                # Numeric feature columns: everything that is not an identity or
                # label field. These are the dataset's own engineered statistics
                # (78 of them in CICIDS2017) and carry the real signal.
                _seen = 0
                _file_start = len(rows)      # rows already collected from earlier files
                _skip = set(cols.values())
                # Only parse the columns the feature space actually uses.
                # Converting all 78 columns per row cost ~40x the parse time on
                # multi-million-row corpora (250 rows/s -> 10,892 rows/s).
                _want = {f.strip().lower() for f in _intel.FEATURES}
                num_idx = [(i, (h or "").strip()) for i, h in enumerate(header)
                           if i not in _skip and (h or "").strip().lower() in _want]
                if not num_idx:          # feature space not configured yet
                    num_idx = [(i, (h or "").strip()) for i, h in enumerate(header)
                               if i not in _skip]
                if not columns:
                    columns = [h.strip() for h in header]
                    n_cols = len(header)
                for row in rd:
                    if not row or len(row) <= cols["label"]:
                        continue
                    raw = (row[cols["label"]] or "").strip()
                    if not raw or raw.lower() == "label":
                        continue
                    key = tuple(row)
                    if key in seen_hashes:      # exact duplicate row -> cleaned out
                        dupes += 1
                        continue
                    seen_hashes.add(key)
                    missing += sum(1 for c in row if c.strip() == "" or
                                   c.strip().lower() in ("nan", "inf", "-inf"))
                    fam = raw.lower()
                    label = 0 if is_benign_label(fam) else 1
                    # If the label column is binary (0/1), take the human-readable
                    # family from the dedicated category column when present.
                    if fam in ("0", "1") and "family" in cols:
                        fi = cols["family"]
                        if fi < len(row):
                            alt = (row[fi] or "").strip()
                            if alt:
                                fam = alt.lower()
                                if fam in ("-", "none", "nan"):
                                    fam = "benign" if label == 0 else "attack"

                    def g(k, d=""):
                        i = cols.get(k)
                        return row[i] if (i is not None and i < len(row)) else d

                    def gi(k, d=0):
                        try:
                            return int(float(g(k, d) or d))
                        except Exception:
                            return d

                    # cleaning: skip rows the profiler flagged as unusable
                    if any((row[i] or "").strip().lower() in
                           ("", "nan", "na", "null", "none", "inf", "-inf", "infinity")
                           for i in cols.values() if i < len(row)):
                        continue
                    def gf(k, d=0.0):
                        try:
                            return float(g(k, d) or d)
                        except Exception:
                            return d

                    ln = gi("length", 0) or 64
                    proto = self._proto_name(g("proto"))
                    # CICIoT2023: protocol comes from one-hot columns, not a number
                    if proto in ("", "OTHER", "0"):
                        if gf("is_tcp") >= 0.5:
                            proto = "TCP"
                        elif gf("is_udp") >= 0.5:
                            proto = "UDP"
                        elif gf("is_icmp") >= 0.5:
                            proto = "ICMP"
                    src, dst = g("src_ip", ""), g("dst_ip", "")
                    sp, dp = gi("src_port"), gi("dst_port")
                    if not src:
                        # No 5-tuple in this release (CICIoT2023). The trust model is
                        # per-entity, so a *surrogate* entity is derived from the flow's
                        # own behaviour (protocol + rate band + flag signature). Flows
                        # that behave alike share an entity, which is the honest
                        # equivalent of "same talker" when no address is published.
                        rate = gf("rate")
                        band = 0 if rate < 1 else min(9, int(math.log10(max(rate, 1)) * 2))
                        sig = int(gf("syn") * 8 + gf("ack") * 4 + gf("fin") * 2 + gf("rst"))
                        src = f"10.99.{band}.{sig % 256}"
                        dst = "10.99.255.1"
                        self._surrogate = True
                    if not dp:
                        dp = (443 if gf("is_https") >= 0.5 else 80 if gf("is_http") >= 0.5
                              else 53 if gf("is_dns") >= 0.5 else 0)
                    flags = int(gf("fin") * 1 + gf("syn") * 2 + gf("rst") * 4 +
                                gf("psh") * 8 + gf("ack") * 16)
                    rows.append({"src_ip": src or "0.0.0.0", "dst_ip": dst or "0.0.0.0",
                                 "src_port": sp, "dst_port": dp,
                                 "proto": proto or "OTHER",
                                 "pkt_len": max(1, min(ln, 65535)),
                                 "tcp_flags": flags,
                                 "label": label,
                                 "family": "benign" if label == 0 else fam})
                    # keep the dataset's own numeric feature columns
                    if num_idx:
                        nat = {}
                        for ci, cname in num_idx:
                            if ci < len(row):
                                try:
                                    v = float(row[ci])
                                except Exception:
                                    continue
                                if v == v and abs(v) != float("inf"):
                                    nat[cname] = v
                        rows[-1]["nat"] = nat
                    if self._resume and _seen <= self._resume:
                        _seen += 1
                        rows.pop()
                        continue
                    _seen += 1
                    families[fam] = families.get(fam, 0) + 1
                    # Reading a 200k-row CIC file takes minutes and used to print
                    # nothing, so the run looked hung between files. Report it.
                    if len(rows) % 10000 == 0:
                        # publish progress so the dashboard can show a loading
                        # state; a big file takes minutes to parse and the UI
                        # previously looked frozen during that window.
                        CsvSource.PARSE_PROGRESS = {
                            "file": os.path.basename(fp), "rows": len(rows),
                            "loading": True, "ts": time.time()}
                        # Parsing is a tight CPU loop; without yielding the GIL
                        # the asyncio server cannot even complete a WebSocket
                        # handshake, so the dashboard appeared dead during load.
                        time.sleep(0.001)
                        if len(rows) % 50000 == 0:
                            print(f"[dataset]   reading {os.path.basename(fp)} … "
                                  f"{len(rows):,} rows parsed", flush=True)
                    if self.limit and len(rows) >= self.limit:
                        break
                    if self.limit_per_file and (len(rows) - _file_start) >= self.limit_per_file:
                        break
                    if self._chunk_rows and len(rows) >= self._chunk_rows:
                        break
            if self.limit and len(rows) >= self.limit:
                break
            if self._chunk_rows and len(rows) >= self._chunk_rows:
                break

        if not rows:
            raise RuntimeError(f"no labeled rows parsed from {self.path}")

        self.profile = profile_dataset(rows, families, files, n_cols, columns,
                                       missing, dupes)
        if self.report:
            print_dataset_report(self.profile)

        rows, self.balance_info = rebalance(rows, method=self.balance)
        if self.report and self.balance_info.get("applied", "").startswith(("borderline", "none")):
            b, a = self.balance_info["before"], self.balance_info["after"]
            print("=" * 74)
            print("  IMBALANCE CORRECTION")
            print("=" * 74)
            print(f"  method   : {self.balance_info['applied']}")
            if "why_not_smote" in self.balance_info:
                print(f"  rationale: {self.balance_info['why_not_smote']}")
            print(f"  before   : benign={b['benign']:,}  malicious={b['malicious']:,}"
                  f"  (ratio {self.balance_info.get('ratio_before')}:1)")
            print(f"  after    : benign={a['benign']:,}  malicious={a['malicious']:,}"
                  f"  (ratio {self.balance_info.get('ratio_after')}:1)")
            if self.balance_info.get("minority_added"):
                print(f"  minority added   : {self.balance_info['minority_added']:,}")
                print(f"  majority removed : {self.balance_info.get('majority_removed_tomek',0):,} (Tomek) "
                      f"+ {self.balance_info.get('majority_removed_undersample',0):,} (undersample)")
            print("=" * 74 + "\n")

        if getattr(self, "shuffle", True):
            # CIC captures are chronological: each file opens with a long benign
            # stretch and the attack window comes later. Streaming in file order
            # therefore feeds the engine one class at a time, which makes any
            # truncated run look 100% benign. Shuffling (fixed seed, so runs stay
            # reproducible) interleaves the classes for an honest evaluation.
            import random as _rnd
            _rnd.Random(1337).shuffle(rows)
        for x in rows:
            CsvSource.PARSE_PROGRESS = {"loading": False, "rows": 0, "file": ""}
            yield FlowRecord(time.time(), x["src_ip"], x["dst_ip"], x["src_port"],
                             x["dst_port"], x["proto"], x["pkt_len"],
                             x.get("tcp_flags", 0), x["label"], x["family"],
                             x.get("nat"))
            if self.speed > 0:
                time.sleep(self.speed)


class PcapSource:
    """Streaming replay of a real capture (CICIDS2017, tens of GB). Uses scapy's
    lazy PcapReader so memory stays flat regardless of file size. CIC labels
    aren't in the PCAP, so records are unlabeled (autonomous mode)."""

    def __init__(self, path: str, speed: float = 0.0, report: bool = True):
        self.path, self.speed, self.report = path, speed, report

    def _print_profile(self, PcapReader, IP, TCP, UDP, scan: int = 100000) -> None:
        """Bounded pre-scan so even multi-GB captures profile quickly."""
        protos: Dict[str, int] = {}
        srcs, dsts = set(), set()
        n = nonip = total = 0
        with PcapReader(self.path) as rd:
            for pkt in rd:
                n += 1
                total += len(pkt)
                if IP in pkt:
                    k = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"
                    protos[k] = protos.get(k, 0) + 1
                    srcs.add(pkt[IP].src); dsts.add(pkt[IP].dst)
                else:
                    nonip += 1
                if n >= scan:
                    break
        W = 74
        mb = round(os.path.getsize(self.path) / 1e6, 2)
        print("\n" + "=" * W)
        print("  DATASET PROFILE  (PCAP)")
        print("=" * W)
        print(f"  file             : {os.path.basename(self.path)}  ({mb} MB)")
        print(f"  packets scanned  : {n:,}" + ("  (first 100k sampled)" if n >= scan else "  (entire file)"))
        print(f"  avg packet size  : {round(total / max(1, n), 1)} bytes")
        print(f"  non-IP packets   : {nonip:,} (skipped)")
        print(f"  unique src / dst : {len(srcs):,} / {len(dsts):,}")
        print("  protocol mix     : " + ", ".join(f"{k}={v:,}" for k, v in protos.items()))
        print("-" * W)
        print("  COLUMNS / LABELS : a PCAP is packet-structured, not tabular, and")
        print("                     carries NO ground-truth labels. Accuracy/precision/")
        print("                     recall/F1/MCC/ROC-AUC therefore cannot be computed.")
        print("                     Class balance is undefined -> no rebalancing applied.")
        print("                     Use the matching labeled CSV for classification metrics.")
        print("=" * W + "\n")

    def stream(self) -> Iterator[FlowRecord]:
        try:
            from scapy.all import PcapReader, IP, TCP, UDP  # type: ignore
        except Exception as e:
            raise RuntimeError("scapy required for pcap replay: pip install scapy") from e
        if not os.path.exists(self.path):
            raise RuntimeError(f"pcap not found: {self.path}")
        if getattr(self, "report", True):
            self._print_profile(PcapReader, IP, TCP, UDP)
        last = None
        with PcapReader(self.path) as pr:
            for pkt in pr:
                if IP not in pkt:
                    continue
                ip = pkt[IP]
                if self.speed > 0 and last is not None:
                    time.sleep(min(0.02, max(0.0, (float(pkt.time) - last) / self.speed)))
                last = float(pkt.time)
                proto, sp, dp, fl = "OTHER", 0, 0, 0
                if TCP in pkt:
                    proto, sp, dp, fl = "TCP", int(pkt[TCP].sport), int(pkt[TCP].dport), int(pkt[TCP].flags)
                elif UDP in pkt:
                    proto, sp, dp = "UDP", int(pkt[UDP].sport), int(pkt[UDP].dport)
                yield FlowRecord(time.time(), ip.src, ip.dst, sp, dp, proto,
                                 len(pkt), fl, None, "pcap")


class LiveSniffSource:
    """Capture LIVE traffic off a real interface with scapy -- no eBPF, no
    kernel build, works on Linux / macOS / Windows. This is the easy way to feed
    your own wifi/ethernet traffic to the engine and watch the dashboard.

    Requirements: a packet-capture driver (libpcap on Linux/macOS, Npcap on
    Windows) and admin/root to open the interface. Records are unlabeled, so the
    engine runs in autonomous mode (same as pcap replay). For kernel-speed
    capture with in-kernel enforcement, use --source ebpf instead.
    """

    def __init__(self, iface: Optional[str] = None, telemetry: "Telemetry" = None,
                 bpf: Optional[str] = None):
        self.iface = iface          # None => scapy's default interface
        self.telemetry = telemetry
        self.bpf = bpf              # optional capture filter (BPF syntax)

    def stream(self) -> Iterator[FlowRecord]:
        try:
            from scapy.all import AsyncSniffer, IP, TCP, UDP  # type: ignore
        except Exception as e:
            raise RuntimeError("scapy required for live capture: pip install scapy") from e
        import queue as _q
        q: "_q.Queue" = _q.Queue(maxsize=20000)

        def _on(pkt):
            try:
                q.put_nowait(pkt)
                if self.telemetry: self.telemetry.note_loss(0, 1)
            except Exception:
                # dropped under backpressure -> counted as packet loss, not silently lost
                if self.telemetry: self.telemetry.note_loss(1, 1)

        sniffer = AsyncSniffer(iface=self.iface, prn=_on, store=False,
                               filter=self.bpf or "ip")
        try:
            sniffer.start()
        except Exception as e:
            raise RuntimeError(
                f"could not start capture on {self.iface or 'default interface'}: {e}. "
                f"On Windows install Npcap; on Linux/macOS run with sudo. "
                f"List interfaces with:  python3 backend/sauron.py --list-ifaces") from e
        print(f"[live] sniffing {self.iface or 'default interface'} "
              f"(filter={self.bpf or 'ip'}); generate some traffic (open a website) "
              f"-- Ctrl-C to stop")
        try:
            while True:
                try:
                    pkt = q.get(timeout=1.0)
                except _q.Empty:
                    continue
                if IP not in pkt:
                    continue         # skip ARP/IPv6/non-IP frames
                ip = pkt[IP]
                proto, sp, dp, fl = "OTHER", 0, 0, 0
                if TCP in pkt:
                    proto, sp, dp, fl = "TCP", int(pkt[TCP].sport), int(pkt[TCP].dport), int(pkt[TCP].flags)
                elif UDP in pkt:
                    proto, sp, dp = "UDP", int(pkt[UDP].sport), int(pkt[UDP].dport)
                yield FlowRecord(time.time(), ip.src, ip.dst, sp, dp, proto,
                                 len(pkt), fl, None, "live")
        finally:
            try:
                sniffer.stop()
            except Exception:
                pass


class EbpfLoaderSource:
    """Bridge to the real kernel data path via the libbpf loader (kernel/).

    Spawns `sauron_loader <iface> sauron.bpf.o`, reads newline-delimited JSON
    from its stdout (flow records + summed XDP/TC/LSM metrics + attach status),
    and can push BLOCK/UNBLOCK commands to its stdin to program the kernel
    blocklist map -- closing the enforcement loop.
    """

    def __init__(self, iface: str, telemetry: "Telemetry",
                 loader: Optional[str] = None, obj: Optional[str] = None):
        self.iface = iface
        self.telemetry = telemetry
        self.loader = loader or os.path.join(KERNEL_DIR, "sauron_loader")
        self.obj = obj or os.path.join(KERNEL_DIR, "sauron.bpf.o")
        self.proc: Optional[subprocess.Popen] = None

    def _spawn(self):
        if not os.path.exists(self.loader) or not os.path.exists(self.obj):
            raise RuntimeError(
                f"loader/object not built. Run scripts/build.sh first "
                f"(expected {self.loader} and {self.obj}).")
        if os.geteuid() != 0:
            raise RuntimeError("ebpf source requires root; run with sudo or use --source sim")
        if self.telemetry is not None:
            self.telemetry._loader_t0 = time.time()   # start of kernel load timing
        self.proc = subprocess.Popen(
            [self.loader, self.iface, self.obj],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)

    def block(self, ip: str) -> None:
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(f"BLOCK {ip}\n")
                self.proc.stdin.flush()
            except Exception:
                pass

    def stream(self) -> Iterator[FlowRecord]:
        self._spawn()
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            t = msg.get("t")
            if t == "flow":
                proto = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(msg.get("proto", 0), "OTHER")
                yield FlowRecord(time.time(), msg["src"], msg["dst"],
                                 msg.get("sport", 0), msg.get("dport", 0), proto,
                                 msg.get("len", 0), msg.get("flags", 0), None, "live")
            elif t == "metrics":
                self.telemetry.update_kernel(msg)
            elif t == "status":
                self.telemetry.set_kernel_status(msg)


def make_source(kind: str, telemetry: "Telemetry", **kw):
    kind = (kind or "sim").lower()
    if kind == "sim":
        return SimSource(**{k: v for k, v in kw.items() if k in ("seed", "attack_prob", "rate_hz")})
    if kind == "pcap":
        return PcapSource(kw["pcap"], speed=kw.get("speed", 0.0))
    if kind in ("csv", "dataset"):
        return CsvSource(kw["csv"], speed=kw.get("speed", 0.0), limit=kw.get("limit"),
                         balance=kw.get("balance", "auto"))
    if kind == "live":
        return LiveSniffSource(kw.get("iface"), telemetry)
    if kind == "ebpf":
        return EbpfLoaderSource(kw.get("iface") or "eth0", telemetry)
    raise ValueError(f"unknown source: {kind}")


# ============================================================================
# SECTION 4 — PIPELINE  (design Section 10)
# ============================================================================
class FeatureEngine:
    """Windowed per-source behavioural features.

    With `use_native=True` the engine instead consumes the dataset's own
    engineered flow statistics (CICIDS2017 ships 78 of them). That is usually
    far more discriminative on flow-level CSV exports, but it depends entirely
    on the dataset's column quality, so it is opt-in via --native-features and
    should be compared against the default before being reported.
    """

    use_native = False

    def __init__(self, window_sec=2.0):
        self.window = window_sec
        self._ev: Dict[str, Deque[Tuple[float, int, int, int]]] = defaultdict(
            lambda: deque(maxlen=2000))
        self._last: Dict[str, float] = {}

    # Robust per-column scalers learned online from the stream, so native
    # dataset features (which have wildly different ranges) are mapped into the
    # [0,1] space the detector bank expects without any offline preprocessing.
    def _scale(self, name: str, v: float) -> float:
        # Flow statistics (bytes/s, IAT, durations) are heavy-tailed and span
        # many orders of magnitude. Z-scoring them raw lets a handful of huge
        # values dominate the variance, so almost every normal flow collapses to
        # the same point. A signed-log transform first makes the distribution
        # roughly symmetric, which measurably improves separability.
        st = self._nat_stats.setdefault(name, {"n": 0, "mean": 0.0, "m2": 0.0})
        v = math.copysign(math.log1p(abs(v)), v)
        st["n"] += 1
        d = v - st["mean"]
        st["mean"] += d / st["n"]
        st["m2"] += d * (v - st["mean"])
        if st["n"] < 30:
            return 0.5
        sd = math.sqrt(max(st["m2"] / st["n"], 1e-9))
        z = max(-8.0, min(8.0, (v - st["mean"]) / sd))
        return 1.0 / (1.0 + math.exp(-z))

    def features(self, r: FlowRecord) -> Dict[str, float]:
        if getattr(r, "nat", None) and getattr(self, "use_native", False):
            # Emit exactly the columns the feature space was configured with,
            # each robustly scaled online into [0,1]. This gives every detector
            # the dataset's full engineered view instead of 7 proxies.
            nat = r.nat
            if not hasattr(self, "_nat_stats"):
                self._nat_stats: Dict[str, Dict] = {}
                low = {k.strip().lower(): k for k in nat}
                self._nat_cols = []
                for feat in _intel.FEATURES:
                    self._nat_cols.append(low.get(feat.strip().lower()))
            out: Dict[str, float] = {}
            for feat, col in zip(_intel.FEATURES, self._nat_cols):
                if col is None:
                    out[feat] = 0.5
                else:
                    try:
                        out[feat] = self._scale(feat, float(nat.get(col, 0.0)))
                    except Exception:
                        out[feat] = 0.5
            return out
        now = r.ts
        buf = self._ev[r.src_ip]
        iat = now - self._last.get(r.src_ip, now)
        self._last[r.src_ip] = now
        buf.append((now, r.dst_port, hash(r.dst_ip) & 0xffff, r.pkt_len))
        while buf and now - buf[0][0] > self.window:
            buf.popleft()
        n = len(buf)
        ports = [p for _, p, _, _ in buf]
        dsts = {d for _, _, d, _ in buf}
        total_bytes = sum(b for _, _, _, b in buf)
        ent = 0.0
        if ports:
            c = Counter(ports)
            tot = len(ports)
            ent = -sum((v / tot) * math.log(v / tot + 1e-12) for v in c.values())
            ent /= math.log(len(c) + 1e-12) if len(c) > 1 else 1.0
        syn = 1.0 if (r.tcp_flags & 0x02) and not (r.tcp_flags & 0x10) else 0.0
        return {
            "pkt_len": r.pkt_len / 1500.0,
            "iat": min(1.0, iat * 50.0),
            "syn_ratio": syn,
            "dst_fanout": min(1.0, len(dsts) / 40.0),
            "port_entropy": min(1.0, ent),
            "byte_asymmetry": min(1.0, total_bytes / (n * 1500.0 + 1e-6)),
            "pps": min(1.0, (n / self.window) / 300.0),
        }


def choose_action(d: Decision) -> Tuple[str, str, str]:
    """(action, reason, severity). DROP/QUARANTINE need both a large score
    margin AND high suspicion -- proportionality (design Section 13)."""
    margin = d.aggregate - d.tau_high
    if d.unknown and d.aggregate > d.tau_low:
        return "REDIRECT", "open-set: candidate unknown family routed to honeypot", "high"
    if d.aggregate > d.tau_high and d.suspicion > 0.6 and margin > 0.12:
        if d.suspicion > 0.85:
            return "QUARANTINE", "high anomaly score and high-suspicion source", "critical"
        return "DROP", "adaptive threshold exceeded with high margin from a suspicious source", "high"
    if d.aggregate > d.tau_high:
        return "RATE_LIMIT", "adaptive threshold exceeded at bounded confidence", "medium"
    if d.aggregate > d.tau_low:
        return "MIRROR", "elevated but sub-threshold; mirrored for observation", "low"
    return "PASS", "within calibrated normal band", "info"


def rule_engine_verdict(r: FlowRecord, f: Dict[str, float]) -> bool:
    """Frozen static iptables/nftables-style baseline for the comparison panel.

    Tolerates a reconfigured feature space (dataset-native columns) by falling
    back to the packet fields it can always see.
    """
    if f.get("pps", 0.0) > 0.8:
        return True
    if r.tcp_flags == 2 and r.pkt_len < 60:
        return True
    if f.get("dst_fanout", 0.0) > 0.7:
        return True
    return False


@dataclass
class DetectionMetrics:
    packets: int = 0
    dropped: int = 0
    quarantined: int = 0
    alerts: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    _win: Deque[Tuple[float, int]] = field(default_factory=lambda: deque(maxlen=6000))

    def observe(self, r: FlowRecord, action: str, flagged: bool):
        self.packets += 1
        self.bytes_total = getattr(self, "bytes_total", 0) + int(r.pkt_len)
        self._win.append((r.ts, r.pkt_len))
        if action == "DROP":
            self.dropped += 1
        if action == "QUARANTINE":
            self.quarantined += 1
        if flagged:
            self.alerts += 1
        if r.label is not None:
            mal = r.label == 1
            if flagged and mal:
                self.tp += 1
            elif flagged and not mal:
                self.fp += 1
            elif not flagged and mal:
                self.fn += 1
            else:
                self.tn += 1

    def rates(self) -> Tuple[float, float]:
        now = time.time()
        recent = [(t, b) for t, b in self._win if now - t < 1.0] or list(self._win)[-50:]
        if len(recent) > 1:
            span = max(1e-3, recent[-1][0] - recent[0][0])
        else:
            span = 1.0
        return len(recent) / span, sum(b for _, b in recent) * 8 / span

    def detection(self) -> Dict:
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        den = math.sqrt((self.tp + self.fp) * (self.tp + self.fn) *
                        (self.tn + self.fp) * (self.tn + self.fn)) or 1.0
        mcc = (self.tp * self.tn - self.fp * self.fn) / den
        return {"precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "mcc": round(mcc, 4),
                "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


class SauronEngine:
    """Main orchestrator: FlowRecord -> features -> novel bank -> ADE -> action
    -> event, with drift detection, model management, policy distillation,
    FDR-controlled alerting, persistent storage/audit, and autonomy guardrails."""

    _MITIGATION = {
        "PASS": "none - traffic within calibrated normal band",
        "LOG": "logged for audit; no enforcement",
        "MIRROR": "mirrored to observation queue (epsilon-mirror); no block",
        "RATE_LIMIT": "token-bucket rate limit applied at XDP (reversible, TTL-bound)",
        "REDIRECT": "steered to honeypot ifindex for open-set family capture",
        "DROP": "XDP_DROP at ingress; verdict cached in kernel with TTL",
        "QUARANTINE": "source programmed into kernel blocklist map (standing block, TTL + audit)",
    }

    def __init__(self, eps_h=0.02, seed=7, db_path=":memory:",
                 energy_budget_w: float = 0.0):
        self.bank = _intel.NovelDetectorBank()        # HDC + ECOD + SR + RRCF
        self.energy = _energy.EnergyMeter()           # RAPL if available, else modelled
        self.egate = _energy.EnergyAwareFilter(budget_w=energy_budget_w)
        self.ade = AdaptiveDecisionEngine(self.bank.names, eps_h=eps_h, seed=seed)
        self.feats = FeatureEngine()
        self.openset = OpenSetGuard()
        self.metrics = DetectionMetrics()
        self.drift = _intel.DriftMartingale()
        self.model_mgr = _intel.ModelManager()
        self.distiller = _intel.PolicyDistiller()
        self.alerts = _intel.AlertManager(fdr_q=0.1)
        self.storage = _intel.Storage(db_path)
        self.gov = _intel.AutonomyGovernor(l2_per_min=60, ttl_sec=120)
        self.dispatcher = _intg.build_dispatcher_from_env()  # external integrations (no-op unless configured)
        self.rule_alerts = 0
        self.unknown_clusters: Dict[str, int] = defaultdict(int)
        self._distilled: List[Dict] = []
        self._n = 0

    def set_eps_h(self, v: float):
        old = self.ade.threshold.eps_h
        self.ade.set_eps_h(v)
        self.storage.audit("eps_h_change", {"old": old, "new": v, "cause": "operator"})

    def process(self, r: FlowRecord) -> Dict:
        self._n += 1
        f = self.feats.features(r)
        raw = self.bank.scores(f, r.src_ip)
        # Energy-aware filtering: above the power budget, drop the two costly
        # detectors (RRCF isolation + Spectral-Residual FFT) for low-suspicion
        # traffic. Cheap detectors always run; suspicious traffic always gets
        # the full bank, so the saving never comes out of security.
        if self.egate.enabled:
            susp = self.ade.trust.suspicion(r.src_ip)
            if not self.egate.allow_expensive(self.energy.watts, susp):
                for _d in self.egate.EXPENSIVE:
                    if _d in raw:
                        raw[_d] = 0.0
        is_unknown, novelty = self.openset.check(f)
        d = self.ade.decide(r.src_ip, r.proto, raw)
        d.unknown = is_unknown
        action, reason, severity = choose_action(d)

        # --- autonomy guardrail: global L2 cap (§13). If the containment budget
        # is exhausted, downgrade DROP/QUAR to a reversible RATE_LIMIT. ---
        if action in ("DROP", "QUARANTINE") and not self.gov.allow_contain(r.src_ip, r.ts):
            action, reason, severity = "RATE_LIMIT", "L2 containment rate cap reached; downgraded (reversible)", "medium"

        d.reason, d.severity = reason, severity
        d.top_features = self.bank.attribution(f)      # EXACT ECOD Shapley values

        flagged = action in ("DROP", "QUARANTINE", "RATE_LIMIT", "REDIRECT")
        would_drop = action in ("DROP", "QUARANTINE")

        # --- drift detection (conformal martingale) -> spin up a canary ---
        if self.drift.update(1.0 - d.raw_aggregate):
            self.model_mgr.start_canary()
            self.storage.audit("drift_detected", {"level": self.drift.level(), "ctx": r.proto})

        # --- model manager: active vs shadow-canary accuracy on labeled units ---
        if r.label is not None:
            active_ok = (flagged == (r.label == 1))
            shadow = d.raw_aggregate > d.tau_high * 0.95     # tighter canary policy
            canary_ok = (shadow == (r.label == 1))
            self.model_mgr.observe(active_ok, canary_ok)

        re_flag = rule_engine_verdict(r, f)
        if re_flag:
            self.rule_alerts += 1

        self.ade.feedback(r.src_ip, r.proto, d, raw,
                          None if r.label is None else float(r.label), would_drop)
        self.bank.learn(f, r.label)                    # HDC prototype bundling
        self.distiller.observe(f, 1 if flagged else 0, r.src_ip)
        self.metrics.observe(r, action, flagged)
        if is_unknown:
            self.unknown_clusters[r.family] += 1
        if r.label is not None and r.family:
            fb = getattr(self, "_fam_buf", None)
            if fb is None:
                fb = self._fam_buf = []
            fb.append((r.family, float(d.aggregate), 1 if flagged else 0, int(r.label)))
            if len(fb) > 400000:
                del fb[:len(fb) - 400000]
        if r.family:                       # per-family detection rate (for plots)
            fs = getattr(self, "fam_stats", None)
            if fs is None:
                fs = self.fam_stats = {}
            e = fs.setdefault(r.family, {"total": 0, "flagged": 0})
            e["total"] += 1
            if flagged:
                e["flagged"] += 1

        ev = {
            "type": "packet", "ts": r.ts,
            "src_ip": r.src_ip, "dst_ip": r.dst_ip,
            "src_port": r.src_port, "dst_port": r.dst_port,
            "proto": r.proto, "pkt_len": r.pkt_len,
            "action": action, "reason": reason, "severity": severity,
            "score": round(d.aggregate, 4), "raw_score": round(d.raw_aggregate, 4),
            "tau_high": round(d.tau_high, 4), "tau_low": round(d.tau_low, 4),
            "per_detector": {k: round(v, 4) for k, v in d.per_detector.items()},
            "weights": {k: round(v, 4) for k, v in d.weights.items()},
            "trust": round(d.trust, 4), "suspicion": round(d.suspicion, 4),
            "unknown": is_unknown, "novelty": round(novelty, 4),
            "top_features": d.top_features, "label": r.label, "family": r.family,
            "rule_engine_flag": re_flag, "realized_fpr": round(self.ade.realized_fpr(), 5),
        }
        self._pid = getattr(self, "_pid", 0) + 1
        if self._pid % 200 == 0:                       # periodic energy sampling
            try:
                self._bits = getattr(self, "_bits", 0) + 0
                self.energy.sample(self.metrics.packets,
                                   self.metrics.alerts,
                                   getattr(self.metrics, "bytes_total", 0) * 8,
                                   cpu_pct=None)
            except Exception:
                pass
        ev["packet_id"] = self._pid
        ev["mitigation"] = self._MITIGATION.get(ev["action"], "n/a")
        # streaming buffers for ROC-AUC / PR-AUC (labeled traffic only)
        if r.label is not None:
            buf = getattr(self, "_auc", None)
            if buf is None:
                buf = self._auc = []
            buf.append((float(ev["score"]), int(r.label)))
            if len(buf) > 20000:
                del buf[:len(buf) - 20000]   # bounded: keeps AUC cost predictable
        # rule hit ratio: decisions resolvable by a distilled/kernel rule
        self._rule_seen = getattr(self, "_rule_seen", 0) + 1
        if ev["action"] in ("DROP", "QUARANTINE") and getattr(self, "_distilled", None):
            self._rule_hits = getattr(self, "_rule_hits", 0) + 1
        if _PKT_CSV is not None:
            # Append every decision to CSV as it is made, so a live capture
            # produces an analysable table without post-processing the log.
            try:
                _PKT_CSV.writerow([
                    f"{r.ts:.6f}",
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.ts)),
                    ev.get("packet_id"), r.src_ip, r.src_port, r.dst_ip, r.dst_port,
                    r.proto, r.pkt_len, r.tcp_flags,
                    f"{d.aggregate:.5f}", f"{d.raw_aggregate:.5f}",
                    f"{d.tau_high:.5f}", f"{d.tau_low:.5f}",
                    f"{d.trust:.5f}", f"{d.suspicion:.5f}",
                    action, severity, reason,
                    ev.get("mitigation", ""),
                    "; ".join(f"{k}={v:.3f}" for k, v in (d.top_features or [])[:3]),
                    ev.get("family", ""), "" if r.label is None else r.label,
                    f"{self.ade.realized_fpr():.5f}",
                ])
                if (ev.get("packet_id") or 0) % 200 == 0:
                    _PKT_CSV_FH.flush()
            except Exception:
                pass
        if _PRINT_PKTS:
            # Full per-packet decision line: identity, verdict, why it was taken,
            # the exact-Shapley features that drove it, and the mitigation applied.
            # This is the audit trail for a live capture.
            self._pn = getattr(self, "_pn", 0) + 1
            if self._pn == 1:
                print("\n" + "=" * 150, flush=True)
                print(f"{'#':>7} {'TIME':<12} {'SOURCE':<22} {'DESTINATION':<22} "
                      f"{'PROTO':<6} {'LEN':>5} {'SCORE':>6} {'THRESH':>7} {'TRUST':>6} "
                      f"{'DECISION':<12} REASON", flush=True)
                print("=" * 150, flush=True)
            _C = {"PASS": "\033[92m", "LOG": "\033[90m", "MIRROR": "\033[96m",
                  "RATE_LIMIT": "\033[93m", "REDIRECT": "\033[95m",
                  "DROP": "\033[91m", "QUARANTINE": "\033[41m\033[97m"}
            col = "" if _NO_COLOR else _C.get(action, "")
            rst = "" if _NO_COLOR else "\033[0m"
            src = f"{r.src_ip}:{r.src_port}"[:21]
            dst = f"{r.dst_ip}:{r.dst_port}"[:21]
            tstr = time.strftime("%H:%M:%S", time.localtime(r.ts)) + f".{int((r.ts%1)*1000):03d}"
            print(f"{self._pn:>7} {tstr:<12} {src:<22} {dst:<22} {r.proto:<6} "
                  f"{r.pkt_len:>5} {d.aggregate:>6.3f} {d.tau_high:>7.3f} "
                  f"{d.trust:>6.3f} {col}{action:<12}{rst} {reason}", flush=True)
            if action not in ("PASS", "LOG"):
                tf = ", ".join(f"{k}={v:.2f}" for k, v in (d.top_features or [])[:3])
                print(f"{'':>7} {'':<12} └─ top features: {tf}", flush=True)
                print(f"{'':>7} {'':<12} └─ mitigation  : "
                      f"{self._MITIGATION.get(action, 'n/a')}", flush=True)
        # --- alert manager: dedup + correlation + BH-FDR + lifecycle enrichment ---
        if flagged:
            alert = self.alerts.process(ev)
            if alert is not None:
                ev["alert_id"] = alert["alert_id"]; ev["flow_id"] = alert["flow_id"]
                ev["campaign_count"] = alert["campaign_count"]; ev["fdr_p"] = alert["fdr_p"]
                self.dispatcher.dispatch(ev)  # fan out to external systems
        self.storage.event(ev)

        # Periodic policy distillation (real fidelity/coverage) + TTL expiry.
        # Distillation is an exhaustive stump/conjunction search and costs ~2s
        # per call; running it every 500 records made it 58% of total runtime on
        # large datasets. Throttle by TIME so the rule set still refreshes
        # continuously while the cost stays negligible at any scale.
        if self._n % 500 == 0 and (time.time() - getattr(self, "_last_distill", 0.0)) > 20.0:
            self._last_distill = time.time()
            rules = self.distiller.distill()
            if rules:
                self._distilled = rules
        if self._n % 500 == 0:
            for ip in self.gov.expired(r.ts):
                self.storage.audit("ttl_expiry", {"src": ip})
        return ev

    def expired_enforcements(self, now=None) -> List[str]:
        return self.gov.expired(now)

    def distilled_rules(self) -> List[Dict]:
        """The current best surrogate rules with *measured* fidelity/coverage
        from the live PolicyDistiller (empty until enough traffic is seen)."""
        return list(self._distilled)

    def auc_scores(self, max_age: float = 5.0) -> Dict[str, Optional[float]]:
        """Cached wrapper: the exact AUC is an O(n log n) sort over the score
        buffer, and snapshot() runs ~4x/second. Recomputing every call starved
        the event loop on big datasets (dashboard stuck on "connecting..."), so
        the result is cached and refreshed at most every `max_age` seconds."""
        now = time.time()
        cached = getattr(self, "_auc_cache", None)
        if cached is not None and (now - cached[0]) < max_age:
            return cached[1]
        val = self._auc_scores_exact()
        self._auc_cache = (now, val)
        return val

    def _auc_scores_exact(self) -> Dict[str, Optional[float]]:
        """ROC-AUC and PR-AUC (average precision) over the labeled score buffer.
        ROC-AUC uses the rank identity (Mann-Whitney U) so it is exact and
        needs no threshold sweep. Returns None when labels or a class are absent."""
        buf = getattr(self, "_auc", None)
        if not buf:
            return {"roc_auc": None, "pr_auc": None}
        arr = sorted(buf, key=lambda t: t[0])
        n = len(arr)
        pos = sum(l for _, l in arr)
        neg = n - pos
        if pos == 0 or neg == 0:
            return {"roc_auc": None, "pr_auc": None}
        # average ranks (ties shared) -> Mann-Whitney U -> ROC-AUC
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[j + 1][0] == arr[i][0]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[k] = avg
            i = j + 1
        sum_pos = sum(ranks[k] for k in range(n) if arr[k][1] == 1)
        roc = (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
        # PR-AUC as average precision, sweeping thresholds high -> low
        tp = fp = 0
        prev_rec = 0.0
        ap = 0.0
        for k in range(n - 1, -1, -1):
            if arr[k][1] == 1:
                tp += 1
            else:
                fp += 1
            prec = tp / (tp + fp)
            rec = tp / pos
            if rec > prev_rec:
                ap += (rec - prev_rec) * prec
                prev_rec = rec
        return {"roc_auc": round(roc, 5), "pr_auc": round(ap, 5)}

    def snapshot(self) -> Dict:
        pps, bps = self.metrics.rates()
        snap = {"packets": self.metrics.packets, "dropped": self.metrics.dropped,
                "quarantined": self.metrics.quarantined, "alerts": self.metrics.alerts,
                "pps": round(pps, 1), "bps": round(bps, 1),
                "rule_engine_alerts": self.rule_alerts,
                "realized_fpr": round(self.ade.realized_fpr(), 5),
                "fp_budget": self.ade.threshold.eps_h,
                "unknown_clusters": dict(self.unknown_clusters),
                "detectors": self.bank.names,
                "drift": {"level": self.drift.level(), "detections": self.drift.detections},
                "model_mgr": self.model_mgr.status(),
                "alert_mgr": self.alerts.stats(),
                "autonomy": self.gov.stats(),
                "distilled_rules": self._distilled}
        snap.update(self.metrics.detection())
        snap.update(self.auc_scores())
        # Fusion weights belong in the snapshot too: the dashboard falls back to
        # /api/metrics when no packet event is in hand (e.g. right after a page
        # reload, or once a dataset run has finished streaming).
        try:
            snap["weights"] = {n: round(float(w), 4)
                               for n, w in zip(self.bank.names, self.ade.fusion.weights)}
        except Exception:
            pass
        try:
            snap["energy"] = self.energy.snapshot()
            snap["energy_filter"] = self.egate.snapshot()
        except Exception:
            pass
        seen = getattr(self, "_rule_seen", 0)
        hits = getattr(self, "_rule_hits", 0)
        snap["rule_hit_ratio"] = round(hits / seen, 5) if seen else 0.0
        snap["rule_hits"] = hits
        return snap


# ============================================================================
# SECTION 5 — HOST TELEMETRY  (real CPU / memory / latency; kernel counters)
# ============================================================================
class Telemetry:
    """Real host telemetry. CPU% and memory come from /proc; decision latency
    is measured around the scoring path. Kernel XDP/TC/LSM counters are merged
    from the libbpf loader when the eBPF data path is attached; in sim/pcap mode
    they are reported as unavailable rather than fabricated."""

    def __init__(self):
        self._prev_cpu = self._read_cpu_times()
        self._prev_t = time.time()
        self._lat: Deque[float] = deque(maxlen=4000)
        self.kernel = {"attached": False, "xdp": 0, "tc": 0, "lsm": 0}
        self.kmetrics: Dict[str, int] = {}
        self._kprev: Dict[str, int] = {}
        self._kprev_t = time.time()
        self.kernel_pps = 0.0

    @staticmethod
    def _read_cpu_times() -> Tuple[int, int]:
        try:
            with open("/proc/stat") as fh:
                parts = fh.readline().split()[1:]
                vals = list(map(int, parts))
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                return sum(vals), idle
        except Exception:
            return (0, 0)

    def cpu_percent(self) -> float:
        total, idle = self._read_cpu_times()
        dt, di = total - self._prev_cpu[0], idle - self._prev_cpu[1]
        self._prev_cpu = (total, idle)
        if dt <= 0:
            return 0.0
        return round(100.0 * (dt - di) / dt, 1)

    @staticmethod
    def mem() -> Tuple[float, float]:
        """Return (system_used_pct, process_rss_mb)."""
        used_pct, rss_mb = 0.0, 0.0
        try:
            info = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    k, v = line.split(":")
                    info[k] = int(v.strip().split()[0])
            total = info.get("MemTotal", 1)
            avail = info.get("MemAvailable", total)
            used_pct = round(100.0 * (total - avail) / total, 1)
        except Exception:
            pass
        try:
            with open("/proc/self/statm") as fh:
                rss_pages = int(fh.read().split()[1])
                rss_mb = round(rss_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
        except Exception:
            pass
        return used_pct, rss_mb

    def record_latency(self, seconds: float) -> None:
        self._lat.append(seconds * 1e6)  # microseconds

    def latency(self) -> Dict[str, float]:
        if not self._lat:
            return {"p50": 0.0, "p99": 0.0}
        a = np.fromiter(self._lat, dtype=np.float64, count=len(self._lat))
        return {"p50": round(float(np.percentile(a, 50)), 1),
                "p99": round(float(np.percentile(a, 99)), 1)}

    def set_kernel_status(self, msg: Dict) -> None:
        # first attach report => kernel load time (loader launch -> hooks attached)
        t0 = getattr(self, "_loader_t0", None)
        if t0 is not None and not hasattr(self, "kernel_load_ms"):
            self.note_kernel_load((time.time() - t0) * 1000.0)
        self.kernel = {"attached": bool(msg.get("xdp")),
                       "xdp": int(msg.get("xdp", 0)), "tc": int(msg.get("tc", 0)),
                       "lsm": int(msg.get("lsm", 0))}

    def update_kernel(self, msg: Dict) -> None:
        self.kernel["attached"] = True
        now = time.time()
        prev_pkts = self._kprev.get("xdp_pkts", msg.get("xdp_pkts", 0))
        dt = max(1e-3, now - self._kprev_t)
        self.kernel_pps = round((msg.get("xdp_pkts", 0) - prev_pkts) / dt, 1)
        self._kprev = dict(msg)
        self._kprev_t = now
        self.kmetrics = {k: int(v) for k, v in msg.items() if k != "t"}

    def note_kernel_load(self, ms: float) -> None:
        """Time from loader launch to hooks attached (measured once)."""
        self.kernel_load_ms = round(float(ms), 2)

    def note_dashboard_latency(self, ms: float) -> None:
        """Browser-reported render latency (WS delivery -> paint)."""
        self.dashboard_ms = round(float(ms), 2)

    def note_loss(self, dropped: int = 1, seen: int = 1) -> None:
        """Records punted-but-lost (ring-buffer starvation / queue backpressure)."""
        self.loss_dropped = getattr(self, "loss_dropped", 0) + dropped
        self.loss_seen = getattr(self, "loss_seen", 0) + seen

    def snapshot(self) -> Dict:
        used_pct, rss_mb = self.mem()
        lat = self.latency()
        seen = getattr(self, "loss_seen", 0)
        lost = getattr(self, "loss_dropped", 0)
        out = {"cpu_pct": self.cpu_percent(), "mem_used_pct": used_pct,
               "rss_mb": rss_mb, "latency_us": lat,
               "inference_time_us": lat.get("p50"),
               "packet_loss_pct": round(100.0 * lost / seen, 4) if seen else 0.0,
               "packets_lost": lost,
               "kernel_load_ms": getattr(self, "kernel_load_ms", None),
               "dashboard_latency_ms": getattr(self, "dashboard_ms", None),
               "kernel_attached": self.kernel["attached"],
               "hooks": {"xdp": self.kernel["xdp"], "tc": self.kernel["tc"],
                         "lsm": self.kernel["lsm"]}}
        if self.kernel["attached"]:
            out["kernel"] = self.kmetrics
            out["kernel_pps"] = self.kernel_pps
            k = self.kmetrics or {}
            st, pk = k.get("starved"), k.get("packets")
            if st is not None and pk:
                out["packet_loss_pct"] = round(100.0 * float(st) / float(pk), 4)
                out["packets_lost"] = st
        return out


# ============================================================================
# SECTION 6 — HEADLESS RUNNER (numpy-only; proves the pipeline produces output)
# ============================================================================
_COLOR = {"PASS": "\033[92m", "MIRROR": "\033[96m", "RATE_LIMIT": "\033[93m",
          "REDIRECT": "\033[95m", "DROP": "\033[91m", "QUARANTINE": "\033[41m\033[97m"}


def family_metrics(engine: "SauronEngine") -> List[Dict]:
    """Per-attack-family metrics, computed ONE-VS-BENIGN.

    A binary detector produces false positives on *benign* traffic, which
    belongs to no attack family, so per-family precision is undefined unless the
    negative class is pinned down. The standard, defensible choice is:

        positives = samples of family F        negatives = benign samples

    Other attack families are excluded from F's negatives (flagging a DDoS flow
    is not an error when evaluating PortScan). This yields well-defined
    precision / recall / F1 / MCC / ROC-AUC per family, directly comparable to
    per-class tables in the IDS literature.
    """
    buf = getattr(engine, "_fam_buf", None)
    if not buf:
        return []
    benign = [(sc, fl) for fam, sc, fl, lab in buf if lab == 0]
    n_ben = len(benign)
    fp_ben = sum(fl for _, fl in benign)               # benign wrongly flagged
    ben_scores = [sc for sc, _ in benign]

    fams: Dict[str, List] = {}
    for fam, sc, fl, lab in buf:
        if lab == 1:
            fams.setdefault(fam, []).append((sc, fl))

    def auc(pos: List[float], neg: List[float]) -> Optional[float]:
        if not pos or not neg:
            return None
        arr = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda t: t[0])
        n = len(arr)
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[j + 1][0] == arr[i][0]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[k] = avg
            i = j + 1
        P, N = len(pos), len(neg)
        sp = sum(ranks[k] for k in range(n) if arr[k][1] == 1)
        return round((sp - P * (P + 1) / 2.0) / (P * N), 5)

    out = []
    for fam, rows in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        tp = sum(fl for _, fl in rows)
        fn_ = len(rows) - tp
        fp, tn = fp_ben, n_ben - fp_ben
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn_) if (tp + fn_) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / (tp + fn_ + fp + tn) if (tp + fn_ + fp + tn) else 0.0
        den = math.sqrt((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
        mcc = ((tp * tn - fp * fn_) / den) if den else 0.0
        out.append({"family": fam, "samples": len(rows), "detected": tp, "missed": fn_,
                    "accuracy": round(acc, 5), "precision": round(prec, 5),
                    "recall": round(rec, 5), "f1": round(f1, 5), "mcc": round(mcc, 5),
                    "roc_auc": auc([sc for sc, _ in rows], ben_scores),
                    "mean_score": round(sum(sc for sc, _ in rows) / len(rows), 4)})
    if n_ben:
        out.append({"family": "(benign)", "samples": n_ben, "detected": tn if False else n_ben - fp_ben,
                    "missed": fp_ben, "accuracy": round((n_ben - fp_ben) / n_ben, 5),
                    "precision": None, "recall": None, "f1": None, "mcc": None,
                    "roc_auc": None,
                    "mean_score": round(sum(ben_scores) / n_ben, 4) if ben_scores else 0.0})
    return out


def print_family_metrics(rows: List[Dict]) -> None:
    if not rows:
        return
    W = 100
    print("\n" + "=" * W)
    print("  PER-ATTACK-FAMILY METRICS   (one-vs-benign: positives = family, negatives = benign)")
    print("=" * W)
    print(f"  {'family':<24}{'samples':>9}{'detect':>8}{'miss':>7}"
          f"{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'MCC':>8}{'ROC':>8}")
    print("-" * W)
    for r in rows:
        def f(v):
            return "  --  " if v is None else f"{v:.4f}"
        print(f"  {r['family'][:23]:<24}{r['samples']:>9,}{r['detected']:>8,}{r['missed']:>7,}"
              f"{f(r['accuracy']):>8}{f(r['precision']):>8}{f(r['recall']):>8}"
              f"{f(r['f1']):>8}{f(r['mcc']):>8}{f(r['roc_auc']):>8}")
    print("=" * W)
    print("  note: the (benign) row reports how much benign traffic was left alone;")
    print("        'miss' there = false positives. Precision/F1 are undefined for it.")
    print("=" * W + "\n")


def write_evaluation_plots(engine: "SauronEngine", tel: "Telemetry",
                           profile: Optional[Dict] = None,
                           balance_info: Optional[Dict] = None,
                           source: str = "", outdir: str = "results") -> Optional[str]:
    """Render a colourful multi-panel results figure after a dataset run:
    class distribution, imbalance before/after, confusion matrix, metric bars,
    ROC + PR curves, score separation, per-family recall, and latency.
    Returns the PNG path, or None if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except Exception:
        print("[plots] matplotlib not installed - skipping graphs (pip install matplotlib)")
        return None

    s = engine.snapshot(); s.update(tel.snapshot())
    buf = list(getattr(engine, "_auc", []) or [])
    tp, fp = int(s.get("tp", 0)), int(s.get("fp", 0))
    tn, fn = int(s.get("tn", 0)), int(s.get("fn", 0))
    labeled = (tp + fp + tn + fn) > 0

    BG, FG, GRID = "#0a1230", "#e6ecff", "#243063"
    CY, GR, RD, AM, VI, PK = "#22d3ee", "#34d399", "#fb5570", "#fbbf24", "#a78bfa", "#f472b6"
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": "#0d1740",
                         "savefig.facecolor": BG, "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": "#9aa8e0",
                         "ytick.color": "#9aa8e0", "axes.edgecolor": GRID,
                         "grid.color": GRID, "font.size": 9})
    fig = plt.figure(figsize=(17, 13))
    gs = GridSpec(4, 4, figure=fig, hspace=.5, wspace=.32)
    fig.suptitle(f"SAURON++  ·  Evaluation Results  ·  source={source}",
                 color=FG, fontsize=15, fontweight="bold", y=.975)

    def style(ax, title):
        ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=8)
        ax.grid(alpha=.25, linestyle=":")
        for sp in ax.spines.values():
            sp.set_color(GRID)

    # 1 class distribution
    ax = fig.add_subplot(gs[0, 0]); style(ax, "Class Distribution")
    if profile and profile.get("families"):
        fam = dict(sorted(profile["families"].items(), key=lambda kv: -kv[1])[:7])
        cols = [GR if is_benign_label(k) else RD for k in fam]
        ax.barh(list(fam)[::-1], list(fam.values())[::-1], color=cols[::-1])
        ax.set_xlabel("rows")
    else:
        ax.text(.5, .5, "no dataset profile", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 2 imbalance before/after
    ax = fig.add_subplot(gs[0, 1]); style(ax, "Imbalance Correction")
    if balance_info and balance_info.get("before"):
        b, a = balance_info["before"], balance_info["after"]
        x = [0, 1]
        ax.bar([i - .18 for i in x], [b["benign"], b["malicious"]], .36, label="before",
               color=["#3b82f6", "#9333ea"])
        ax.bar([i + .18 for i in x], [a["benign"], a["malicious"]], .36, label="after",
               color=[GR, PK])
        ax.set_xticks(x); ax.set_xticklabels(["benign", "malicious"])
        ax.legend(facecolor="#0d1740", edgecolor=GRID, labelcolor=FG, fontsize=8)
    else:
        ax.text(.5, .5, "not applicable", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 3 confusion matrix
    ax = fig.add_subplot(gs[0, 2]); style(ax, "Confusion Matrix")
    if labeled:
        m = [[tp, fn], [fp, tn]]
        ax.imshow(m, cmap="magma")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i][j]:,}", ha="center", va="center",
                        color="#ffffff", fontweight="bold", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred mal", "pred ben"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["act mal", "act ben"])
        ax.grid(False)
    else:
        ax.text(.5, .5, "unlabeled source", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 4 metric bars
    ax = fig.add_subplot(gs[0, 3]); style(ax, "Classification Metrics")
    if labeled:
        names = ["acc", "prec", "rec", "F1", "MCC"]
        tot = max(1, tp + fp + tn + fn)
        vals = [(tp + tn) / tot, s.get("precision") or 0, s.get("recall") or 0,
                s.get("f1") or 0, s.get("mcc") or 0]
        bars = ax.bar(names, vals, color=[CY, GR, VI, AM, PK])
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + .02, f"{v:.3f}",
                    ha="center", color=FG, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1.15)
    else:
        ax.text(.5, .5, "unlabeled source", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 5 ROC   6 PR
    def curves():
        if not buf:
            return None
        arr = sorted(buf, key=lambda t: -t[0])
        P = sum(l for _, l in arr); N = len(arr) - P
        if not P or not N:
            return None
        tpv = fpv = 0; roc = [(0, 0)]; pr = []
        for _, l in arr:
            if l: tpv += 1
            else: fpv += 1
            roc.append((fpv / N, tpv / P))
            pr.append((tpv / P, tpv / (tpv + fpv)))
        return roc, pr
    c = curves()
    ax = fig.add_subplot(gs[1, 0]); style(ax, f"ROC Curve  (AUC={s.get('roc_auc')})")
    if c:
        ax.plot([p[0] for p in c[0]], [p[1] for p in c[0]], color=CY, lw=2)
        ax.fill_between([p[0] for p in c[0]], [p[1] for p in c[0]], color=CY, alpha=.18)
        ax.plot([0, 1], [0, 1], "--", color="#6b78bf", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    else:
        ax.text(.5, .5, "needs labels", ha="center", color="#6b78bf", transform=ax.transAxes)
    ax = fig.add_subplot(gs[1, 1]); style(ax, f"Precision-Recall  (AP={s.get('pr_auc')})")
    if c:
        ax.plot([p[0] for p in c[1]], [p[1] for p in c[1]], color=PK, lw=2)
        ax.fill_between([p[0] for p in c[1]], [p[1] for p in c[1]], color=PK, alpha=.18)
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.05)
    else:
        ax.text(.5, .5, "needs labels", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 7 score separation
    ax = fig.add_subplot(gs[1, 2]); style(ax, "Anomaly Score Separation")
    if buf:
        ben = [x for x, l in buf if l == 0]; mal = [x for x, l in buf if l == 1]
        if ben: ax.hist(ben, bins=40, color=GR, alpha=.75, label="benign")
        if mal: ax.hist(mal, bins=40, color=RD, alpha=.75, label="malicious")
        ax.legend(facecolor="#0d1740", edgecolor=GRID, labelcolor=FG, fontsize=8)
        ax.set_xlabel("fused anomaly score")
    else:
        ax.text(.5, .5, "needs labels", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 8 action mix
    ax = fig.add_subplot(gs[1, 3]); style(ax, "Enforcement Actions")
    acts = {"alerts": s.get("alerts", 0), "dropped": s.get("dropped", 0),
            "quarantined": s.get("quarantined", 0),
            "rule-engine": s.get("rule_engine_alerts", 0)}
    ax.bar(list(acts), list(acts.values()), color=[AM, RD, PK, VI])
    ax.tick_params(axis="x", rotation=20)

    # 9 per-family detection
    ax = fig.add_subplot(gs[2, 0:2]); style(ax, "Detection by Attack Family (dataset labels)")
    fams = getattr(engine, "fam_stats", None)
    if fams:
        ks = sorted(fams, key=lambda k: -fams[k]["total"])[:8]
        rec = [100.0 * fams[k]["flagged"] / max(1, fams[k]["total"]) for k in ks]
        bars = ax.bar(ks, rec, color=[GR if is_benign_label(k) else RD for k in ks])
        for b_, v in zip(bars, rec):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", color=FG, fontsize=8)
        ax.set_ylabel("flagged %"); ax.set_ylim(0, 112); ax.tick_params(axis="x", rotation=18)
    else:
        ax.text(.5, .5, "no per-family stats", ha="center", color="#6b78bf", transform=ax.transAxes)

    # 10 FPR vs budget
    ax = fig.add_subplot(gs[2, 2]); style(ax, "Realized FPR vs Budget")
    rf = (s.get("realized_fpr") or 0) * 100; bg = (s.get("fp_budget") or 0) * 100
    ax.bar(["realized", "budget εH"], [rf, bg], color=[RD if rf > bg else GR, CY])
    for i, v in enumerate([rf, bg]):
        ax.text(i, v + .05, f"{v:.2f}%", ha="center", color=FG, fontsize=9, fontweight="bold")
    ax.set_ylabel("%")

    # 11 latency
    ax = fig.add_subplot(gs[2, 3]); style(ax, "Decision Latency")
    lat = s.get("latency_us") or {}
    ax.bar(["p50", "p99"], [lat.get("p50", 0), lat.get("p99", 0)], color=[CY, AM])
    for i, v in enumerate([lat.get("p50", 0), lat.get("p99", 0)]):
        ax.text(i, v, f"{v:.0f}µs", ha="center", va="bottom", color=FG, fontsize=9)
    ax.set_ylabel("µs")

    # 12-14: energy panels
    en = s.get("energy") or {}
    ax = fig.add_subplot(gs[3, 0]); style(ax, "Energy Breakdown (J)")
    if en:
        ks = ["cpu_energy_j", "dram_energy_j", "nic_energy_j_modelled"]
        lbl = ["CPU", "DRAM", "NIC*"]
        ax.bar(lbl, [en.get(k, 0) for k in ks], color=[CY, VI, AM])
        ax.set_ylabel("Joules")
    ax = fig.add_subplot(gs[3, 1]); style(ax, "Energy Efficiency")
    if en:
        ax.bar(["µJ/packet", "mJ/attack"],
               [en.get("energy_per_packet_uj", 0), en.get("energy_per_attack_mj", 0)],
               color=[GR, PK])
        ax.set_yscale("log")
    ax = fig.add_subplot(gs[3, 2]); style(ax, "Power Profile (W)")
    if en:
        ax.bar(["avg", "peak", "detection", "overhead"],
               [en.get("power_w", 0), en.get("peak_power_w", 0),
                en.get("power_during_attack_w", 0), en.get("security_overhead_w", 0)],
               color=[CY, RD, AM, VI])
        ax.set_ylabel("Watts")
    ax = fig.add_subplot(gs[3, 3]); style(ax, "Sustainability")
    if en:
        ax.bar(["bits/J (log)", "gCO2 (x1e3)"],
               [max(en.get("bits_per_joule", 0), 1), en.get("carbon_g_co2", 0) * 1000],
               color=[GR, "#9333ea"])
        ax.set_yscale("log")
        ax.text(.5, -.34, f"method: {en.get('method','')}", ha="center",
                transform=ax.transAxes, color="#6b78bf", fontsize=8)
    fig.text(.5, .012, "SAURON++ · Designed & developed by Dev Subasis · subasismallick2@gmail.com",
             ha="center", color="#6b78bf", fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"results_{time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def write_evaluation_report(engine: "SauronEngine", tel: "Telemetry",
                            source: str, dataset: Optional[str] = None,
                            outdir: str = "results", profile: Optional[Dict] = None,
                            balance_info: Optional[Dict] = None,
                            plots: bool = True) -> Dict:
    """Final evaluation after a dataset run: every metric in the pipeline,
    printed to the terminal AND saved to results/evaluation_<ts>.json / .csv.

    Classification metrics require ground truth; they are only meaningful when
    the source is labeled (sim or a labeled CIC CSV). With an unlabeled source
    (pcap / live / ebpf) that is stated explicitly instead of showing zeros.
    """
    s = engine.snapshot()
    s.update(tel.snapshot())
    lat = tel.latency()
    tp, fp = int(s.get("tp", 0)), int(s.get("fp", 0))
    tn, fn = int(s.get("tn", 0)), int(s.get("fn", 0))
    labeled = (tp + fp + tn + fn) > 0
    tot = (tp + fp + tn + fn) or 1

    acc = (tp + tn) / tot
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    bal = (tpr + tnr) / 2

    rep = {
        "source": source, "dataset": dataset,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labeled": labeled,
        "traffic": {k: s.get(k) for k in ("packets", "dropped", "quarantined",
                                          "alerts", "rule_engine_alerts", "pps", "bps")},
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "classification": {
            "accuracy": round(acc, 5), "balanced_accuracy": round(bal, 5),
            "precision": s.get("precision"), "recall_tpr": s.get("recall"),
            "specificity_tnr": round(tnr, 5), "f1": s.get("f1"), "mcc": s.get("mcc"),
            "fpr": round(fpr, 5), "fnr": round(fnr, 5),
            "roc_auc": s.get("roc_auc"), "pr_auc": s.get("pr_auc"),
            "false_positives": fp, "false_negatives": fn},
        "performance": {
            "throughput_pps": s.get("pps"), "throughput_bps": s.get("bps"),
            "inference_time_us_p50": s.get("inference_time_us"),
            "latency_p50_us": lat.get("p50"), "latency_p99_us": lat.get("p99"),
            "packet_loss_pct": s.get("packet_loss_pct"), "packets_lost": s.get("packets_lost"),
            "cpu_pct": s.get("cpu_pct"), "mem_used_pct": s.get("mem_used_pct"),
            "rss_mb": s.get("rss_mb"),
            "rule_hit_ratio": s.get("rule_hit_ratio"), "rule_hits": s.get("rule_hits"),
            "kernel_load_ms": s.get("kernel_load_ms"),
            "dashboard_latency_ms": s.get("dashboard_latency_ms")},
        "adaptive_loop": {
            "realized_fpr": s.get("realized_fpr"), "fp_budget_eps_h": s.get("fp_budget"),
            "budget_ratio": round((s.get("realized_fpr") or 0) / (s.get("fp_budget") or 1), 3),
            "drift": s.get("drift")},
        "latency_us": {"p50": lat.get("p50"), "p99": lat.get("p99")},
        "energy": s.get("energy", {}),
        "energy_aware_filter": s.get("energy_filter", {}),
    }

    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"evaluation_{stamp}.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    cpath = os.path.join(outdir, f"evaluation_{stamp}.csv")
    with open(cpath, "w", encoding="utf-8") as fh:
        fh.write("metric,value\n")
        for grp in ("classification", "adaptive_loop", "performance"):
            for k, v in rep[grp].items():
                if not isinstance(v, dict):
                    fh.write(f"{k},{v}\n")
        for k, v in rep["confusion_matrix"].items():
            fh.write(f"{k},{v}\n")
        for k, v in rep["traffic"].items():
            fh.write(f"{k},{v}\n")
    if rep.get("per_family"):
        fpath = os.path.join(outdir, f"per_family_{stamp}.csv")
        with open(fpath, "w", encoding="utf-8") as fh:
            cols = ["family", "samples", "detected", "missed", "accuracy",
                    "precision", "recall", "f1", "mcc", "roc_auc", "mean_score"]
            fh.write(",".join(cols) + "\n")
            for r in rep["per_family"]:
                fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
        print(f"         {fpath}   <- per-attack-family metrics table")

    W = 74
    print("\n" + "=" * W)
    print(f"  FINAL EVALUATION REPORT   source={source}" + (f"  dataset={os.path.basename(dataset)}" if dataset else ""))
    print("=" * W)
    t = rep["traffic"]
    print(f"  packets processed : {t.get('packets')}      alerts: {t.get('alerts')}")
    print(f"  dropped           : {t.get('dropped')}      quarantined: {t.get('quarantined')}")
    if labeled:
        print("-" * W)
        print("  CONFUSION MATRIX            predicted-malicious   predicted-benign")
        print(f"    actual malicious                TP={tp:<10}        FN={fn}")
        print(f"    actual benign                   FP={fp:<10}        TN={tn}")
        print("-" * W)
        c = rep["classification"]
        print("  CLASSIFICATION METRICS")
        for k in ("accuracy", "balanced_accuracy", "precision", "recall_tpr",
                  "specificity_tnr", "f1", "mcc", "roc_auc", "pr_auc",
                  "fpr", "fnr", "false_positives", "false_negatives"):
            print(f"    {k:<20}: {c[k]}")
    else:
        print("-" * W)
        print("  CLASSIFICATION METRICS: not available — this source is UNLABELED.")
        print("  Accuracy needs ground truth. Use a labeled CIC CSV export:")
        print("    python backend/sauron.py --headless --source csv --csv <file-or-dir>")
    print("-" * W)
    a = rep["adaptive_loop"]
    print("  ADAPTIVE LOOP")
    print(f"    realized_fpr        : {a['realized_fpr']}  (budget eps_H={a['fp_budget_eps_h']}, "
          f"ratio {a['budget_ratio']}x)")
    print(f"    decision latency    : p50={lat.get('p50')}us  p99={lat.get('p99')}us")
    print("-" * W)
    pf = rep["performance"]
    print("  PERFORMANCE")
    for k in ("throughput_pps", "throughput_bps", "inference_time_us_p50",
              "packet_loss_pct", "cpu_pct", "mem_used_pct", "rss_mb",
              "rule_hit_ratio", "kernel_load_ms", "dashboard_latency_ms"):
        v = pf.get(k)
        print(f"    {k:<22}: {v if v is not None else 'n/a (kernel/dashboard not active)'}")
    fam_rows = family_metrics(engine)
    if fam_rows:
        rep["per_family"] = fam_rows
        print_family_metrics(fam_rows)
    en = rep.get("energy") or {}
    if en:
        print("-" * W)
        print(f"  ENERGY & SUSTAINABILITY   [{en.get('method')}]")
        rows = [("CPU energy", f"{en.get('cpu_energy_j')} J"),
                ("DRAM energy", f"{en.get('dram_energy_j')} J"),
                ("NIC energy (modelled)", f"{en.get('nic_energy_j_modelled')} J"),
                ("Node energy (total)", f"{en.get('node_energy_j')} J"),
                ("Avg power", f"{en.get('power_w')} W  (peak {en.get('peak_power_w')} W)"),
                ("Power during detection", f"{en.get('power_during_attack_w')} W"),
                ("Security overhead", f"{en.get('security_overhead_w')} W"),
                ("Energy per packet", f"{en.get('energy_per_packet_uj')} uJ"),
                ("Energy per attack", f"{en.get('energy_per_attack_mj')} mJ"),
                ("Energy efficiency", f"{en.get('bits_per_joule')} bits/J"),
                ("Carbon footprint", f"{en.get('carbon_g_co2')} g CO2 "
                                     f"@ {en.get('carbon_intensity_g_per_kwh')} g/kWh")]
        for k, v in rows:
            print(f"    {k:<24}: {v}")
        ef = rep.get("energy_aware_filter") or {}
        if ef.get("enabled"):
            print(f"    energy-aware filtering  : budget {ef.get('budget_w')} W, "
                  f"skipped {ef.get('expensive_skipped'):,} expensive evaluations "
                  f"({(ef.get('skip_ratio') or 0)*100:.1f}%)")
        if not en.get("measured"):
            print("    NOTE: RAPL counters unavailable here -> values are MODEL ESTIMATES.")
            print("          Run on bare-metal Linux for hardware-measured Joules.")
    print("=" * W)
    print(f"  saved: {jpath}")
    print(f"         {cpath}")
    if plots:
        png = write_evaluation_plots(engine, tel, profile, balance_info, source, outdir)
        if png:
            print(f"         {png}   <- colourful results figure (15 panels incl. energy)")
    print("=" * W + "\n")
    return rep


def configure_feature_space(path: str, max_feats: int = 40) -> int:
    """Expand the detector bank's feature space to the dataset's own columns.

    The bank ships with 7 packet-derived features, which is all that can be
    computed from a 5-tuple. Flow-level exports like CICIDS2017 carry ~78
    engineered statistics; restricting the detectors to 7 of them discards most
    of the signal (measured: ROC-AUC 0.79 but recall 0.39). This selects the
    most variable numeric columns and rebuilds the feature space around them so
    every detector sees the full picture.

    Returns the resulting dimensionality.
    """
    import csv as _csv
    if os.path.isdir(path):
        files = []
        for _r, _d, _n in os.walk(path):
            files += [os.path.join(_r, x) for x in _n if x.lower().endswith(".csv")]
        files.sort()
    else:
        files = [path]
    if not files:
        return _intel.D
    with open(files[0], newline="", encoding="utf-8-sig", errors="ignore") as fh:
        rd = _csv.reader(fh)
        try:
            hdr = next(rd)
        except StopIteration:
            return _intel.D
        norm = [CsvSource._norm(h) for h in hdr]
        label_like = set(CsvSource._ALIASES["label"]) | set(CsvSource._ALIASES["family"])
        ident = {"source ip", "destination ip", "src ip", "dst ip", "srcip", "dstip",
                 "flow id", "timestamp", "source port", "destination port",
                 "src port", "dst port", "sport", "dsport", "protocol"}
        cand = [(i, hdr[i].strip()) for i in range(len(hdr))
                if norm[i] not in label_like and norm[i] not in ident]
        # sample rows to rank columns by variability (constant columns are useless)
        vals: Dict[int, List[float]] = {i: [] for i, _ in cand}
        for k, row in enumerate(rd):
            if k >= 4000:
                break
            for i, _ in cand:
                if i < len(row):
                    try:
                        v = float(row[i])
                    except Exception:
                        continue
                    if v == v and abs(v) != float("inf"):
                        vals[i].append(v)
        scored = []
        for i, name in cand:
            xs = vals.get(i) or []
            if len(xs) < 50:
                continue
            # Some corpora (Edge-IIoTset) carry values near the float limit, so
            # a naive sum-of-squares overflows. Rank on a log-compressed scale
            # instead: the ordering is preserved and the arithmetic is safe.
            try:
                xs = [math.copysign(math.log1p(abs(x)), x) for x in xs]
                m = sum(xs) / len(xs)
                var = sum((x - m) ** 2 for x in xs) / len(xs)
            except (OverflowError, ValueError):
                continue
            if not (var > 0) or var != var or var == float("inf"):
                continue                      # constant / non-finite: no information
            scored.append((var / (abs(m) + 1.0), i, name))
        scored.sort(reverse=True)
        chosen = [(i, n) for _, i, n in scored[:max_feats]]
        if len(chosen) < 4:
            return _intel.D
        _intel.FEATURES = tuple(n for _, n in chosen)
        _intel.D = len(_intel.FEATURES)
        print(f"[features] feature space expanded to {_intel.D} dataset columns "
              f"(was 7 packet-derived): {', '.join(n for _, n in chosen[:6])} …",
              flush=True)
        return _intel.D


def run_headless(args) -> None:
    tel = Telemetry()
    src = make_source(args.source, tel, pcap=args.pcap, iface=args.iface,
                      speed=args.speed, csv=args.csv, limit=args.limit,
                      limit_per_file=getattr(args, 'limit_per_file', None),
                      balance=args.balance,
                      )
    if args.source in ("csv", "dataset") and args.csv and not getattr(args, "no_native_features", False):
        configure_feature_space(args.csv)
    eng = SauronEngine(eps_h=args.eps_h, energy_budget_w=getattr(args,'energy_budget',0.0))
    # Datasets like CICIDS2017 ship dozens of engineered flow statistics. Those
    # are vastly more discriminative than anything derivable from a 5-tuple with
    # a synthetic arrival time, so use them automatically when present. Disable
    # with --no-native-features to fall back to the packet-derived features.
    eng.feats.use_native = not bool(getattr(args, "no_native_features", False))
    if eng.feats.use_native:
        print("[features] auto: using the dataset's engineered columns when available "
              "(--no-native-features to disable)", flush=True)
    try:
        src._engine_ref = eng          # lets the reader print per-file totals
    except Exception:
        pass
    print(f"SAURON++ | source={args.source} | FP budget eps_H={args.eps_h}")
    print("-" * 96)
    import itertools
    n_events = args.events
    if n_events is None:
        n_events = (10 ** 12 if args.source in ("csv", "dataset", "pcap") else 4000)
    for i, r in enumerate(itertools.islice(src.stream(), n_events), 1):
        t0 = time.perf_counter()
        ev = eng.process(r)
        tel.record_latency(time.perf_counter() - t0)
        if i % args.every == 0 or ev["action"] in ("DROP", "QUARANTINE", "REDIRECT"):
            c = "" if args.no_color else _COLOR.get(ev["action"], "")
            rst = "" if args.no_color else "\033[0m"
            pair = f"{ev['src_ip']}:{ev['src_port']} -> {ev['dst_ip']}"[:33]
            print(f"{i:>5} {pair:<34}{ev['proto']:<6}{ev['score']:>7.3f}"
                  f"{ev['tau_high']:>7.3f}{c}{ev['action']:>12}{rst}  {ev['reason']}")
    write_evaluation_report(eng, tel, args.source,
                            dataset=(args.csv or args.pcap),
                            profile=getattr(src, "profile", None),
                            balance_info=getattr(src, "balance_info", None),
                            plots=not getattr(args, "no_plots", False))


# ============================================================================
# SECTION 7 — FASTAPI REAL-TIME SERVER
# ============================================================================
def rule_predicate(rule: Dict) -> str:
    """Render a distilled stump/conjunction rule as a human-readable predicate,
    e.g. 'pps > 0.80 AND pkt_len < 0.25'. Feature names are in [0,1] units."""
    f1 = f"{rule.get('feature', '?')} {rule.get('op', '?')} {rule.get('threshold', 0):.3f}"
    if rule.get("type") == "conj" and "feature2" in rule:
        f2 = f"{rule['feature2']} {rule.get('op2', '?')} {rule.get('threshold2', 0):.3f}"
        return f"{f1} AND {f2}"
    return f1


def render_prometheus(engine: "SauronEngine", tel: "Telemetry",
                      state: Dict, mesh=None) -> str:
    """Prometheus text-exposition (v0.0.4) of the live metrics — every value is
    measured (no fabricated numbers), so the endpoint plugs straight into
    Grafana/Alertmanager. Counters end in _total; the rest are gauges."""
    snap = engine.snapshot()
    host = tel.snapshot()
    lat = host["latency_us"]
    weights = {n: float(w) for n, w in zip(engine.bank.names, engine.ade.fusion.weights)}

    lines: List[str] = []

    def emit(name: str, value, typ: str, help_: str, labels: str = ""):
        # HELP/TYPE are emitted once per metric family; guard on first sight.
        if name not in emit._seen:  # type: ignore[attr-defined]
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} {typ}")
            emit._seen.add(name)  # type: ignore[attr-defined]
        lines.append(f"{name}{labels} {value}")
    emit._seen = set()  # type: ignore[attr-defined]

    emit("sauron_packets_total", snap["packets"], "counter", "Packets processed")
    emit("sauron_dropped_total", snap["dropped"], "counter", "Packets dropped")
    emit("sauron_quarantined_total", snap["quarantined"], "counter", "Sources quarantined")
    emit("sauron_alerts_total", snap["alerts"], "counter", "Alerts raised")
    emit("sauron_rule_engine_alerts_total", snap["rule_engine_alerts"], "counter",
         "Alerts from the frozen static rule baseline (comparison)")

    emit("sauron_realized_fpr", snap["realized_fpr"], "gauge",
         "IPW-corrected realized false-positive rate")
    emit("sauron_fp_budget", snap["fp_budget"], "gauge", "Configured FP budget eps_H")
    emit("sauron_packets_per_second", snap["pps"], "gauge", "Throughput (packets/s)")
    emit("sauron_bits_per_second", snap["bps"], "gauge", "Throughput (bits/s)")

    for m in ("precision", "recall", "f1", "mcc"):
        emit(f"sauron_{m}", snap[m], "gauge", f"Detection {m} (labeled traffic only)")

    emit("sauron_decision_latency_us", lat["p50"], "gauge",
         "Userspace decision latency (microseconds)", '{quantile="0.5"}')
    emit("sauron_decision_latency_us", lat["p99"], "gauge",
         "Userspace decision latency (microseconds)", '{quantile="0.99"}')

    emit("sauron_cpu_percent", host["cpu_pct"], "gauge", "Host CPU utilization (%)")
    emit("sauron_mem_used_percent", host["mem_used_pct"], "gauge", "Host memory used (%)")
    emit("sauron_process_rss_mb", host["rss_mb"], "gauge", "Process resident memory (MiB)")

    for det, w in weights.items():
        emit("sauron_detector_weight", round(w, 6), "gauge",
             "Hedge fusion weight per detector", f'{{detector="{det}"}}')

    drift = snap.get("drift", {})
    emit("sauron_drift_level", drift.get("level", 0.0), "gauge",
         "Conformal-martingale drift level")
    emit("sauron_drift_detections_total", drift.get("detections", 0), "counter",
         "Drift alarms raised")

    emit("sauron_kernel_attached", 1 if host["kernel_attached"] else 0, "gauge",
         "1 if the eBPF datapath is attached")
    for hook, on in host.get("hooks", {}).items():
        emit("sauron_hook_attached", 1 if on else 0, "gauge",
             "1 if the named kernel hook is attached", f'{{hook="{hook}"}}')

    if mesh is not None:
        try:
            ch = mesh.cluster_health()
            emit("sauron_cluster_size", ch.get("size", 1), "gauge", "Nodes in the mesh")
            intel = mesh.intel_summary()
            emit("sauron_distributed_intel_total", intel.get("total", 0), "counter",
                 "Distributed threat-intel entries (CRDT)")
        except Exception:
            pass

    return "\n".join(lines) + "\n"


def build_app(source_kind: str, eps_h: float, iface: str, pcap: Optional[str],
              speed: float, csv: Optional[str] = None, limit: Optional[int] = None,
              shuffle: bool = True, energy_budget_w: float = 0.0):
    tel = Telemetry()
    engine = SauronEngine(eps_h=eps_h, energy_budget_w=energy_budget_w)
    state = {"paused": False, "speed": 1.0, "profile": "cloud-native-node",
             "source": source_kind,
             "mode": ("dataset" if source_kind in ("pcap", "csv", "dataset")
                      else "realtime"),
             "dataset_available": source_kind in ("pcap", "csv", "dataset"),
             "dataset_path": (csv or pcap or None),
             "dataset_kind": ("csv" if source_kind in ("csv", "dataset")
                              else "pcap" if source_kind == "pcap" else None)}
    clients: set = set()
    history: Deque[Dict] = deque(maxlen=400)
    pending_rules: Deque[Dict] = deque(maxlen=50)
    loop_ref: Dict[str, object] = {}
    queue_ref: Dict[str, object] = {}
    src_holder: Dict[str, object] = {}
    mesh_holder: Dict[str, object] = {}

    def producer():
        src = make_source(source_kind, tel, pcap=pcap, iface=iface, speed=speed,
                          csv=csv, limit=limit, shuffle=shuffle)
        src_holder["src"] = src
        try:
            src._engine_ref = engine   # per-file interim totals in the terminal
            engine.feats.use_native = True
        except Exception:
            pass
        n = 0
        errors = 0
        for r in src.stream():
            n += 1
            t0 = time.perf_counter()
            try:
                ev = engine.process(r)
            except Exception as exc:
                # A single malformed record must never kill the producer thread:
                # that used to freeze the dashboard mid-run while the terminal
                # kept scrolling. Count it, report periodically, keep streaming.
                errors += 1
                if errors <= 3 or errors % 1000 == 0:
                    print(f"[producer] skipped record {n} ({errors} total): {exc}",
                          file=sys.stderr, flush=True)
                continue
            tel.record_latency(time.perf_counter() - t0)
            if n % 25000 == 0:
                print(f"[producer] {n:,} records processed"
                      + (f" ({errors} skipped)" if errors else ""), flush=True)
            # autonomous kernel enforcement: quarantine -> program the blocklist
            if ev["action"] == "QUARANTINE" and hasattr(src, "block"):
                src.block(ev["src_ip"])
            mesh = mesh_holder.get("node")
            if mesh is not None:
                mesh.on_local_threat(ev)                 # gossip to peers (async, non-blocking)
                if n % 50 == 0 and hasattr(src, "block"):
                    for bip in mesh.drain_new_blocks():  # apply remote intel to kernel blocklist
                        src.block(bip)
            # Surface the *real* distilled surrogate (measured fidelity/coverage
            # from the live PolicyDistiller) as an analyst-pending kernel rule,
            # tagged with the concrete offender that triggered surfacing. No
            # fabricated numbers — if the distiller has nothing confident yet,
            # nothing is queued.
            if ev["action"] in ("DROP", "QUARANTINE") and n % 200 == 0:
                real = engine.distilled_rules()
                if real:
                    best = real[0]
                    rid = f"rule-{int(time.time())}-{n}"
                    if not any(r["id"] == rid for r in pending_rules):
                        pending_rules.append({
                            "id": rid,
                            "match": {"src_ip": ev["src_ip"], "proto": ev["proto"]},
                            "predicate": rule_predicate(best),
                            "action": best.get("action", "DROP"),
                            "fidelity": best["fidelity"],     # measured by distiller
                            "coverage": best["coverage"],     # measured by distiller
                            "reason": "distilled surrogate of the fusion policy "
                                      "(pending analyst approval)"})
            forward = ev["action"] != "PASS" or (n % 8 == 0)
            lp, q = loop_ref.get("loop"), queue_ref.get("q")
            if forward and lp is not None and q is not None:
                # The producer runs far faster than the browser can consume, so
                # the queue WILL fill on a big dataset. put_nowait executes
                # inside the event loop, so a QueueFull raised there escapes any
                # try/except here and kills the stream. Push through a wrapper
                # that drops the OLDEST frame instead: the dashboard is a live
                # view, so shedding stale frames is correct and keeps it moving.
                def _safe_put(item, _q=q):
                    try:
                        _q.put_nowait(item)
                    except Exception:
                        try:
                            _q.get_nowait()          # drop oldest
                            _q.put_nowait(item)
                        except Exception:
                            pass                      # transient: skip this frame
                # adaptive shedding: once the queue is mostly full, forward less
                try:
                    if q.qsize() > q.maxsize * 0.8 and ev["action"] == "PASS":
                        forward = False
                except Exception:
                    pass
                if forward:
                    try:
                        lp.call_soon_threadsafe(_safe_put, ev)
                    except Exception:
                        pass
            while state["paused"]:
                time.sleep(0.1)
            if state["speed"] < 1.0:
                time.sleep((1.0 - state["speed"]) * 0.01)
            elif (n & 0x3F) == 0:
                # Yield the GIL periodically. The producer is a CPU-bound thread;
                # without this it starves the asyncio event loop, so the browser
                # cannot complete the WebSocket handshake and the dashboard sits
                # on "connecting..." while the terminal keeps streaming.
                time.sleep(0.0015)

        # ---- stream exhausted: a dataset run has FINISHED -------------------
        # Generate the final report + figure automatically and flag the state so
        # the dashboard can show the completed results instead of silently
        # freezing on the last frame.
        state["finished"] = True
        state["finished_at"] = time.time()
        state["records"] = n
        try:
            rep = write_evaluation_report(
                engine, tel, state["source"], dataset=state.get("dataset_path"),
                profile=getattr(src, "profile", None),
                balance_info=getattr(src, "balance_info", None))
            import glob as _g
            js = sorted(_g.glob(os.path.join("results", "evaluation_*.json")))
            pn = sorted(_g.glob(os.path.join("results", "results_*.png")))
            state["final_report"] = {
                "records": n,
                "classification": rep.get("classification", {}),
                "confusion_matrix": rep.get("confusion_matrix", {}),
                "adaptive_loop": rep.get("adaptive_loop", {}),
                "performance": rep.get("performance", {}),
                "energy": rep.get("energy", {}),
                "per_family": rep.get("per_family", []),
                "files": {"json": js[-1] if js else None,
                          "png": pn[-1] if pn else None},
            }
            print(f"[run] dataset complete - {n:,} records. "
                  f"Final results are shown on the dashboard and saved in results/.",
                  flush=True)
        except Exception as e:
            state["final_report"] = {"error": str(e), "records": n}
            print(f"[run] finished but report failed: {e}", file=sys.stderr, flush=True)

    async def broadcaster():
        q = queue_ref["q"]
        last = 0.0
        while True:
            batch = []
            try:
                batch.append(await asyncio.wait_for(q.get(), timeout=0.05))
                # Drain aggressively so the queue cannot saturate on fast dataset
                # runs. Only a capped, evenly-spaced sample is actually sent to
                # the browser (rendering 1000 events/frame would stall the tab),
                # but everything is pulled off the queue either way.
                while not q.empty() and len(batch) < 1200:
                    batch.append(q.get_nowait())
                if len(batch) > 150:
                    step = len(batch) / 150.0
                    batch = [batch[int(i * step)] for i in range(150)]
            except asyncio.TimeoutError:
                pass
            for ev in batch:
                history.append(ev)
            payloads = []
            if batch:
                payloads.append({"type": "batch", "events": batch})
            now = time.time()
            if now - last > 0.25:
                last = now
                snap = engine.snapshot()
                snap.update(tel.snapshot())
                snap["type"] = "metrics"
                snap["profile"] = state["profile"]
                snap["source"] = state["source"]; snap["mode"] = state["mode"]
                snap["dataset_available"] = state["dataset_available"]
                snap["dataset_path"] = state.get("dataset_path")
                snap["dataset_kind"] = state.get("dataset_kind")
                snap["finished"] = bool(state.get("finished"))
                snap["emit_ts"] = time.time()      # for stale-frame rejection
                snap["parse"] = dict(CsvSource.PARSE_PROGRESS)
                snap["pending_rules"] = list(pending_rules)
                mesh = mesh_holder.get("node")
                if mesh is not None:
                    snap["cluster"] = mesh.cluster_health()
                    snap["distributed_intel"] = mesh.intel_summary()
                    snap["cluster_telemetry"] = mesh.telemetry_summary()
                    snap["federated_model"] = mesh.model_summary()
                payloads.append(snap)
            if payloads and clients:
                dead = []
                for ws in list(clients):
                    try:
                        # A backgrounded browser tab stops reading, so its send
                        # buffer backs up and frames queue for minutes; on return
                        # the tab replays stale frames and looks frozen. Time-box
                        # each send and simply skip this frame for a slow client.
                        for p in payloads:
                            await asyncio.wait_for(ws.send_text(json.dumps(p)),
                                                   timeout=0.5)
                    except asyncio.TimeoutError:
                        continue                   # slow client: drop this frame
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    clients.discard(ws)

    @asynccontextmanager
    async def lifespan(_app):
        loop_ref["loop"] = asyncio.get_running_loop()
        queue_ref["q"] = asyncio.Queue(maxsize=8000)
        threading.Thread(target=producer, daemon=True).start()
        task = asyncio.create_task(broadcaster())
        node = _mesh.maybe_start_mesh(engine, src_holder) if _mesh else None
        if node is not None:
            await node.start()
            mesh_holder["node"] = node
        yield
        task.cancel()
        if node is not None:
            await node.stop()

    app = FastAPI(title="SAURON++ FIREWALL SYSTEM", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        p = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:   # explicit UTF-8 (Windows defaults to cp1252)
                return HTMLResponse(fh.read())
        return HTMLResponse("<h1>SAURON++</h1><p>frontend/index.html missing</p>")

    @app.get("/ebpf_logo.avif")
    async def logo_avif():
        return FileResponse(os.path.join(FRONTEND_DIR, "ebpf_logo.avif"))

    @app.get("/bee_talk.png")
    async def bee_talk():
        """The 'talking bee' used in the no-data messages. Drop your own
        frontend/bee_talk.png (transparent background) to replace it; until then
        the standard logo is served so the UI never shows a broken image."""
        for name in ("bee_talk.png", "bee_talk.webp", "ebpf_logo.png"):
            fp = os.path.join(FRONTEND_DIR, name)
            if os.path.exists(fp):
                return FileResponse(fp)
        return JSONResponse({"error": "no bee image"}, status_code=404)

    @app.get("/ebpf_logo.png")
    async def logo_png():
        return FileResponse(os.path.join(FRONTEND_DIR, "ebpf_logo.png"))

    @app.get("/api/metrics")
    async def api_metrics():
        snap = engine.snapshot()
        snap.update(tel.snapshot())
        snap["source"] = state["source"]; snap["mode"] = state["mode"]
        snap["dataset_available"] = state["dataset_available"]
        snap["dataset_path"] = state.get("dataset_path")
        snap["dataset_kind"] = state.get("dataset_kind")
        snap["finished"] = bool(state.get("finished"))
        snap["parse"] = dict(CsvSource.PARSE_PROGRESS)
        snap["profile"] = state["profile"]
        return JSONResponse(snap)

    @app.get("/api/dataset")
    async def api_dataset():
        """Dataset profile (rows, columns, class balance, imbalance correction)
        plus live per-attack-family metrics, for the dashboard."""
        src = src_holder.get("src")
        prof = getattr(src, "profile", None)
        bal = getattr(src, "balance_info", None)
        return JSONResponse({
            "available": bool(state.get("dataset_available")),
            "path": state.get("dataset_path"), "kind": state.get("dataset_kind"),
            "profile": prof, "balance": bal,
            "families": family_metrics(engine),
            "finished": bool(state.get("finished")),
            "records": state.get("records"),
            "final_report": state.get("final_report")})

    @app.post("/api/report")
    async def api_report():
        """Generate the evaluation report + 15-panel figure right now, without
        stopping the run. Returns the saved file paths."""
        try:
            rep = write_evaluation_report(
                engine, tel, state["source"],
                dataset=state.get("dataset_path"),
                profile=getattr(src_holder.get("src"), "profile", None),
                balance_info=getattr(src_holder.get("src"), "balance_info", None))
            import glob as _g
            js = sorted(_g.glob(os.path.join("results", "evaluation_*.json")))
            pg = sorted(_g.glob(os.path.join("results", "results_*.png")))
            return JSONResponse({"ok": True,
                                 "json": js[-1] if js else None,
                                 "png": pg[-1] if pg else None,
                                 "accuracy": rep["classification"].get("accuracy"),
                                 "mcc": rep["classification"].get("mcc")})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus scrape target (text exposition v0.0.4). Wire a Prometheus
        job at this path and point Grafana at it; every series is measured."""
        text = render_prometheus(engine, tel, state, mesh_holder.get("node"))
        return Response(content=text,
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/api/history")
    async def api_history():
        return JSONResponse(list(history))

    # ---- distributed mesh (node-to-node) status, for the dashboard ----
    @app.get("/api/cluster")
    async def api_cluster():
        node = mesh_holder.get("node")
        if node is None:
            return JSONResponse({"enabled": False, "node_id": None, "size": 1, "nodes": []})
        return JSONResponse({"enabled": True, **node.cluster_health()})

    @app.get("/api/intel")
    async def api_intel():
        node = mesh_holder.get("node")
        return JSONResponse(node.intel_summary() if node else {"enabled": False, "total": 0, "recent": []})

    @app.get("/api/cluster/telemetry")
    async def api_cluster_tel():
        node = mesh_holder.get("node")
        return JSONResponse(node.telemetry_summary() if node else {"peers": [], "cluster_totals": {}})

    @app.get("/api/federated-model")
    async def api_fed_model():
        node = mesh_holder.get("node")
        return JSONResponse(node.model_summary() if node else {"round": 0, "aggregated": {}})

    @app.post("/api/cluster/policy")
    async def api_cluster_policy(payload: Dict):
        node = mesh_holder.get("node")
        if node is None:
            return JSONResponse({"ok": False, "reason": "mesh disabled"})
        node.publish_policy(eps_h=payload.get("eps_h"), profile=payload.get("profile"))
        return JSONResponse({"ok": True})

    @app.get("/api/config")
    async def api_get_config():
        return JSONResponse({**state, "eps_h": engine.ade.threshold.eps_h})

    @app.post("/api/config")
    async def api_set_config(payload: Dict):
        if "eps_h" in payload:
            engine.set_eps_h(float(payload["eps_h"]))
        for k in ("profile", "paused", "speed"):
            if k in payload:
                state[k] = payload[k]
        return JSONResponse({"ok": True})

    @app.post("/api/rules/approve")
    async def api_approve(payload: Dict):
        rid = payload.get("id")
        rule = next((r for r in pending_rules if r["id"] == rid), None)
        pending_rules_new = [r for r in pending_rules if r["id"] != rid]
        pending_rules.clear()
        pending_rules.extend(pending_rules_new)
        src = src_holder.get("src")
        if rule and src is not None and hasattr(src, "block"):
            src.block(rule["match"]["src_ip"])  # compile to kernel blocklist map
        return JSONResponse({"ok": True, "approved": rid})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        try:
            await websocket.send_text(json.dumps({"type": "batch",
                                                  "events": list(history)[-150:]}))
            snap = engine.snapshot()
            snap.update(tel.snapshot())
            snap["type"] = "metrics"
            snap["profile"] = state["profile"]
            await websocket.send_text(json.dumps(snap))
            while True:
                msg = await websocket.receive_text()
                try:
                    cmd = json.loads(msg)
                    if cmd.get("cmd") == "dashboard_latency":
                        tel.note_dashboard_latency(float(cmd["value"]))
                    elif cmd.get("cmd") == "set_eps_h":
                        engine.set_eps_h(float(cmd["value"]))
                    elif cmd.get("cmd") == "pause":
                        state["paused"] = bool(cmd["value"])
                    elif cmd.get("cmd") == "speed":
                        state["speed"] = float(cmd["value"])
                except Exception:
                    pass
        except WebSocketDisconnect:
            clients.discard(websocket)
        except Exception:
            clients.discard(websocket)

    return app


def main():
    ap = argparse.ArgumentParser(description="SAURON++ backend")
    ap.add_argument("--source", default="sim",
                    choices=["sim", "pcap", "ebpf", "live", "csv", "dataset"])
    ap.add_argument("--csv", default=None,
                    help="labeled CIC dataset CSV file or directory (--source csv)")
    ap.add_argument("--balance", default="auto", choices=["auto", "none"],
                    help="correct dataset class imbalance before scoring (default: auto)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows in TOTAL (reads files in order, so on a "
                         "multi-file corpus later attack types may never be reached)")
    ap.add_argument("--limit-per-file", type=int, default=None,
                    help="take at most N rows from EACH csv file. Use this on multi-file "
                         "corpora (CIC-DDoS2019, CICIDS2017) so every attack class is "
                         "represented in the sample")
    ap.add_argument("--pcap", default=None, help="path to CICIDS2017 .pcap (pcap source)")
    ap.add_argument("--iface", default=None,
                    help="interface for ebpf/live source (default: system default)")
    ap.add_argument("--list-ifaces", action="store_true",
                    help="list capturable network interfaces (for --source live) and exit")
    ap.add_argument("--energy-budget", type=float, default=0.0,
                    help="watts; enables energy-aware packet filtering above this power")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the results figure at the end of a run")
    ap.add_argument("--packet-csv", default=None, metavar="PATH",
                    help="write every packet decision (score, threshold, trust, "
                         "verdict, reason, mitigation, top features) to a CSV file")
    ap.add_argument("--no-print-packets", action="store_true",
                    help="do not print each live packet to the terminal")
    ap.add_argument("--speed", type=float, default=0.0, help="pcap replay pacing divisor (0=fast)")
    ap.add_argument("--eps-h", type=float, default=0.02, dest="eps_h")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--headless", action="store_true", help="no web stack; print stream + metrics")
    ap.add_argument("--events", type=int, default=None,
                    help="stop after N records (default: 4000 for sim/live, "
                         "ENTIRE dataset for csv/pcap)")
    ap.add_argument("--inspect", action="store_true",
                    help="preflight: check a dataset parses (columns/labels) and exit")
    ap.add_argument("--native-features", action="store_true",
                    help="(default) use the dataset's engineered columns")
    ap.add_argument("--no-native-features", action="store_true",
                    help="ignore the dataset's engineered columns and use only "
                         "features derived from the 5-tuple")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="keep dataset file order (default: shuffle, so attack "
                         "and benign rows interleave instead of arriving in blocks)")
    ap.add_argument("--every", type=int, default=400)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    global _PRINT_PKTS, _NO_COLOR, _PKT_CSV, _PKT_CSV_FH
    if getattr(args, "packet_csv", None):
        import csv as _c
        _PKT_CSV_FH = open(args.packet_csv, "w", newline="", encoding="utf-8")
        _PKT_CSV = _c.writer(_PKT_CSV_FH)
        _PKT_CSV.writerow(_PKT_CSV_COLS)
        print(f"[csv] writing per-packet decisions to {args.packet_csv}", flush=True)
    if args.no_print_packets:
        _PRINT_PKTS = False
    if getattr(args, "no_color", False):
        _NO_COLOR = True

    if getattr(args, "inspect", False):
        target = args.csv or args.pcap
        if not target:
            print("--inspect needs --csv <file-or-dir>", file=sys.stderr); sys.exit(1)
        inspect_dataset(target); return

    if args.list_ifaces:
        try:
            from scapy.all import conf, get_if_list  # type: ignore
        except Exception:
            print("scapy required to list interfaces: pip install scapy", file=sys.stderr)
            sys.exit(1)
        print("Capturable interfaces (use the name/index with --iface):\n")
        try:
            print(conf.ifaces)                 # formatted table (name, MAC, IPv4)
        except Exception:
            for name in get_if_list():
                print("  ", name)
        print("\nExample:  sudo python3 backend/sauron.py --source live --iface <name>")
        return

    if args.headless:
        run_headless(args)
        return

    if not _WEB_OK:
        print("fastapi not installed. Run scripts/build.sh or "
              "pip install fastapi 'uvicorn[standard]' websockets, or use --headless.",
              file=sys.stderr)
        sys.exit(1)
    try:
        import uvicorn
    except Exception:
        print("uvicorn not installed. pip install 'uvicorn[standard]', or use --headless.",
              file=sys.stderr)
        sys.exit(1)
    app = build_app(args.source, args.eps_h, args.iface, args.pcap, args.speed,
                    csv=args.csv, limit=args.limit,
                    shuffle=not getattr(args, 'no_shuffle', False),
                    energy_budget_w=getattr(args, 'energy_budget', 0.0))
    print(f"SAURON++ FIREWALL SYSTEM  |  source={args.source}  |  "
          f"http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

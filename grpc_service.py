"""
SAURON++ distributed mesh  (node <-> node only, over gRPC/asyncio).

This module adds distributed threat intelligence, adaptive policy sync, telemetry
sharing, federated model sync, heartbeat monitoring, node discovery and cluster
health to SAURON++ WITHOUT touching the packet path. It is entirely optional:
the backend imports it lazily and only starts a mesh node when SAURON_MESH_ENABLE
is set. WebSocket stays the backend<->dashboard transport; ring buffer + BPF maps
stay the kernel<->user transport. gRPC is used exclusively between nodes.

Design highlights (deliberately beyond a plain gRPC bus):
  * SWIM-style membership with incarnation numbers + suspicion, driven by a
    phi-accrual failure detector (Hayashibara et al.) rather than fixed timeouts.
  * Epidemic anti-entropy gossip that disseminates both membership and threat
    intelligence in O(log N) rounds.
  * Distributed threat intelligence as a conflict-free LWW-element-set CRDT with
    TTL expiry -> eventually consistent, merge-safe, no coordinator.
  * Causal adaptive-policy synchronisation via vector clocks (last-writer-wins is
    unsafe for policy; we respect causality and break concurrency deterministically).
  * Byzantine-robust federated model sync: coordinate-wise trimmed mean + Krum
    over peer fusion weights, so a minority of malicious/miscalibrated nodes cannot
    poison the cluster model (application back into the ADE is opt-in and off by
    default to preserve single-node stability).
  * Per-peer circuit breaker + decorrelated-jitter backoff + deadlines; bounded
    queues for backpressure; optional mTLS. Every remote failure is contained.

The only backend touch-points are a handful of calls documented in
`maybe_start_mesh` and the MeshNode public API (on_local_threat, drain_new_blocks,
cluster_health, intel_summary, telemetry_summary, model_summary, publish_policy).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 0.  Stub bootstrap  — generate sauron_pb2 / sauron_pb2_grpc from the .proto  #
#     on first import, so the only source files are this module + sauron.proto #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _ensure_stubs() -> None:
    try:
        import sauron_pb2  # noqa: F401
        import sauron_pb2_grpc  # noqa: F401
        return
    except ImportError:
        pass
    proto = os.path.join(_HERE, "sauron.proto")
    if not os.path.exists(proto):
        raise RuntimeError("sauron.proto not found next to grpc_service.py")
    from grpc_tools import protoc
    rc = protoc.main([
        "protoc", f"-I{_HERE}",
        f"--python_out={_HERE}", f"--grpc_python_out={_HERE}", proto,
    ])
    if rc != 0:
        raise RuntimeError(f"protoc failed to compile sauron.proto (rc={rc})")


try:
    import grpc  # grpcio
    from grpc import aio as grpc_aio
    _ensure_stubs()
    import sauron_pb2 as pb
    import sauron_pb2_grpc as pbg
    GRPC_OK = True
    _IMPORT_ERR = ""
except Exception as _e:  # grpc not installed / codegen failed -> mesh disabled
    GRPC_OK = False
    _IMPORT_ERR = repr(_e)


# --------------------------------------------------------------------------- #
# 1.  Phi-accrual failure detector                                            #
# --------------------------------------------------------------------------- #
class PhiAccrual:
    """Adaptive failure detector: outputs a suspicion level phi that rises the
    longer a heartbeat is overdue, calibrated to the observed inter-arrival
    distribution. phi > threshold  =>  treat the peer as failed."""

    def __init__(self, window: int = 200, min_std: float = 50.0, first_est_ms: float = 750.0):
        self._iat: Deque[float] = deque(maxlen=window)
        self._last: Optional[float] = None
        self._min_std = min_std
        self._first = first_est_ms

    def heartbeat(self, now_ms: Optional[float] = None) -> None:
        now_ms = now_ms if now_ms is not None else time.time() * 1000.0
        if self._last is not None:
            self._iat.append(now_ms - self._last)
        self._last = now_ms

    def _mean_std(self) -> Tuple[float, float]:
        if not self._iat:
            return self._first, self._first / 2
        n = len(self._iat)
        m = sum(self._iat) / n
        var = sum((x - m) ** 2 for x in self._iat) / n if n > 1 else 0.0
        return m, max(var ** 0.5, self._min_std)

    def phi(self, now_ms: Optional[float] = None) -> float:
        if self._last is None:
            return 0.0
        now_ms = now_ms if now_ms is not None else time.time() * 1000.0
        m, s = self._mean_std()
        diff = now_ms - self._last
        # log10 survival of a normal CDF (logistic approximation of the tail)
        y = (diff - m) / s
        e = 2.718281828459045
        p = e ** (-y * (1.5976 + 0.070566 * y * y)) if y >= 0 else 1 - (
            e ** (y * (1.5976 - 0.070566 * y * y)))
        p = min(max(p, 1e-10), 1.0)
        import math
        return -math.log10(p)


# --------------------------------------------------------------------------- #
# 2.  Circuit breaker + decorrelated-jitter backoff                           #
# --------------------------------------------------------------------------- #
class Breaker:
    def __init__(self, fail_max: int = 4, reset_after: float = 8.0):
        self.fail = 0
        self.fail_max = fail_max
        self.reset_after = reset_after
        self.open_until = 0.0

    def allow(self) -> bool:
        if self.fail < self.fail_max:
            return True
        if time.time() >= self.open_until:      # half-open probe
            return True
        return False

    def ok(self) -> None:
        self.fail = 0
        self.open_until = 0.0

    def bad(self) -> None:
        self.fail += 1
        if self.fail >= self.fail_max:
            self.open_until = time.time() + self.reset_after


def _backoff(prev: float, base: float = 0.2, cap: float = 5.0) -> float:
    return min(cap, random.uniform(base, max(base, prev * 3.0)))


# --------------------------------------------------------------------------- #
# 3.  Vector clock (causal policy ordering)                                    #
# --------------------------------------------------------------------------- #
class VClock:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.v: Dict[str, int] = defaultdict(int)

    def tick(self) -> Dict[str, int]:
        self.v[self.node_id] += 1
        return dict(self.v)

    def merge(self, other: Dict[str, int]) -> None:
        for k, val in other.items():
            if val > self.v[k]:
                self.v[k] = val

    @staticmethod
    def dominates(a: Dict[str, int], b: Dict[str, int]) -> bool:
        """True if a >= b on every component and strictly greater on one."""
        ge = all(a.get(k, 0) >= v for k, v in b.items())
        gt = any(a.get(k, 0) > b.get(k, 0) for k in set(a) | set(b))
        return ge and gt


# --------------------------------------------------------------------------- #
# 4.  Byzantine-robust federated aggregation                                  #
# --------------------------------------------------------------------------- #
def _trimmed_mean(vecs: List[List[float]], beta: float = 0.2) -> List[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    out = []
    k = int(len(vecs) * beta)
    for j in range(dim):
        col = sorted(v[j] for v in vecs if len(v) == dim)
        col = col[k: len(col) - k] if len(col) - 2 * k > 0 else col
        out.append(sum(col) / len(col) if col else 0.0)
    return out


def _krum(vecs: List[List[float]], f: int = 1) -> Optional[List[float]]:
    """Select the vector closest to its n-f-2 nearest neighbours (Blanchard '17)."""
    n = len(vecs)
    if n <= 2 or n - f - 2 <= 0:
        return _trimmed_mean(vecs)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j]))
            dist[i][j] = dist[j][i] = d
    best, best_i = float("inf"), 0
    m = n - f - 2
    for i in range(n):
        nearest = sorted(dist[i])[1: m + 1]
        s = sum(nearest)
        if s < best:
            best, best_i = s, i
    return vecs[best_i]


# --------------------------------------------------------------------------- #
# 5.  The mesh node                                                           #
# --------------------------------------------------------------------------- #
class MeshNode(pbg.SauronMeshServicer if GRPC_OK else object):
    """One SAURON++ node in the distributed mesh."""

    def __init__(self, engine, src_holder: Dict[str, Any], cfg: Dict[str, Any]):
        self.engine = engine
        self.src_holder = src_holder
        self.cfg = cfg
        self.node_id: str = cfg["node_id"]
        self.address: str = cfg["advertise"]
        self.region: str = cfg.get("region", "cloud")
        self.seeds: List[str] = cfg.get("seeds", [])
        self.apply_model: bool = cfg.get("apply_model", False)
        self.confidence_gate: float = cfg.get("confidence_gate", 0.6)

        # membership (SWIM)
        self.incarnation = 0
        self.heartbeat = 0
        self.members: Dict[str, Dict[str, Any]] = {}   # node_id -> member dict
        self.phi: Dict[str, PhiAccrual] = defaultdict(PhiAccrual)
        self.breaker: Dict[str, Breaker] = defaultdict(Breaker)

        # logical clocks
        self.lamport = 0
        self.vclock = VClock(self.node_id)

        # distributed threat intelligence CRDT (LWW-element-set with TTL)
        self.intel: Dict[str, Dict[str, Any]] = {}     # src_ip -> event dict
        self._intel_lock = threading.Lock()

        # cross-thread queues (producer thread <-> event loop)
        self._local_threats: _SimpleQ = _SimpleQ()
        self._new_blocks: _SimpleQ = _SimpleQ()
        self._seen_blocks: set = set()

        # telemetry + model buffers from peers
        self.peer_tel: Dict[str, Dict[str, Any]] = {}
        self.peer_model: Dict[str, Dict[str, Any]] = {}
        self.model_round = 0
        self.aggregated_model: Dict[str, Any] = {}

        # policy
        self.policy: Dict[str, Any] = {"policy_id": "", "eps_h": None, "profile": ""}

        # channels
        self._chan: Dict[str, "grpc_aio.Channel"] = {}
        self._stub: Dict[str, "pbg.SauronMeshStub"] = {}

        self._server = None
        self._tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event() if GRPC_OK else None
        self.stats = {"intel_rx": 0, "intel_tx": 0, "gossip": 0, "hb_rx": 0,
                      "hb_fail": 0, "policy_rx": 0, "model_rx": 0}

    # ---- self descriptor -------------------------------------------------- #
    def _self_info(self) -> "pb.NodeInfo":
        return pb.NodeInfo(node_id=self.node_id, address=self.address,
                           incarnation=self.incarnation, heartbeat=self.heartbeat,
                           state="ALIVE", ts=time.time(), region=self.region)

    def _member_dict(self, ni: "pb.NodeInfo") -> Dict[str, Any]:
        return {"node_id": ni.node_id, "address": ni.address,
                "incarnation": ni.incarnation, "heartbeat": ni.heartbeat,
                "state": ni.state or "ALIVE", "ts": ni.ts or time.time(),
                "region": ni.region or "cloud"}

    def _merge_member(self, m: Dict[str, Any]) -> None:
        if m["node_id"] == self.node_id:
            # refute stale rumours about ourselves
            if m.get("state") in ("SUSPECT", "DEAD") and m["incarnation"] >= self.incarnation:
                self.incarnation = m["incarnation"] + 1
            return
        cur = self.members.get(m["node_id"])
        if cur is None:
            self.members[m["node_id"]] = m
            return
        # higher incarnation wins; within same incarnation, DEAD>SUSPECT>ALIVE and
        # higher heartbeat wins
        if m["incarnation"] > cur["incarnation"]:
            self.members[m["node_id"]] = m
        elif m["incarnation"] == cur["incarnation"]:
            rank = {"ALIVE": 0, "SUSPECT": 1, "LEFT": 2, "DEAD": 3}
            if rank.get(m["state"], 0) > rank.get(cur["state"], 0):
                cur["state"] = m["state"]
            if m["heartbeat"] > cur["heartbeat"]:
                cur["heartbeat"] = m["heartbeat"]
                cur["ts"] = m.get("ts", time.time())

    # ---- CRDT intel merge ------------------------------------------------- #
    def _merge_intel(self, ev: Dict[str, Any]) -> bool:
        """LWW by ts. Returns True if this is a *new/updated* actionable block."""
        ip = ev.get("src_ip")
        if not ip:
            return False
        with self._intel_lock:
            cur = self.intel.get(ip)
            if cur is None or ev["ts"] > cur["ts"]:
                self.intel[ip] = ev
                new = True
            else:
                new = False
        actionable = (ev.get("confidence", 0.0) >= self.confidence_gate and
                      ev.get("action") in ("DROP", "QUARANTINE"))
        if new and actionable and ip not in self._seen_blocks:
            self._seen_blocks.add(ip)
            self._new_blocks.put(ip)
            return True
        return False

    def _intel_to_pb(self, ev: Dict[str, Any]) -> "pb.ThreatEvent":
        return pb.ThreatEvent(
            src_ip=ev.get("src_ip", ""), dst_ip=ev.get("dst_ip", ""),
            proto=ev.get("proto", ""), action=ev.get("action", ""),
            severity=ev.get("severity", ""), score=float(ev.get("score", 0.0)),
            confidence=float(ev.get("confidence", 0.0)), ts=float(ev.get("ts", time.time())),
            origin_node=ev.get("origin_node", self.node_id), reason=ev.get("reason", ""),
            lamport=int(ev.get("lamport", 0)), ttl=float(ev.get("ttl", 900.0)))

    def _pb_to_intel(self, e: "pb.ThreatEvent") -> Dict[str, Any]:
        return {"src_ip": e.src_ip, "dst_ip": e.dst_ip, "proto": e.proto,
                "action": e.action, "severity": e.severity, "score": e.score,
                "confidence": e.confidence, "ts": e.ts, "origin_node": e.origin_node,
                "reason": e.reason, "lamport": e.lamport, "ttl": e.ttl}

    # ===================================================================== #
    #  gRPC servicer side  (inbound RPCs from peers)                        #
    # ===================================================================== #
    async def ShareThreatIntel(self, request, context):
        self.stats["intel_rx"] += 1
        for e in request.events:
            self.lamport = max(self.lamport, e.lamport) + 1
            self._merge_intel(self._pb_to_intel(e))
        return pb.Ack(ok=True, node_id=self.node_id, lamport=self.lamport)

    async def SyncPolicy(self, request, context):
        self.stats["policy_rx"] += 1
        incoming = dict(request.vclock.clock)
        cur = dict(self.vclock.v)
        take = VClock.dominates(incoming, cur) or (
            not VClock.dominates(cur, incoming) and request.origin_node < self.node_id)
        if take:
            self.vclock.merge(incoming)
            self.policy = {"policy_id": request.policy_id, "eps_h": request.eps_h,
                           "profile": request.profile, "origin": request.origin_node}
            if request.eps_h and hasattr(self.engine, "set_eps_h"):
                try:
                    self.engine.set_eps_h(float(request.eps_h))
                except Exception:
                    pass
        return pb.Ack(ok=take, node_id=self.node_id, lamport=self.lamport)

    async def ShareTelemetry(self, request, context):
        self.peer_tel[request.node_id] = {
            "node_id": request.node_id, "pps": request.pps,
            "active_flows": request.active_flows, "dropped": request.dropped,
            "quarantined": request.quarantined, "cpu": request.cpu, "mem": request.mem,
            "latency_ms": request.latency_ms, "fpr": request.fpr, "ts": request.ts,
            "region": request.region, "address": request.address}
        return pb.Ack(ok=True, node_id=self.node_id)

    async def SyncModel(self, request, context):
        self.stats["model_rx"] += 1
        self.peer_model[request.node_id] = {
            "weights": list(request.fusion_weights), "threshold": request.threshold,
            "eps_h": request.eps_h, "samples": request.samples, "ts": request.ts}
        return pb.Ack(ok=True, node_id=self.node_id)

    async def Heartbeat(self, request, context):
        self.stats["hb_rx"] += 1
        self._merge_member(self._member_dict(request.from_field)) if False else None
        fi = getattr(request, "from")  # 'from' is reserved in python
        self._merge_member(self._member_dict(fi))
        self.phi[fi.node_id].heartbeat()
        for ni in request.digest:
            self._merge_member(self._member_dict(ni))
        digest = [self._self_info()] + [self._info_from_member(m)
                                        for m in list(self.members.values())[:16]]
        return pb.Pong(**{"from": self._self_info()}, digest=digest, ts=time.time(), ack=True)

    async def Gossip(self, request, context):
        self.stats["gossip"] += 1
        for ni in request.members:
            self._merge_member(self._member_dict(ni))
        for e in request.intel:
            self._merge_intel(self._pb_to_intel(e))
        return pb.GossipDigest(from_node=self.node_id,
                               members=[self._self_info()] + [self._info_from_member(m)
                                                              for m in list(self.members.values())[:24]],
                               intel=[self._intel_to_pb(e) for e in self._recent_intel(40)])

    async def StreamIntel(self, request_iter, context):
        async for e in request_iter:
            self.lamport = max(self.lamport, e.lamport) + 1
            self._merge_intel(self._pb_to_intel(e))
            yield self._intel_to_pb({"src_ip": self.node_id, "action": "ACK",
                                     "ts": time.time(), "confidence": 0.0})

    def _info_from_member(self, m: Dict[str, Any]) -> "pb.NodeInfo":
        return pb.NodeInfo(node_id=m["node_id"], address=m["address"],
                           incarnation=m["incarnation"], heartbeat=m["heartbeat"],
                           state=m["state"], ts=m.get("ts", time.time()),
                           region=m.get("region", "cloud"))

    def _recent_intel(self, n: int) -> List[Dict[str, Any]]:
        with self._intel_lock:
            evs = sorted(self.intel.values(), key=lambda e: e["ts"], reverse=True)
        return evs[:n]

    # ===================================================================== #
    #  client side helpers                                                  #
    # ===================================================================== #
    def _credentials(self):
        cert, key, ca = self.cfg.get("tls_cert"), self.cfg.get("tls_key"), self.cfg.get("tls_ca")
        if cert and key and ca:
            return ("mtls",
                    grpc.ssl_server_credentials([(open(key, "rb").read(), open(cert, "rb").read())],
                                                root_certificates=open(ca, "rb").read(),
                                                require_client_auth=True),
                    grpc.ssl_channel_credentials(root_certificates=open(ca, "rb").read(),
                                                 private_key=open(key, "rb").read(),
                                                 certificate_chain=open(cert, "rb").read()))
        return ("insecure", None, None)

    async def _get_stub(self, addr: str):
        if addr not in self._stub:
            mode, _, chcreds = self._credentials()
            opts = [("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.max_receive_message_length", 16 * 1024 * 1024)]
            if mode == "mtls":
                ch = grpc_aio.secure_channel(addr, chcreds, options=opts)
            else:
                ch = grpc_aio.insecure_channel(addr, options=opts)
            self._chan[addr] = ch
            self._stub[addr] = pbg.SauronMeshStub(ch)
        return self._stub[addr]

    async def _call(self, addr: str, method: str, req, timeout: float = 3.0):
        br = self.breaker[addr]
        if not br.allow():
            return None
        try:
            stub = await self._get_stub(addr)
            resp = await getattr(stub, method)(req, timeout=timeout)
            br.ok()
            return resp
        except Exception:
            br.bad()
            return None

    def _peer_addrs(self) -> List[str]:
        return [m["address"] for m in self.members.values()
                if m["node_id"] != self.node_id and m["state"] in ("ALIVE", "SUSPECT")]

    # ===================================================================== #
    #  public API used by the backend (thread-safe)                         #
    # ===================================================================== #
    def on_local_threat(self, ev: Dict[str, Any]) -> None:
        """Called from the producer thread when a local enforcement decision is
        made. Non-blocking: enqueues for the gossip loop to disseminate."""
        act = ev.get("action")
        if act not in ("DROP", "QUARANTINE", "REDIRECT", "RATE_LIMIT"):
            return
        self.lamport += 1
        conf = float(ev.get("aggregate", ev.get("raw_aggregate", ev.get("score", 0.8))) or 0.8)
        rec = {"src_ip": ev.get("src_ip", ""), "dst_ip": ev.get("dst_ip", ""),
               "proto": ev.get("proto", ""), "action": act,
               "severity": ev.get("severity", ""), "score": float(ev.get("aggregate", 0.0) or 0.0),
               "confidence": min(1.0, conf), "ts": time.time(), "origin_node": self.node_id,
               "reason": ev.get("reason", ""), "lamport": self.lamport, "ttl": 900.0}
        self._merge_intel(rec)                 # add to our own CRDT
        self._local_threats.put(rec)           # queue for dissemination

    def drain_new_blocks(self, limit: int = 64) -> List[str]:
        """Return src IPs newly learned from *remote* intel that should be
        programmed into the kernel blocklist. Called from the producer thread."""
        return self._new_blocks.drain(limit)

    def publish_policy(self, eps_h: Optional[float] = None, profile: Optional[str] = None) -> None:
        vc = self.vclock.tick()
        self.policy = {"policy_id": f"pol-{int(time.time())}",
                       "eps_h": eps_h, "profile": profile, "origin": self.node_id}
        upd = pb.PolicyUpdate(origin_node=self.node_id, policy_id=self.policy["policy_id"],
                              eps_h=float(eps_h or 0.0), profile=profile or "",
                              payload=json.dumps(self.policy).encode(),
                              vclock=pb.VectorClock(clock=vc), ts=time.time())
        for addr in self._peer_addrs():
            asyncio.create_task(self._call(addr, "SyncPolicy", upd))

    # ---- dashboard-facing summaries (served over the existing WS/REST) ---- #
    def cluster_health(self) -> Dict[str, Any]:
        now = time.time() * 1000.0
        nodes = [{"node_id": self.node_id, "address": self.address, "region": self.region,
                  "state": "ALIVE", "phi": 0.0, "self": True,
                  "heartbeat": self.heartbeat, "incarnation": self.incarnation}]
        for nid, m in self.members.items():
            nodes.append({"node_id": nid, "address": m["address"], "region": m.get("region", "cloud"),
                          "state": m["state"], "phi": round(self.phi[nid].phi(now), 2),
                          "self": False, "heartbeat": m["heartbeat"], "incarnation": m["incarnation"]})
        alive = sum(1 for n in nodes if n["state"] == "ALIVE")
        return {"node_id": self.node_id, "region": self.region, "size": len(nodes),
                "alive": alive, "suspect": sum(1 for n in nodes if n["state"] == "SUSPECT"),
                "dead": sum(1 for n in nodes if n["state"] == "DEAD"),
                "quorum": alive > len(nodes) // 2, "nodes": nodes, "stats": dict(self.stats)}

    def intel_summary(self, n: int = 60) -> Dict[str, Any]:
        evs = self._recent_intel(n)
        by_origin: Dict[str, int] = defaultdict(int)
        for e in evs:
            by_origin[e.get("origin_node", "?")] += 1
        with self._intel_lock:
            total = len(self.intel)
        return {"total": total, "by_origin": dict(by_origin),
                "recent": [{"src_ip": e["src_ip"], "action": e["action"],
                            "severity": e["severity"], "confidence": round(e["confidence"], 3),
                            "origin": e["origin_node"], "age": round(time.time() - e["ts"], 1),
                            "reason": e.get("reason", "")} for e in evs]}

    def telemetry_summary(self) -> Dict[str, Any]:
        rows = list(self.peer_tel.values())
        agg = {"pps": sum(r["pps"] for r in rows), "dropped": sum(r["dropped"] for r in rows),
               "quarantined": sum(r["quarantined"] for r in rows), "nodes": len(rows) + 1}
        return {"peers": rows, "cluster_totals": agg}

    def model_summary(self) -> Dict[str, Any]:
        return {"round": self.model_round, "aggregated": self.aggregated_model,
                "contributors": list(self.peer_model.keys()) + [self.node_id],
                "applied": self.apply_model}

    # ===================================================================== #
    #  local model export (reads ADE state without disturbing it)           #
    # ===================================================================== #
    def _export_model(self) -> Dict[str, Any]:
        w, thr, eps = [], 0.6, 0.02
        try:
            w = [float(x) for x in self.engine.ade.fusion.weights()]
        except Exception:
            pass
        try:
            taus = list(self.engine.ade.threshold._tau.values())
            thr = float(sum(taus) / len(taus)) if taus else float(self.engine.ade.threshold.tau_init)
            eps = float(self.engine.ade.threshold.eps_h)
        except Exception:
            pass
        return {"weights": w, "threshold": thr, "eps_h": eps}

    def _aggregate_model(self) -> None:
        local = self._export_model()
        vecs = [m["weights"] for m in self.peer_model.values() if m["weights"]]
        if local["weights"]:
            vecs.append(local["weights"])
        if len(vecs) < 2:
            self.aggregated_model = {**local, "method": "local-only", "n": len(vecs)}
            return
        f = max(1, len(vecs) // 4)
        krum = _krum(vecs, f=f)
        agg = _trimmed_mean([krum] + vecs, beta=0.2) if krum else _trimmed_mean(vecs, 0.2)
        s = sum(agg) or 1.0
        agg = [x / s for x in agg]
        thr_vals = [m["threshold"] for m in self.peer_model.values()] + [local["threshold"]]
        thr = sorted(thr_vals)[len(thr_vals) // 2]     # median (robust)
        self.aggregated_model = {"weights": [round(x, 4) for x in agg],
                                 "threshold": round(thr, 4), "eps_h": local["eps_h"],
                                 "method": "krum+trimmed-mean", "n": len(vecs)}
        if self.apply_model:                            # opt-in; off by default
            try:
                import numpy as _np
                self.engine.ade.fusion.w = _np.array(agg, dtype=float)
            except Exception:
                pass

    # ===================================================================== #
    #  background loops                                                     #
    # ===================================================================== #
    async def _loop_heartbeat(self):
        while not self._stop.is_set():
            self.heartbeat += 1
            addrs = self._peer_addrs()
            if addrs:
                target = random.choice(addrs)
                ping = pb.Ping(**{"from": self._self_info()},
                               digest=[self._info_from_member(m) for m in list(self.members.values())[:16]],
                               ts=time.time())
                pong = await self._call(target, "Heartbeat", ping, timeout=2.5)
                if pong is None:
                    self.stats["hb_fail"] += 1
                    # SWIM indirect probe: ask k random peers to relay
                    for a in random.sample(addrs, min(2, len(addrs))):
                        if a != target:
                            await self._call(a, "Heartbeat", ping, timeout=2.0)
                else:
                    fi = getattr(pong, "from")
                    self.phi[fi.node_id].heartbeat()
                    self._merge_member(self._member_dict(fi))
                    for ni in pong.digest:
                        self._merge_member(self._member_dict(ni))
            await asyncio.sleep(self.cfg.get("hb_interval", 1.0))

    async def _loop_suspect(self):
        while not self._stop.is_set():
            now = time.time() * 1000.0
            phi_sus = self.cfg.get("phi_suspect", 8.0)
            phi_dead = self.cfg.get("phi_dead", 12.0)
            for nid, m in list(self.members.items()):
                p = self.phi[nid].phi(now)
                if p >= phi_dead and m["state"] != "DEAD":
                    m["state"] = "DEAD"
                elif p >= phi_sus and m["state"] == "ALIVE":
                    m["state"] = "SUSPECT"
                elif p < phi_sus and m["state"] == "SUSPECT":
                    m["state"] = "ALIVE"
            await asyncio.sleep(1.0)

    async def _loop_gossip(self):
        while not self._stop.is_set():
            addrs = self._peer_addrs()
            if addrs:
                fanout = min(self.cfg.get("gossip_fanout", 3), len(addrs))
                digest = pb.GossipDigest(
                    from_node=self.node_id,
                    members=[self._self_info()] + [self._info_from_member(m)
                                                   for m in list(self.members.values())[:24]],
                    intel=[self._intel_to_pb(e) for e in self._recent_intel(40)])
                for a in random.sample(addrs, fanout):
                    resp = await self._call(a, "Gossip", digest, timeout=2.5)
                    if resp:
                        for ni in resp.members:
                            self._merge_member(self._member_dict(ni))
                        for e in resp.intel:
                            self._merge_intel(self._pb_to_intel(e))
            await asyncio.sleep(self.cfg.get("gossip_interval", 1.5))

    async def _loop_flush_intel(self):
        while not self._stop.is_set():
            batch = self._local_threats.drain(64)
            if batch:
                self.stats["intel_tx"] += 1
                msg = pb.ThreatBatch(origin_node=self.node_id,
                                     events=[self._intel_to_pb(e) for e in batch])
                for a in self._peer_addrs():
                    asyncio.create_task(self._call(a, "ShareThreatIntel", msg))
            await asyncio.sleep(self.cfg.get("intel_interval", 0.5))

    async def _loop_telemetry(self):
        while not self._stop.is_set():
            snap = {}
            try:
                snap = self.engine.snapshot()
            except Exception:
                pass
            tel = pb.TelemetrySnapshot(
                node_id=self.node_id, pps=float(snap.get("pps", 0.0)),
                active_flows=float(snap.get("active_flows", 0.0)),
                dropped=float(snap.get("dropped", 0.0)),
                quarantined=float(snap.get("quarantined", 0.0)),
                cpu=float(snap.get("cpu", 0.0)), mem=float(snap.get("mem", 0.0)),
                latency_ms=float(snap.get("latency_ms", 0.0)),
                fpr=float(snap.get("realized_fpr", 0.0)), ts=time.time(),
                region=self.region, address=self.address)
            for a in self._peer_addrs():
                asyncio.create_task(self._call(a, "ShareTelemetry", tel))
            await asyncio.sleep(self.cfg.get("telemetry_interval", 2.0))

    async def _loop_model(self):
        while not self._stop.is_set():
            await asyncio.sleep(self.cfg.get("model_interval", 6.0))
            self.model_round += 1
            local = self._export_model()
            if local["weights"]:
                delta = pb.ModelDelta(node_id=self.node_id, fusion_weights=local["weights"],
                                      threshold=local["threshold"], eps_h=local["eps_h"],
                                      samples=int(getattr(self.engine, "_n", 1)),
                                      round=self.model_round, ts=time.time())
                for a in self._peer_addrs():
                    asyncio.create_task(self._call(a, "SyncModel", delta))
            self._aggregate_model()

    async def _loop_reap(self):
        while not self._stop.is_set():
            now = time.time()
            with self._intel_lock:
                stale = [ip for ip, e in self.intel.items() if now - e["ts"] > e.get("ttl", 900.0)]
                for ip in stale:
                    del self.intel[ip]
                    self._seen_blocks.discard(ip)
            await asyncio.sleep(15.0)

    async def _join_seeds(self):
        for addr in self.seeds:
            if addr and addr != self.address:
                ping = pb.Ping(**{"from": self._self_info()}, digest=[], ts=time.time())
                pong = await self._call(addr, "Heartbeat", ping, timeout=3.0)
                if pong:
                    fi = getattr(pong, "from")
                    self._merge_member(self._member_dict(fi))
                    self.phi[fi.node_id].heartbeat()
                    for ni in pong.digest:
                        self._merge_member(self._member_dict(ni))

    # ===================================================================== #
    #  lifecycle                                                            #
    # ===================================================================== #
    async def start(self):
        mode, servcreds, _ = self._credentials()
        self._server = grpc_aio.server(options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000)])
        pbg.add_SauronMeshServicer_to_server(self, self._server)
        bind = self.cfg["bind"]
        if mode == "mtls":
            self._server.add_secure_port(bind, servcreds)
        else:
            self._server.add_insecure_port(bind)
        await self._server.start()
        await self._join_seeds()
        for coro in (self._loop_heartbeat, self._loop_suspect, self._loop_gossip,
                     self._loop_flush_intel, self._loop_telemetry, self._loop_model,
                     self._loop_reap):
            self._tasks.append(asyncio.create_task(coro()))
        print(f"[mesh] node {self.node_id} listening on {bind} "
              f"({mode}); seeds={self.seeds or 'none'}; region={self.region}")

    async def stop(self):
        if self._stop:
            self._stop.set()
        for t in self._tasks:
            t.cancel()
        if self._server:
            await self._server.stop(grace=1.0)
        for ch in self._chan.values():
            try:
                await ch.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 6.  thread-safe simple queue                                                #
# --------------------------------------------------------------------------- #
class _SimpleQ:
    def __init__(self, maxlen: int = 20000):
        self._dq: Deque[Any] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, x: Any) -> None:
        with self._lock:
            self._dq.append(x)

    def drain(self, limit: int) -> List[Any]:
        out: List[Any] = []
        with self._lock:
            while self._dq and len(out) < limit:
                out.append(self._dq.popleft())
        return out


# --------------------------------------------------------------------------- #
# 7.  backend entry point                                                     #
# --------------------------------------------------------------------------- #
def _env_seeds() -> List[str]:
    raw = os.environ.get("SAURON_MESH_SEEDS", "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def maybe_start_mesh(engine, src_holder: Dict[str, Any]) -> Optional[MeshNode]:
    """Construct a mesh node from the environment. Returns None (mesh disabled)
    unless SAURON_MESH_ENABLE is truthy. The caller (build_app lifespan) awaits
    node.start(). Env vars:

        SAURON_MESH_ENABLE   = 1 to enable
        SAURON_NODE_ID       = unique node id            (default: hostname)
        SAURON_MESH_BIND     = listen addr               (default: 0.0.0.0:50151)
        SAURON_MESH_ADVERTISE= addr peers dial           (default: <host>:50151)
        SAURON_MESH_SEEDS    = comma list of seed peers  (e.g. n1:50151,n2:50151)
        SAURON_MESH_REGION   = edge|5g|6g|b6g|cloud      (default: cloud)
        SAURON_MESH_APPLY_MODEL = 1 to apply federated model back into ADE (off)
        SAURON_MESH_TLS_CERT/_KEY/_CA = mTLS material    (optional)
    """
    if os.environ.get("SAURON_MESH_ENABLE", "").lower() not in ("1", "true", "yes", "on"):
        return None
    if not GRPC_OK:
        print(f"[mesh] SAURON_MESH_ENABLE set but grpc unavailable: {_IMPORT_ERR}\n"
              f"[mesh] install with: pip install grpcio grpcio-tools --break-system-packages")
        return None
    host = socket.gethostname()
    bind = os.environ.get("SAURON_MESH_BIND", "0.0.0.0:50151")
    port = bind.rsplit(":", 1)[-1]
    advertise = os.environ.get("SAURON_MESH_ADVERTISE", f"{host}:{port}")
    cfg = {
        "node_id": os.environ.get("SAURON_NODE_ID", host),
        "bind": bind, "advertise": advertise, "seeds": _env_seeds(),
        "region": os.environ.get("SAURON_MESH_REGION", "cloud"),
        "apply_model": os.environ.get("SAURON_MESH_APPLY_MODEL", "").lower() in ("1", "true", "yes"),
        "tls_cert": os.environ.get("SAURON_MESH_TLS_CERT"),
        "tls_key": os.environ.get("SAURON_MESH_TLS_KEY"),
        "tls_ca": os.environ.get("SAURON_MESH_TLS_CA"),
    }
    return MeshNode(engine, src_holder, cfg)

"""
SAURON++ — Energy & sustainability instrumentation
==================================================

Measures the energy cost of running the detection pipeline and derives the
efficiency metrics a security-systems paper needs.

MEASUREMENT vs ESTIMATION (stated explicitly everywhere, including the report):

  * MEASURED  — Intel/AMD RAPL energy counters exposed at
                /sys/class/powercap/intel-rapl*/energy_uj. These are real
                hardware Joule counters for the CPU package and (where present)
                the DRAM domain. Requires bare-metal Linux; unavailable inside
                WSL2, most containers, and most VMs.
  * ESTIMATED — When RAPL is absent we fall back to a CPU-utilisation x TDP
                model. That is a *model*, not a measurement, and every value it
                produces is flagged `"method": "estimated"` so results are never
                silently misreported.

Network-interface energy has no hardware counter on commodity NICs, so it is
always modelled from a per-bit energy coefficient (see NIC_NJ_PER_BIT).

Public API
----------
    em = EnergyMeter()                 # auto-detects RAPL
    em.sample(packets, attacks, bits)  # call periodically
    em.snapshot()                      # -> dict of all energy metrics
    gate = EnergyAwareFilter(budget_w=8.0)
    gate.allow_expensive(current_watts) # -> bool, for energy-aware filtering
"""
from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Tunable physical constants (documented so a reviewer can check them)
# ---------------------------------------------------------------------------
DEFAULT_TDP_W = 45.0          # assumed package TDP for the estimation fallback
NIC_NJ_PER_BIT = 0.5          # ~0.1-1 nJ/bit for modern wired NICs (modelled)
NIC_IDLE_W = 0.8              # NIC baseline draw when link is up (modelled)
# Grid carbon intensity, gCO2 per kWh. World avg ~475; India ~713 (IEA/Ember).
CARBON_G_PER_KWH = float(os.environ.get("SAURON_CARBON_G_PER_KWH", "713"))


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except Exception:
        return None


class RaplReader:
    """Reads Intel/AMD RAPL energy counters (microjoules, monotonic, wrapping).

    Each powercap domain exposes energy_uj plus max_energy_range_uj; the counter
    wraps, so deltas are computed modulo that range.
    """

    def __init__(self) -> None:
        self.domains: List[Dict] = []
        for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            ep = os.path.join(d, "energy_uj")
            if not os.path.exists(ep):
                continue
            name = "unknown"
            try:
                with open(os.path.join(d, "name")) as fh:
                    name = fh.read().strip()
            except Exception:
                pass
            rng = _read_int(os.path.join(d, "max_energy_range_uj")) or (2 ** 32)
            if _read_int(ep) is None:      # exists but unreadable (needs root)
                continue
            self.domains.append({"path": ep, "name": name, "range": rng,
                                 "last": _read_int(ep)})

    @property
    def available(self) -> bool:
        return bool(self.domains)

    def delta_joules(self) -> Dict[str, float]:
        """Joules consumed per domain since the previous call."""
        out: Dict[str, float] = {}
        for d in self.domains:
            cur = _read_int(d["path"])
            if cur is None:
                continue
            prev = d["last"]
            diff = cur - prev
            if diff < 0:                    # counter wrapped
                diff += d["range"]
            d["last"] = cur
            out[d["name"]] = out.get(d["name"], 0.0) + diff / 1e6
        return out


class EnergyMeter:
    """Accumulates energy and derives the efficiency metrics.

    Call sample() periodically with cumulative counters; it converts them into
    per-interval deltas internally.
    """

    def __init__(self, tdp_w: float = DEFAULT_TDP_W,
                 carbon_g_per_kwh: float = CARBON_G_PER_KWH) -> None:
        self.rapl = RaplReader()
        self.method = "measured (RAPL)" if self.rapl.available else "estimated (CPU% x TDP)"
        self.tdp_w = tdp_w
        self.carbon = carbon_g_per_kwh

        self.t0 = time.time()
        self._last_t = self.t0
        self._last_cpu_time = self._proc_cpu_seconds()

        # cumulative totals
        self.cpu_j = 0.0            # CPU package (+ core) energy
        self.dram_j = 0.0           # DRAM domain, when present
        self.nic_j = 0.0            # modelled NIC energy
        self.baseline_j = 0.0       # idle-attributable share (for overhead calc)
        self.packets = 0
        self.attacks = 0
        self.bits = 0
        self.watts = 0.0            # instantaneous package power
        self.peak_watts = 0.0
        self.attack_watts: List[float] = []   # power sampled while attacks seen
        self.idle_watts: List[float] = []     # power sampled with no attacks
        self._prev = {"packets": 0, "attacks": 0, "bits": 0}

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _proc_cpu_seconds() -> float:
        try:
            return sum(os.times()[:2])          # user + system CPU time
        except Exception:
            return 0.0

    # -- main sampling ----------------------------------------------------
    def sample(self, packets: int, attacks: int, bits: int,
               cpu_pct: Optional[float] = None) -> None:
        """Record one interval. `packets`/`attacks`/`bits` are CUMULATIVE."""
        now = time.time()
        dt = max(1e-6, now - self._last_t)
        self._last_t = now

        d_pkts = max(0, packets - self._prev["packets"])
        d_atk = max(0, attacks - self._prev["attacks"])
        d_bits = max(0, bits - self._prev["bits"])
        self._prev = {"packets": packets, "attacks": attacks, "bits": bits}
        self.packets, self.attacks, self.bits = packets, attacks, bits

        if self.rapl.available:
            dj = self.rapl.delta_joules()
            pkg = sum(v for k, v in dj.items() if k.startswith("package") or k == "core")
            dram = sum(v for k, v in dj.items() if k == "dram")
            self.cpu_j += pkg
            self.dram_j += dram
            self.watts = pkg / dt
        else:
            # Estimation fallback: process CPU-seconds x TDP fraction.
            cpu_now = self._proc_cpu_seconds()
            d_cpu = max(0.0, cpu_now - self._last_cpu_time)
            self._last_cpu_time = cpu_now
            if cpu_pct is not None and d_cpu == 0.0:
                d_cpu = dt * (cpu_pct / 100.0)
            j = d_cpu * self.tdp_w
            self.cpu_j += j
            self.watts = j / dt

        # NIC energy is always modelled (no hardware counter on commodity NICs)
        self.nic_j += (d_bits * NIC_NJ_PER_BIT / 1e9) + (NIC_IDLE_W * dt)

        self.peak_watts = max(self.peak_watts, self.watts)
        (self.attack_watts if d_atk > 0 else self.idle_watts).append(self.watts)
        if len(self.attack_watts) > 4000:
            del self.attack_watts[:2000]
        if len(self.idle_watts) > 4000:
            del self.idle_watts[:2000]

    # -- derived metrics --------------------------------------------------
    def snapshot(self) -> Dict:
        elapsed = max(1e-6, time.time() - self.t0)
        node_j = self.cpu_j + self.dram_j + self.nic_j
        avg_idle = (sum(self.idle_watts) / len(self.idle_watts)) if self.idle_watts else 0.0
        avg_atk = (sum(self.attack_watts) / len(self.attack_watts)) if self.attack_watts else 0.0
        # Security overhead: power above the observed idle floor, attributed to
        # the detection pipeline running.
        overhead_w = max(0.0, self.watts - avg_idle) if avg_idle else 0.0
        kwh = node_j / 3.6e6
        return {
            "method": self.method,
            "measured": self.rapl.available,
            "elapsed_s": round(elapsed, 2),

            # --- energy totals (Joules) ---
            "cpu_energy_j": round(self.cpu_j, 3),
            "dram_energy_j": round(self.dram_j, 3),
            "nic_energy_j_modelled": round(self.nic_j, 3),
            "node_energy_j": round(node_j, 3),

            # --- power ---
            "power_w": round(self.watts, 3),
            "peak_power_w": round(self.peak_watts, 3),
            "avg_power_idle_w": round(avg_idle, 3),
            "power_during_attack_w": round(avg_atk, 3),
            "security_overhead_w": round(overhead_w, 3),

            # --- efficiency ---
            "energy_per_packet_uj": round(self.cpu_j / self.packets * 1e6, 3) if self.packets else 0.0,
            "energy_per_attack_mj": round(self.cpu_j / self.attacks * 1e3, 3) if self.attacks else 0.0,
            "bits_per_joule": round(self.bits / self.cpu_j, 1) if self.cpu_j > 0 else 0.0,
            "packets_per_joule": round(self.packets / self.cpu_j, 1) if self.cpu_j > 0 else 0.0,

            # --- sustainability ---
            "energy_kwh": round(kwh, 9),
            "carbon_g_co2": round(kwh * self.carbon, 6),
            "carbon_intensity_g_per_kwh": self.carbon,
        }

    def cluster_snapshot(self, peer_node_energy_j: Optional[List[float]] = None) -> Dict:
        """Cluster energy = this node + energy reported by mesh peers."""
        mine = self.snapshot()["node_energy_j"]
        peers = list(peer_node_energy_j or [])
        return {"cluster_nodes": 1 + len(peers),
                "cluster_energy_j": round(mine + sum(peers), 3),
                "node_energy_j": mine}


class EnergyAwareFilter:
    """Energy-aware packet filtering (novel contribution).

    Under a power budget the pipeline sheds its most expensive work first: when
    measured power exceeds the budget, the costly detectors (RRCF isolation and
    Spectral Residual FFT) are skipped for low-suspicion traffic, while the
    cheap ones (HDC bipolar dot-product, ECOD tail lookup) always run. High
    suspicion always gets the full bank, so the energy saving never comes out of
    security for traffic that actually looks dangerous.

    This gives a controllable energy-latency-accuracy trade-off rather than a
    fixed cost.
    """

    EXPENSIVE = ("rrcf", "sr")

    def __init__(self, budget_w: float = 0.0, suspicion_override: float = 0.6) -> None:
        self.budget_w = budget_w          # 0 disables the mechanism
        self.suspicion_override = suspicion_override
        self.skipped = 0
        self.evaluated = 0
        self.energy_saved_j = 0.0

    @property
    def enabled(self) -> bool:
        return self.budget_w > 0

    def allow_expensive(self, watts: float, suspicion: float = 0.0) -> bool:
        """True if the expensive detectors should run for this record."""
        self.evaluated += 1
        if not self.enabled:
            return True
        if suspicion >= self.suspicion_override:
            return True                    # never trade away security under threat
        if watts > self.budget_w:
            self.skipped += 1
            # ~60% of per-packet detector cost sits in RRCF+SR
            self.energy_saved_j += 0.6 * (watts / max(1.0, self.evaluated))
            return False
        return True

    def snapshot(self) -> Dict:
        return {"enabled": self.enabled, "budget_w": self.budget_w,
                "records_evaluated": self.evaluated,
                "expensive_skipped": self.skipped,
                "skip_ratio": round(self.skipped / self.evaluated, 5) if self.evaluated else 0.0,
                "energy_saved_j_est": round(self.energy_saved_j, 3)}


if __name__ == "__main__":   # quick self-test
    em = EnergyMeter()
    print("RAPL available:", em.rapl.available, "| method:", em.method)
    pkts = bits = atks = 0
    for i in range(5):
        pkts += 1000; bits += 8_000_000; atks += 12
        sum(x * x for x in range(200000))          # burn CPU
        em.sample(pkts, atks, bits, cpu_pct=100.0)
        time.sleep(0.1)
    for k, v in em.snapshot().items():
        print(f"  {k:<32}: {v}")

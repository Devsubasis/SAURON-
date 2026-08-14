"""Dataset profiling, cleaning and imbalance handling for SAURON++.

Prints a full profile of a labeled dataset (rows, columns, dtypes, missing
values, duplicates, class distribution, imbalance ratio), applies a documented
cleaning pipeline, and — when explicitly asked — rebalances.

A deliberate methodological note, because it matters for the numbers you
report:

    Synthetic oversampling (SMOTE and friends) must NEVER be applied to the
    data you evaluate on. It fabricates minority rows, so the model is scored
    on points that were invented from the answers; precision/recall/F1 come out
    inflated and the result is not defensible. This is one of the most common
    errors in published IDS work.

    The right way to handle class imbalance in an evaluation is to leave the
    test distribution untouched and report imbalance-robust metrics: MCC,
    PR-AUC, balanced accuracy and per-class recall — all of which SAURON++
    already computes.

So the default here is `none` (clean but do not resample), and anything else
prints a loud warning describing exactly what was done, so it can be reported
honestly.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Profiling                                                                     #
# --------------------------------------------------------------------------- #


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


class DatasetProfile:
    """Streaming profile of a labeled CSV dataset (memory-flat)."""

    def __init__(self, columns: Sequence[str], label_idx: int):
        self.columns = list(columns)
        self.label_idx = label_idx
        self.rows = 0
        self.classes: Counter = Counter()      # raw label -> count
        self.binary: Counter = Counter()       # 0/1 -> count
        self.missing = [0] * len(self.columns)
        self.non_numeric = [0] * len(self.columns)
        self.infinite = 0
        self.duplicates = 0
        self.malformed = 0
        self._seen = set()
        self._const_probe: List[Optional[str]] = [None] * len(self.columns)
        self._const = [True] * len(self.columns)

    def observe(self, row: Sequence[str], label_bin: int, raw_label: str) -> bool:
        """Record one row. Returns False if the row should be dropped."""
        self.rows += 1
        if len(row) < len(self.columns):
            self.malformed += 1
            return False
        keep = True
        for i, v in enumerate(row[:len(self.columns)]):
            sv = (v or "").strip()
            if sv == "" or sv.lower() in ("nan", "na", "null", "none", "?"):
                self.missing[i] += 1
                keep = False
            elif sv.lower() in ("inf", "-inf", "infinity", "-infinity"):
                self.infinite += 1
                keep = False
            elif not _is_num(sv):
                self.non_numeric[i] += 1
            if self._const[i]:
                if self._const_probe[i] is None:
                    self._const_probe[i] = sv
                elif self._const_probe[i] != sv:
                    self._const[i] = False
        # duplicate detection on a bounded hash set
        if len(self._seen) < 2_000_000:
            h = hash(tuple(row))
            if h in self._seen:
                self.duplicates += 1
                keep = False
            else:
                self._seen.add(h)
        self.classes[raw_label] += 1
        self.binary[label_bin] += 1
        return keep

    # ---- derived -------------------------------------------------------- #
    @property
    def constant_columns(self) -> List[str]:
        return [self.columns[i] for i, c in enumerate(self._const)
                if c and self._const_probe[i] is not None]

    @property
    def imbalance_ratio(self) -> float:
        maj, mino = self.binary.get(0, 0), self.binary.get(1, 0)
        hi, lo = max(maj, mino), min(maj, mino)
        return (hi / lo) if lo else float("inf")

    def severity(self) -> str:
        r = self.imbalance_ratio
        if r == float("inf"):
            return "DEGENERATE (only one class present)"
        if r < 1.5:
            return "BALANCED"
        if r < 4:
            return "MILD imbalance"
        if r < 10:
            return "MODERATE imbalance"
        if r < 100:
            return "SEVERE imbalance"
        return "EXTREME imbalance"

    # ---- reporting ------------------------------------------------------ #
    def report(self, name: str = "", width: int = 78) -> str:
        L: List[str] = []
        a = L.append
        a("=" * width)
        a(f"  DATASET PROFILE {('· ' + name) if name else ''}")
        a("=" * width)
        a(f"  rows read            : {self.rows:,}")
        a(f"  columns              : {len(self.columns)}")
        a(f"  label column         : '{self.columns[self.label_idx].strip()}' "
          f"(index {self.label_idx})")
        a("-" * width)
        a("  CLASS DISTRIBUTION")
        tot = sum(self.classes.values()) or 1
        for lab, n in self.classes.most_common(12):
            bar = "#" * max(1, int(38 * n / tot))
            a(f"    {lab[:22]:<22} {n:>9,}  {100*n/tot:>6.2f}%  {bar}")
        if len(self.classes) > 12:
            a(f"    ... and {len(self.classes)-12} more classes")
        a("-" * width)
        ben, mal = self.binary.get(0, 0), self.binary.get(1, 0)
        a(f"  binary split         : benign={ben:,} ({100*ben/tot:.2f}%)  "
          f"attack={mal:,} ({100*mal/tot:.2f}%)")
        ir = self.imbalance_ratio
        a(f"  imbalance ratio      : {('inf' if ir == float('inf') else f'{ir:.2f}:1')}"
          f"   -> {self.severity()}")
        a("-" * width)
        a("  DATA QUALITY")
        a(f"    rows with missing values : {sum(self.missing):,}")
        a(f"    infinite values          : {self.infinite:,}")
        a(f"    duplicate rows           : {self.duplicates:,}")
        a(f"    malformed rows           : {self.malformed:,}")
        cc = self.constant_columns
        a(f"    constant (zero-variance) : {len(cc)}"
          + (f"  e.g. {', '.join(c.strip() for c in cc[:4])}" if cc else ""))
        worst = sorted(range(len(self.columns)), key=lambda i: -self.missing[i])[:3]
        worst = [i for i in worst if self.missing[i]]
        if worst:
            a("    most-missing columns     : "
              + ", ".join(f"{self.columns[i].strip()}({self.missing[i]:,})" for i in worst))
        a("=" * width)
        return "\n".join(L)

    def as_dict(self) -> Dict:
        return {
            "rows": self.rows, "columns": len(self.columns),
            "column_names": [c.strip() for c in self.columns],
            "class_distribution": dict(self.classes),
            "binary_distribution": {"benign": self.binary.get(0, 0),
                                    "attack": self.binary.get(1, 0)},
            "imbalance_ratio": (None if self.imbalance_ratio == float("inf")
                                else round(self.imbalance_ratio, 3)),
            "imbalance_severity": self.severity(),
            "quality": {"rows_with_missing": sum(self.missing),
                        "infinite_values": self.infinite,
                        "duplicate_rows": self.duplicates,
                        "malformed_rows": self.malformed,
                        "constant_columns": self.constant_columns},
        }


# --------------------------------------------------------------------------- #
# Imbalance handling                                                            #
# --------------------------------------------------------------------------- #
BALANCE_METHODS = ("none", "undersample", "class-weight", "smote-tomek")


class Rebalancer:
    """Streaming class-imbalance handling.

    Methods
    -------
    none          Leave the distribution untouched (DEFAULT, and the correct
                  choice for evaluation). Imbalance is handled by *reporting*
                  MCC / PR-AUC / balanced accuracy rather than by editing data.
    undersample   Randomly drop majority-class rows to a target ratio. Removes
                  only real data — never invents any — so metrics stay honest,
                  at the cost of discarding benign samples.
    class-weight  Keeps every row and instead raises the effective cost of the
                  minority class in the decision loop (cost-sensitive learning).
                  Preferred over resampling for streaming detectors.
    smote-tomek   Hybrid over+under sampling (SMOTE-style interpolation plus
                  Tomek-link cleaning). Stronger than plain SMOTE because Tomek
                  links remove borderline/overlapping pairs after interpolation.
                  TRAINING ONLY -- applying it to evaluation data fabricates
                  test points and invalidates the reported metrics.
    """

    def __init__(self, method: str = "none", target_ratio: float = 1.0,
                 seed: int = 7, k: int = 5):
        m = (method or "none").lower()
        if m not in BALANCE_METHODS:
            raise ValueError(f"balance method must be one of {BALANCE_METHODS}")
        self.method = m
        self.target_ratio = max(1.0, float(target_ratio))
        self.rng = random.Random(seed)
        self.k = k
        self.kept = Counter()
        self.dropped = 0
        self.synthesized = 0
        self.tomek_removed = 0
        self._minority_buf: List[Tuple[List[float], str]] = []

    # -- streaming decision: keep this row? ------------------------------- #
    def accept(self, label_bin: int, prior_ratio: float) -> bool:
        if self.method in ("none", "class-weight", "smote-tomek"):
            self.kept[label_bin] += 1
            return True
        if self.method == "undersample":
            # keep every minority row; keep majority rows with probability p
            if label_bin == 1:
                self.kept[1] += 1
                return True
            p = self.target_ratio / prior_ratio if prior_ratio > 0 else 1.0
            if self.rng.random() <= min(1.0, p):
                self.kept[0] += 1
                return True
            self.dropped += 1
            return False
        return True

    def class_weight(self, prior_ratio: float) -> float:
        """Cost multiplier for the minority class (1.0 when not weighting)."""
        return prior_ratio if self.method == "class-weight" else 1.0

    # -- offline SMOTE-Tomek (training use only) -------------------------- #
    def smote_tomek(self, X: List[List[float]], y: List[int]
                    ) -> Tuple[List[List[float]], List[int]]:
        """SMOTE interpolation followed by Tomek-link cleaning.

        Returns a NEW (X, y). Intended for training a downstream model; do not
        use on data you will report metrics on.
        """
        if self.method != "smote-tomek" or not X:
            return X, y
        maj = [i for i, v in enumerate(y) if v == 0]
        mino = [i for i, v in enumerate(y) if v == 1]
        if not mino or not maj:
            return X, y
        minority_is_1 = len(mino) < len(maj)
        small, big = (mino, maj) if minority_is_1 else (maj, mino)
        small_lab = 1 if minority_is_1 else 0
        need = int(len(big) / self.target_ratio) - len(small)
        Xo = [list(r) for r in X]
        yo = list(y)

        def d2(a, b):
            return sum((p - q) ** 2 for p, q in zip(a, b))

        # --- SMOTE: interpolate between a point and one of its k neighbours
        for _ in range(max(0, need)):
            i = self.rng.choice(small)
            nb = sorted((j for j in small if j != i), key=lambda j: d2(X[i], X[j]))[:self.k]
            if not nb:
                break
            j = self.rng.choice(nb)
            g = self.rng.random()
            Xo.append([a + g * (b - a) for a, b in zip(X[i], X[j])])
            yo.append(small_lab)
            self.synthesized += 1

        # --- Tomek links: drop majority members of mutually-nearest opposite pairs
        n = len(Xo)
        if n <= 4000:                       # O(n^2); guard on size
            nearest = []
            for i in range(n):
                best, bd = -1, float("inf")
                for j in range(n):
                    if i == j:
                        continue
                    d = d2(Xo[i], Xo[j])
                    if d < bd:
                        bd, best = d, j
                nearest.append(best)
            drop = set()
            for i in range(n):
                j = nearest[i]
                if j >= 0 and nearest[j] == i and yo[i] != yo[j]:
                    drop.add(i if yo[i] != small_lab else j)   # remove majority side
            if drop:
                self.tomek_removed = len(drop)
                Xo = [r for i, r in enumerate(Xo) if i not in drop]
                yo = [v for i, v in enumerate(yo) if i not in drop]
        return Xo, yo

    # -- reporting -------------------------------------------------------- #
    def report(self, width: int = 78) -> str:
        L = ["-" * width, "  IMBALANCE HANDLING"]
        if self.method == "none":
            L += ["    method  : none (distribution left untouched)",
                  "    rationale: correct for EVALUATION. Imbalance is reported, not",
                  "               edited - see MCC, PR-AUC and balanced accuracy below,",
                  "               which stay meaningful under skew (unlike raw accuracy)."]
        elif self.method == "undersample":
            L += [f"    method  : stratified majority undersampling -> {self.target_ratio:.2f}:1",
                  f"    dropped : {self.dropped:,} majority rows (no data invented)",
                  f"    kept    : benign={self.kept.get(0,0):,} attack={self.kept.get(1,0):,}",
                  "    NOTE    : results are on a resampled subset - state this when reporting."]
        elif self.method == "class-weight":
            L += ["    method  : cost-sensitive class weighting (no rows added or removed)",
                  "    rationale: preferred for streaming detectors - every real sample is",
                  "               kept and the minority class simply costs more to miss."]
        else:
            L += ["    method  : SMOTE-Tomek (interpolate minority, then clean Tomek links)",
                  f"    synthesized      : {self.synthesized:,} rows",
                  f"    tomek-link drops : {self.tomek_removed:,} rows",
                  "    *** WARNING: synthetic rows are present. This is valid for TRAINING",
                  "    *** only. Metrics computed over synthesized data are NOT valid to",
                  "    *** report as detection performance."]
        L.append("-" * width)
        return "\n".join(L)

    def as_dict(self) -> Dict:
        return {"method": self.method, "target_ratio": self.target_ratio,
                "dropped_majority": self.dropped, "synthesized": self.synthesized,
                "tomek_removed": self.tomek_removed,
                "kept": {"benign": self.kept.get(0, 0), "attack": self.kept.get(1, 0)}}

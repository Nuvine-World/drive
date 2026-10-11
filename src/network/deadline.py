"""Deadline-constrained per-frame communication for cooperative perception.

Every frame, each collaborator decides how much of its BEV feature map to send
to the receiver before it knows how the uplink will behave. A message that
arrives after the receiver's fusion deadline is not lost: it can still be fused
in later frames as a stale message whose age the detector sees.

``DRIVEController`` chooses the budget ``b`` that maximizes

    V * U(b) - Z * P_miss(b)

where ``U(b)`` is the share of the collaborator's own learned utility that the
``b`` cells carry, ``P_miss(b)`` comes from a past-only log-normal throughput
estimate, and ``Z`` is a virtual queue updated every frame as
``Z <- max(Z + miss - epsilon, 0)`` (Lyapunov drift-plus-penalty). This holds
the long-run fraction of frames whose message is late at or below ``epsilon``
while spending the remaining link time on the most useful cells. Because ``Z``
drains by ``epsilon`` on every frame, a pause in sending always lets it
recover; a constraint counted only over sent messages can lock the controller
into never sending. A chance constraint excludes any message
predicted to miss with probability above ``max_miss_probability``; without
it an empty virtual queue lets the first frames flood the link with a
backlog the queue cannot undo. No decision uses labels, detections, or the
current or future link state.
"""

from collections import deque
from dataclasses import dataclass
import csv
import math

import numpy as np

from src.network.control import select_deadline_budget


@dataclass(frozen=True)
class Option:
    """One choice for a collaborator: ``budget`` 0 sends nothing."""

    budget: float
    message_bytes: int
    utility: float


def sparse_bytes(cells, budget, channels, index_bytes, value_bytes):
    """Wire size of the top ``budget`` cells: one index and every channel value."""
    selected = math.ceil(cells * budget)
    return selected * (index_bytes + channels * value_bytes)


def dense_bytes(cells, channels, value_bytes):
    return cells * channels * value_bytes


class ConstantTrace:
    def __init__(self, mbps):
        self.mbps = float(mbps)

    def bandwidth(self, frame):
        return self.mbps


class GilbertElliottTrace:
    """Two-state Markov uplink (good/bad) with log-normal per-frame jitter."""

    def __init__(self, good_mbps, bad_mbps, p_good_to_bad, p_bad_to_good, jitter, seed):
        self._rng = np.random.default_rng(seed)
        self._levels = (float(good_mbps), float(bad_mbps))
        self._switch = (float(p_good_to_bad), float(p_bad_to_good))
        self._jitter = float(jitter)
        self._values = []
        self._state = 0

    def bandwidth(self, frame):
        while len(self._values) <= frame:
            if self._rng.random() < self._switch[self._state]:
                self._state = 1 - self._state
            noise = math.exp(self._jitter * self._rng.standard_normal())
            self._values.append(self._levels[self._state] * noise)
        return self._values[frame]


class CsvTrace:
    """Measured throughput from a ``time_s,mbps`` CSV, read cyclically."""

    def __init__(self, path, period_ms, offset_s=0.0):
        with open(path, newline="") as stream:
            rows = [
                (float(row["time_s"]), float(row["mbps"]))
                for row in csv.DictReader(stream)
            ]
        if not rows or any(mbps <= 0 for _, mbps in rows):
            raise ValueError(f"Trace {path} needs positive time_s,mbps rows")
        rows.sort()
        self._times = np.array([time for time, _ in rows])
        self._values = np.array([mbps for _, mbps in rows])
        self._period_s = period_ms / 1000
        self._offset_s = float(offset_s)
        # The last sample holds for one typical sampling interval before the
        # trace repeats.
        spacing = float(np.median(np.diff(self._times))) if len(rows) > 1 else 0.0
        self._span = max(self._times[-1] - self._times[0] + spacing, self._period_s)

    def bandwidth(self, frame):
        time = (self._offset_s + frame * self._period_s) % self._span + self._times[0]
        index = max(0, int(np.searchsorted(self._times, time, side="right")) - 1)
        return float(self._values[index])


def make_trace(spec, seed, period_ms):
    kind = spec["type"]
    if kind == "constant":
        return ConstantTrace(spec["mbps"])
    if kind == "gilbert_elliott":
        return GilbertElliottTrace(
            spec["good_mbps"],
            spec["bad_mbps"],
            spec["p_good_to_bad"],
            spec["p_bad_to_good"],
            spec.get("jitter", 0.0),
            seed,
        )
    if kind == "csv":
        # Each link starts at its own point of the measured trace.
        offset = (seed % 10_000) * period_ms / 1000
        return CsvTrace(spec["path"], period_ms, offset)
    raise ValueError(f"Unknown trace type: {kind}")


class Link:
    """FIFO uplink from one collaborator to the receiver."""

    def __init__(self, trace, rtt_ms):
        self.trace = trace
        self.rtt_ms = float(rtt_ms)
        self.busy_until_ms = 0.0

    def queue_ms(self, ready_ms):
        return max(0.0, self.busy_until_ms - ready_ms)

    def send(self, frame, ready_ms, message_bytes):
        """Transmit after the queue drains; return ``(arrival_ms, duration_ms)``."""
        start = max(ready_ms, self.busy_until_ms)
        duration = message_bytes * 8 / (self.trace.bandwidth(frame) * 1e6) * 1000
        self.busy_until_ms = start + duration
        return self.busy_until_ms + self.rtt_ms / 2, duration


class OutcomeLog:
    """Delivery outcomes released to the sender only once they are known.

    The sender learns a message's throughput and whether it met its deadline
    when its transmission completes, never earlier.
    """

    def __init__(self):
        self._pending = deque()

    def record(self, known_ms, frame, message_bytes, duration_ms, missed):
        self._pending.append((known_ms, frame, message_bytes, duration_ms, missed))

    def release(self, now_ms):
        released = []
        while self._pending and self._pending[0][0] <= now_ms:
            released.append(self._pending.popleft()[1:])
        return released


class FixedController:
    """Always the same budget (the static-budget baseline)."""

    def __init__(self, budget):
        self.budget = float(budget)

    def choose(self, options, context):
        return next(option for option in options if option.budget == self.budget)

    def outcome(self, sent, missed):
        pass

    def frame(self):
        pass


class GreedyController:
    """Largest budget whose wire size fits the window at the EWMA throughput."""

    def choose(self, options, context):
        budgets = [option.budget for option in options if option.budget > 0]
        chosen = select_deadline_budget(
            budgets,
            context["cells"],
            context["channels"],
            1,
            context["estimator"].state(context["frame"], context["queue_ms"]),
            context["horizon_ms"],
            context["fixed_ms"],
            context["index_bytes"],
            context["value_bytes"],
        )
        return next(option for option in options if option.budget == chosen)

    def outcome(self, sent, missed):
        pass

    def frame(self):
        pass


class DRIVEController:
    """Lyapunov drift-plus-penalty budget control under a miss-rate target."""

    def __init__(self, tradeoff, target_miss_rate, max_miss_probability=0.5):
        self.tradeoff = float(tradeoff)
        self.target = float(target_miss_rate)
        self.max_miss_probability = float(max_miss_probability)
        self.virtual_queue = 0.0

    def choose(self, options, context):
        estimator, window = context["estimator"], context["window_ms"]
        scored = []
        for option in options:
            sent = option.budget > 0
            miss = estimator.miss_probability(option.message_bytes, window)
            if sent and miss > self.max_miss_probability:
                continue
            scored.append(
                (self.tradeoff * option.utility - self.virtual_queue * miss, option)
            )
        return max(scored, key=lambda item: item[0])[1]

    def frame(self):
        """Every frame drains the virtual queue by the miss-rate target."""
        self.virtual_queue = max(0.0, self.virtual_queue - self.target)

    def outcome(self, sent, missed):
        self.virtual_queue += float(missed)


class OracleController:
    """Upper bound: knows the true throughput of the frame being sent."""

    def choose(self, options, context):
        mbps = context["true_mbps"]
        feasible = [
            option
            for option in options
            if option.message_bytes * 8 / (mbps * 1e6) * 1000 <= context["window_ms"]
        ]
        return max(feasible, key=lambda option: option.budget)

    def outcome(self, sent, missed):
        pass

    def frame(self):
        pass


def feature_controllers(
    names, budgets, tradeoffs, target_miss_rate, max_miss_probability=0.5
):
    """``{label: factory}`` for the feature-message controllers in ``names``.

    DRIVE runs once per tradeoff ``V``, tracing utility against miss rate.
    """
    factories = {}
    if "fixed" in names:
        for budget in budgets:
            factories[f"fixed@{budget:g}"] = lambda budget=budget: FixedController(
                budget
            )
    if "greedy" in names:
        factories["greedy"] = GreedyController
    if "drive" in names:
        for tradeoff in tradeoffs:
            factories[f"drive@{tradeoff:g}"] = (
                lambda tradeoff=tradeoff: DRIVEController(
                    tradeoff, target_miss_rate, max_miss_probability
                )
            )
    if "oracle" in names:
        factories["oracle"] = OracleController
    return factories

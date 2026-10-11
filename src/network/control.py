"""Leakage-safe primitives for deadline-aware communication experiments.

These functions deliberately accept no labels, detections, or future samples.
They form the causal boundary of the deadline controllers in
``src.network.deadline``.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LinkState:
    bandwidth_mbps: float
    queue_ms: float
    observed_through_frame: int


class PastOnlyLinkEstimator:
    """EWMA throughput estimator that rejects future observations.

    Besides the EWMA throughput it tracks an EWMA mean and variance of log
    throughput, from which ``miss_probability`` predicts whether a message
    will cross the link within a time window (log-normal throughput).
    """

    def __init__(self, initial_mbps, alpha=0.2, initial_log_std=0.5):
        if initial_mbps <= 0 or not 0 < alpha <= 1 or initial_log_std <= 0:
            raise ValueError("invalid link-estimator settings")
        self._estimate = float(initial_mbps)
        self._alpha = float(alpha)
        self._last_frame = -1
        self._log_mean = math.log(initial_mbps)
        self._log_variance = float(initial_log_std) ** 2

    def observe(self, frame_index, delivered_bytes, duration_ms):
        if delivered_bytes < 0 or duration_ms <= 0:
            raise ValueError("invalid delivered-byte observation")
        self.observe_rate(frame_index, delivered_bytes * 8 / (duration_ms / 1000) / 1e6)

    def observe_rate(self, frame_index, measured):
        """Incorporate a throughput sample in Mbps, such as a capacity report."""
        if frame_index <= self._last_frame:
            raise ValueError("link observations must arrive in strict time order")
        if measured < 0:
            raise ValueError("invalid throughput observation")
        self._estimate = self._alpha * measured + (1 - self._alpha) * self._estimate
        if measured > 0:
            deviation = math.log(measured) - self._log_mean
            self._log_mean += self._alpha * deviation
            self._log_variance = (1 - self._alpha) * (
                self._log_variance + self._alpha * deviation**2
            )
        self._last_frame = frame_index

    def miss_probability(self, message_bytes, window_ms):
        """Probability that ``message_bytes`` need longer than ``window_ms``."""
        if message_bytes <= 0:
            return 0.0
        if window_ms <= 0:
            return 1.0
        required_mbps = message_bytes * 8 / (window_ms / 1000) / 1e6
        spread = math.sqrt(max(self._log_variance, 1e-8))
        z = (math.log(required_mbps) - self._log_mean) / spread
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def state(self, decision_frame, queue_ms=0.0):
        if decision_frame <= self._last_frame:
            raise ValueError("decision must follow every incorporated observation")
        if queue_ms < 0:
            raise ValueError("queue_ms cannot be negative")
        return LinkState(self._estimate, float(queue_ms), self._last_frame)


def select_deadline_budget(
    budgets,
    cells,
    channels,
    collaborators,
    state,
    deadline_ms,
    fixed_ms,
    index_bytes=4,
    value_bytes=2,
):
    """Largest configured budget predicted to arrive by the deadline.

    This conservative wire-size bound intentionally ignores compression gains;
    no ground truth or current/future measured delivery time is consulted.
    Returns zero when even the smallest message is predicted to be late.
    """
    if min(cells, channels, collaborators) < 1:
        raise ValueError("message dimensions must be positive")
    if deadline_ms <= 0 or fixed_ms < 0 or state.bandwidth_mbps <= 0:
        raise ValueError("invalid timing state")
    choices = sorted(set(float(value) for value in budgets))
    if not choices or choices[0] <= 0 or choices[-1] > 1:
        raise ValueError("budgets must be fractions in (0, 1]")
    available_ms = deadline_ms - fixed_ms - state.queue_ms
    if available_ms <= 0:
        return 0.0
    capacity_bytes = state.bandwidth_mbps * 1e6 * (available_ms / 1000) / 8
    feasible = []
    for budget in choices:
        selected = math.ceil(cells * budget)
        wire_bytes = collaborators * selected * (index_bytes + channels * value_bytes)
        if wire_bytes <= capacity_bytes:
            feasible.append(budget)
    return max(feasible, default=0.0)


def past_frame(history, current, age_frames):
    """Return a strictly earlier frame from the same session/slice.

    The function is suitable for constructing stale collaborator observations;
    it rejects accidental cross-split and cross-sequence lookup.
    """
    if age_frames < 1:
        raise ValueError("age_frames must be at least one")
    position = history.index(current)
    stale_position = position - age_frames
    if stale_position < 0:
        return None
    stale = history[stale_position]
    for key in ("split", "session", "slice"):
        if stale[key] != current[key]:
            return None
    if float(stale["timestamp"]) >= float(current["timestamp"]):
        raise ValueError("stale observation is not strictly earlier")
    return stale

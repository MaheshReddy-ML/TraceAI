"""Conservative descriptive trajectory analysis."""

from __future__ import annotations

from itertools import pairwise
from math import sqrt
from random import Random

from traceai.schemas import ChangePoint, Observation


def wilson_interval(failures: float, count: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% binomial interval; assumes independent cases."""
    proportion = failures / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    spread = (
        z * sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def clustered_bootstrap_interval(
    scores_by_case: dict[str, list[float]], seed: int, draws: int = 1000
) -> tuple[float, float]:
    """Resample case clusters, retaining all repeated-seed outcomes per case."""
    groups = list(scores_by_case.values())
    if not groups:
        raise ValueError("at least one case is required")
    rng = Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        estimates.append(sum(sum(group) for group in sample) / sum(len(group) for group in sample))
    estimates.sort()
    return estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]


class TrajectoryAnalyzer:
    def detect_changes(
        self, observations: list[Observation], checkpoint_order: list[str]
    ) -> list[ChangePoint]:
        result: list[ChangePoint] = []
        for probe in sorted({item.probe for item in observations}):
            by_checkpoint = {item.checkpoint: item for item in observations if item.probe == probe}
            ordered = [by_checkpoint[key] for key in checkpoint_order if key in by_checkpoint]
            for previous, current in pairwise(ordered):
                delta = current.score - previous.score
                if abs(delta) < 0.25:
                    continue
                intervals_separate = (
                    previous.ci_high < current.ci_low or current.ci_high < previous.ci_low
                )
                result.append(
                    ChangePoint(
                        probe=probe,
                        from_checkpoint=previous.checkpoint,
                        to_checkpoint=current.checkpoint,
                        delta=round(delta, 4),
                        classification="interval_separated_shift"
                        if intervals_separate
                        else "descriptive_shift",
                        method=(
                            "adjacent score delta >= 0.25; case-cluster bootstrap intervals shown for context"
                            if current.uncertainty_method.startswith("case_cluster")
                            else "adjacent score delta >= 0.25; Wilson intervals shown for context"
                        ),
                    )
                )
        return result

    def detect_trends(
        self, observations: list[Observation], checkpoint_order: list[str]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for probe in {item.probe for item in observations}:
            by_checkpoint = {item.checkpoint: item for item in observations if item.probe == probe}
            ordered = [by_checkpoint[key] for key in checkpoint_order if key in by_checkpoint]
            if len(ordered) > 1:
                result[probe] = round(ordered[-1].score - ordered[0].score, 4)
        return result

    def compare_checkpoints(
        self, observations: list[Observation], first: str, second: str
    ) -> dict[str, float]:
        first_scores = {item.probe: item.score for item in observations if item.checkpoint == first}
        second_scores = {
            item.probe: item.score for item in observations if item.checkpoint == second
        }
        return {
            probe: round(second_scores[probe] - score, 4)
            for probe, score in first_scores.items()
            if probe in second_scores
        }

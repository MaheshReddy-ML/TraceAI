"""Evidence-first report renderers."""

from __future__ import annotations

import json

from traceai.schemas import Report


def render_terminal(report: Report) -> str:
    lines = [
        f"TraceAI experiment {report.experiment_id}",
        f"Status: {report.status} | Target: {report.config.target.runtime}/{report.config.target.model}",
        "Observability: outputs only",
        "",
        "Observed failure rates (higher means more failures on this narrow probe):",
    ]
    lines.extend(
        f"  {item.checkpoint:>8}  {item.probe:<22} {item.score:.2f}  n={item.sample_count}  interval=[{item.ci_low:.2f}, {item.ci_high:.2f}]"
        for item in report.observations
    )
    lines.append("\nTrajectory observations:")
    lines.extend(
        f"  {item.probe}: {item.from_checkpoint} → {item.to_checkpoint}, Δ={item.delta:+.2f} ({item.classification})"
        for item in report.changes
    )
    if not report.changes:
        lines.append("  No adjacent score change reached the descriptive threshold.")
    lines.append(f"\nEvidence: {len(report.evidence)} raw prompt/output records saved.")
    lines.append("\nLimitations:")
    lines.extend(f"  - {item}" for item in report.limitations)
    return "\n".join(lines)


def render_markdown(report: Report) -> str:
    config = report.config
    lines = [
        f"# TraceAI experiment {report.experiment_id}",
        "",
        f"**Status:** {report.status}  ",
        f"**Target:** `{config.target.runtime}/{config.target.model}`  ",
        f"**Monitor:** `{config.monitor.runtime}/{config.monitor.model}`  "
        if config.monitor
        else "**Monitor:** none (deterministic rubrics)  ",
        f"**Seed:** {config.seed}  ",
        f"**TraceAI:** {report.environment.get('traceai_version')}  ",
        "**Observability:** outputs only  ",
        f"**Git revision:** {report.environment.get('git_revision') or 'unavailable'}",
        "",
        "## Observed measurements",
        "",
        "These are failure rates on the specified probe cases; higher values mean more observed failures.",
        "",
        "| Checkpoint | Probe | Failure rate | Cases | Approx. 95% interval |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {item.checkpoint} | {item.probe} | {item.score:.3f} | {item.sample_count} | {item.ci_low:.3f}–{item.ci_high:.3f} |"
        for item in report.observations
    )
    lines.extend(["", "## Trajectory", ""])
    if report.changes:
        lines.extend(
            f"- **OBSERVED:** {item.probe} changed by {item.delta:+.3f} between {item.from_checkpoint} and {item.to_checkpoint} ({item.classification})."
            for item in report.changes
        )
    else:
        lines.append("- **OBSERVED:** No adjacent score difference reached 0.25.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **INFERRED:** These measurements describe performance on fixed prompts only; no broader behavior is established.",
            "- **UNCERTAIN:** Internal intentions, causal training mechanisms, and out-of-distribution behavior are not determined.",
            "",
            "## Reproducibility",
            "",
            "```json",
            json.dumps(report.environment, indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence",
            "",
            "Each line is a complete prompt, response, rubric result, model, seed, and timestamp record.",
            "",
            "```jsonl",
        ]
    )
    lines.extend(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False) for item in report.evidence
    )
    lines.extend(["```", "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"

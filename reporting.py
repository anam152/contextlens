from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_markdown_report(
    source_name: str,
    target: str,
    task: str,
    analysis: dict[str, Any],
    narrative: dict[str, Any],
    modelling_result: dict[str, Any] | None,
) -> str:
    """Build a portable Markdown report from analysis and optional model results."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# ContextLens Analysis Report",
        "",
        f"- **Generated:** {generated}",
        f"- **Source:** `{source_name}`",
        f"- **Target:** `{target}`",
        f"- **Task:** {task.title()}",
        "",
        "## Context summary",
        "",
        narrative["summary"].replace("**", ""),
        "",
        "### Observations",
        "",
    ]
    lines.extend(f"- {item}" for item in narrative["observations"])

    lines.extend(["", "### Evaluation guidance", ""])
    lines.extend(f"- {item}" for item in narrative["evaluation_guidance"])

    lines.extend(
        [
            "",
            "## Dataset profile",
            "",
            f"- Rows: {analysis['shape']['rows']:,}",
            f"- Candidate features: {analysis['shape']['features']:,}",
            f"- Numeric features: {len(analysis['columns']['numeric'])}",
            f"- Categorical features: {len(analysis['columns']['categorical'])}",
            f"- Date-like features: {len(analysis['columns']['datetime_like'])}",
            f"- Missing cells: {analysis['missing']['total_cells']:,}",
            f"- Exact duplicate rows: {analysis['duplicates']:,}",
            "",
            "## Contextual issues",
            "",
        ]
    )

    if analysis["issues"]:
        for issue in analysis["issues"]:
            lines.extend(
                [
                    f"### {issue['title']} ({issue['severity']})",
                    "",
                    issue["detail"],
                    "",
                    f"**Suggested action:** {issue['recommendation']}",
                    "",
                ]
            )
    else:
        lines.extend(["No major structural issues were detected.", ""])

    if modelling_result:
        lines.extend(
            [
                "## Model comparison",
                "",
                f"- Training rows: {modelling_result['train_rows']:,}",
                f"- Test rows: {modelling_result['test_rows']:,}",
                f"- Primary metric: `{modelling_result['primary_metric']}`",
                f"- Best baseline: **{modelling_result['best_model']}**",
                "",
                "| Model | Metrics |",
                "|---|---|",
            ]
        )
        for row in modelling_result["leaderboard"]:
            model = row["model"]
            metric_text = ", ".join(
                f"{key}={value:.4f}"
                for key, value in row.items()
                if key != "model"
            )
            lines.append(f"| {model} | {metric_text} |")

        if modelling_result["warnings"]:
            lines.extend(["", "### Modelling notes", ""])
            lines.extend(f"- {item}" for item in modelling_result["warnings"])
    else:
        lines.extend(
            [
                "## Model comparison",
                "",
                "No model run was attached to this report.",
            ]
        )

    lines.extend(
        [
            "",
            "## Responsible interpretation",
            "",
            "This report is an exploratory baseline. Hold-out performance does not establish "
            "clinical, operational, or causal validity. Domain review, robust cross-validation, "
            "external validation, fairness analysis, and appropriate governance may be required.",
            "",
        ]
    )
    return "\n".join(lines)

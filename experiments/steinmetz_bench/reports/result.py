"""Result schema + per-experiment report writer for the Steinmetz benchmark suite.

A :class:`BenchResult` is the single record every experiment and backtest emits. It
carries the headline number (always computed by code from a real solve — never a
hand-written constant), the dataset it came from, an optional bootstrap CI, an
optional DC-vs-reference fidelity band, and free-form ``assumptions`` /
``sensitivities`` dicts. It round-trips losslessly to/from JSON and renders a
markdown stub that embeds the same JSON so the stub re-parses back to an identical
result (the whitepaper builder in Phase 5 reads these back).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from attrs import define, field

from experiments.steinmetz_bench.scoring.metrics import CIResult, FidelityBand


def _ci_from_dict(d: Optional[dict]) -> Optional[CIResult]:
    if d is None:
        return None
    return CIResult(
        lo=float(d["lo"]),
        mid=float(d["mid"]),
        hi=float(d["hi"]),
        confidence=float(d["confidence"]),
    )


def _fidelity_from_dict(d: Optional[dict]) -> Optional[FidelityBand]:
    if d is None:
        return None
    return FidelityBand(
        reference=str(d["reference"]),
        metric=str(d["metric"]),
        units=str(d["units"]),
        max_abs_gap=float(d["max_abs_gap"]),
        mean_abs_gap=float(d["mean_abs_gap"]),
        p90_abs_gap=float(d["p90_abs_gap"]),
        n=int(d["n"]),
    )


# Marker fence the markdown stub wraps its embedded JSON in, so it re-parses.
_JSON_BLOCK = re.compile(r"<!--\s*benchresult:json\s*-->\s*```json\n(.*?)\n```", re.DOTALL)


@define
class BenchResult:
    """One benchmark/backtest headline plus its provenance and uncertainty.

    ``headline_number`` and every value inside ``ci`` / ``fidelity_band`` must be
    produced by an actual zap solve; this schema is purely the container. ``ci`` is
    the bootstrap interval on the headline (if applicable), ``fidelity_band`` the
    DC-vs-reference gap, and ``assumptions`` / ``sensitivities`` are JSON-able dicts
    for the knobs and per-input derivatives the experiment chose to record.
    """

    experiment_id: str
    dataset: str
    headline_number: float
    units: str
    ci: Optional[CIResult] = None
    fidelity_band: Optional[FidelityBand] = None
    assumptions: dict = field(factory=dict)
    sensitivities: dict = field(factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "headline_number": self.headline_number,
            "units": self.units,
            "ci": self.ci.to_dict() if self.ci is not None else None,
            "fidelity_band": (
                self.fidelity_band.to_dict() if self.fidelity_band is not None else None
            ),
            "assumptions": dict(self.assumptions),
            "sensitivities": dict(self.sensitivities),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchResult":
        return cls(
            experiment_id=str(d["experiment_id"]),
            dataset=str(d["dataset"]),
            headline_number=float(d["headline_number"]),
            units=str(d["units"]),
            ci=_ci_from_dict(d.get("ci")),
            fidelity_band=_fidelity_from_dict(d.get("fidelity_band")),
            assumptions=dict(d.get("assumptions") or {}),
            sensitivities=dict(d.get("sensitivities") or {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "BenchResult":
        return cls.from_dict(json.loads(text))

    def write_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")
        return path

    def to_markdown(self) -> str:
        """Render a human-readable stub that embeds the canonical JSON.

        The embedded ``benchresult:json`` block is what :func:`parse_markdown`
        reads back, so the prose and the machine-readable record can never drift.
        """
        lines = [
            f"# {self.experiment_id}",
            "",
            f"- **Dataset:** {self.dataset}",
            f"- **Headline:** {self.headline_number} {self.units}".rstrip(),
        ]
        if self.ci is not None:
            lines.append(
                f"- **CI ({self.ci.confidence:.0%}):** "
                f"[{self.ci.lo}, {self.ci.hi}] (mid {self.ci.mid})"
            )
        else:
            lines.append("- **CI:** n/a")
        if self.fidelity_band is not None:
            fb = self.fidelity_band
            lines.append(
                f"- **Fidelity band ({fb.reference}/{fb.metric}):** "
                f"max {fb.max_abs_gap}, mean {fb.mean_abs_gap}, "
                f"p90 {fb.p90_abs_gap} {fb.units} (n={fb.n})".rstrip()
            )
        else:
            lines.append("- **Fidelity band:** n/a")
        if self.assumptions:
            lines.append("")
            lines.append("## Assumptions")
            for k, v in self.assumptions.items():
                lines.append(f"- {k}: {v}")
        if self.sensitivities:
            lines.append("")
            lines.append("## Sensitivities")
            for k, v in self.sensitivities.items():
                lines.append(f"- {k}: {v}")
        lines += [
            "",
            "<!-- benchresult:json -->",
            "```json",
            self.to_json(),
            "```",
            "",
        ]
        return "\n".join(lines)

    def write_markdown(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())
        return path


def parse_markdown(text: str) -> BenchResult:
    """Recover a :class:`BenchResult` from the JSON block embedded in its stub."""
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ValueError("no benchresult:json block found in markdown")
    return BenchResult.from_json(match.group(1))


def read_markdown(path) -> BenchResult:
    return parse_markdown(Path(path).read_text())


__all__ = ["BenchResult", "parse_markdown", "read_markdown"]

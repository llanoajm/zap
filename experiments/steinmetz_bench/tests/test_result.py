"""Tests for the BenchResult schema: JSON round-trip + markdown stub re-parse.

The CI and fidelity band embedded in the result are computed by the real scoring
functions on synthetic arrays (not hand-typed constants), so the round-trip is
exercised on values shaped like the ones experiments will actually emit.
"""

import numpy as np
import pytest

from experiments.steinmetz_bench.reports import (
    BenchResult,
    parse_markdown,
    read_markdown,
)
from experiments.steinmetz_bench.scoring import bootstrap_ci, fidelity_band


def _example_result(seed: int = 0) -> BenchResult:
    rng = np.random.default_rng(seed)
    savings = rng.normal(120.0, 8.0, size=256)
    dc = rng.normal(0.0, 1.0, size=64)
    ref = dc + rng.normal(0.0, 0.03, size=64)

    ci = bootstrap_ci(savings, seed=1)
    band = fidelity_band(dc, ref, reference="pypsa-dc", metric="lmp", units="$/MWh")
    return BenchResult(
        experiment_id="0.4-example",
        dataset="synthetic-5node",
        headline_number=float(np.mean(savings)),
        units="$/MWh",
        ci=ci,
        fidelity_band=band,
        assumptions={"hours": 24, "congested": True, "seed": seed},
        sensitivities={"d_cost_d_linecap": -3.5},
    )


def test_json_round_trip_is_lossless():
    result = _example_result()
    restored = BenchResult.from_json(result.to_json())

    assert restored.to_dict() == result.to_dict()
    assert restored.ci is not None
    assert restored.ci.as_tuple() == result.ci.as_tuple()
    assert restored.fidelity_band.to_dict() == result.fidelity_band.to_dict()
    assert restored.assumptions == result.assumptions
    assert restored.sensitivities == result.sensitivities


def test_round_trip_handles_optional_fields_absent():
    bare = BenchResult(
        experiment_id="0.4-bare",
        dataset="synthetic",
        headline_number=42.0,
        units="MW",
    )
    restored = BenchResult.from_json(bare.to_json())
    assert restored.ci is None
    assert restored.fidelity_band is None
    assert restored.assumptions == {}
    assert restored.to_dict() == bare.to_dict()


def test_markdown_stub_is_written_and_reparses(tmp_path):
    result = _example_result()
    md_path = result.write_markdown(tmp_path / "result.md")

    assert md_path.exists()
    text = md_path.read_text()
    assert "# 0.4-example" in text
    assert "$/MWh" in text

    reparsed = read_markdown(md_path)
    assert reparsed.to_dict() == result.to_dict()
    # The headline number survives the markdown trip exactly.
    assert reparsed.headline_number == pytest.approx(result.headline_number)


def test_parse_markdown_requires_json_block():
    with pytest.raises(ValueError):
        parse_markdown("# heading\n\nno embedded json here\n")


def test_write_json_creates_file_and_reads_back(tmp_path):
    result = _example_result(seed=3)
    json_path = result.write_json(tmp_path / "nested" / "result.json")

    assert json_path.exists()
    restored = BenchResult.from_json(json_path.read_text())
    assert restored.to_dict() == result.to_dict()

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import matplotlib
import numpy as np
import pytest

import moeabench as mb
from moeabench.core.run import Run
from moeabench.metrics import evaluator
from moeabench.metrics.GEN_mc_ordinal_hypervolume import GEN_mc_ordinal_hypervolume
from moeabench.metrics.GEN_ordinal_hypervolume import GEN_ordinal_hypervolume


matplotlib.use("Agg")

REFERENCE = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])


def _run(history):
    history = [np.asarray(front, dtype=float) for front in history]
    empty = [np.empty((len(front), 0)) for front in history]
    payload = (history, empty, history[-1], history, empty, empty, empty)
    return Run(payload)


def _ohv(front, **kwargs):
    return evaluator.ordinal_hypervolume(
        front, ref=REFERENCE, mode="exact", progress=False, **kwargs
    )


def test_public_signature_and_exact_alias():
    expected = [
        "exp", "ref", "mode", "n_samples", "mc_seed", "gens", "progress", "scale"
    ]
    assert list(inspect.signature(evaluator.ordinal_hypervolume).parameters) == expected
    assert inspect.signature(evaluator.ordinal_hypervolume).parameters["scale"].kind \
        is inspect.Parameter.KEYWORD_ONLY
    assert mb.metrics.ohv is mb.metrics.ordinal_hypervolume


@pytest.mark.parametrize(
    ("front", "expected"),
    [
        (np.array([[0.0, 2.0], [2.0, 0.0]]), 5.0),
        (REFERENCE, 6.0),
    ],
)
def test_known_two_dimensional_lattice(front, expected):
    result = _ohv(front)
    assert result.values.shape == (1, 1)
    assert result.values[0, 0] == expected
    assert result.metric_name == "Ordinal Hypervolume (Raw)"
    assert np.array_equal(result.diagnostics["ordinal_level_counts"], [3, 3])
    assert np.array_equal(result.diagnostics["ordinal_reference_point"], [3, 3])
    assert result.diagnostics["ordinal_box_volume"] == 9.0


def test_distance_and_strictly_monotone_transform_invariance():
    reference_a = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], dtype=float)
    front_a = np.array([[0, 3], [2, 1]], dtype=float)
    spacing = np.array([0.0, 0.0001, 1000.0, 1_000_000.0])
    reference_b = np.column_stack((spacing, spacing[::-1]))
    front_b = reference_b[[0, 2]]

    a = evaluator.ordinal_hypervolume(
        front_a, ref=reference_a, mode="exact", progress=False
    )
    b = evaluator.ordinal_hypervolume(
        front_b, ref=reference_b, mode="exact", progress=False
    )
    assert np.array_equal(a.values, b.values)

    transform = lambda x: np.column_stack((10 * x[:, 0] + 7, x[:, 1] ** 3))
    transformed = evaluator.ordinal_hypervolume(
        transform(front_a), ref=transform(reference_a), mode="exact", progress=False
    )
    assert np.array_equal(a.values, transformed.values)


def test_ties_do_not_add_levels_or_change_ohv():
    duplicated = np.vstack([REFERENCE, REFERENCE[[1, 1]]])
    plain = _ohv(REFERENCE)
    tied = evaluator.ordinal_hypervolume(
        REFERENCE, ref=duplicated, mode="exact", progress=False
    )
    assert np.array_equal(plain.values, tied.values)
    assert np.array_equal(tied.diagnostics["ordinal_level_counts"], [3, 3])
    assert tied.diagnostics["reference_points"] == 5


def test_fixed_scale_and_gens_prefix_use_final_reference_front():
    run = _run([
        [[0.4, 1.6], [1.6, 0.4]],
        [[0.2, 1.8], [1.0, 1.0], [1.8, 0.2]],
        REFERENCE,
    ])
    full = evaluator.ordinal_hypervolume(run, mode="exact", progress=False)
    prefix = evaluator.ordinal_hypervolume(run, mode="exact", gens=2, progress=False)
    assert np.array_equal(prefix.values, full.values[:2])
    assert np.array_equal(prefix.diagnostics["ordinal_level_counts"], [3, 3])
    assert all(np.array_equal(a, b) for a, b in zip(
        prefix.diagnostics["ordinal_levels"], full.diagnostics["ordinal_levels"]
    ))


def test_gens_zero_returns_empty_matrix_and_nan_fraction():
    run = _run([REFERENCE])
    result = evaluator.ordinal_hypervolume(
        run, mode="exact", gens=0, progress=False
    )
    assert result.values.shape == (0, 1)
    assert np.isnan(result.diagnostics["raw_ohv_fraction_of_ordinal_box"]).all()


def test_default_scale_is_explicit_raw_without_numerical_change():
    default = _ohv(REFERENCE)
    explicit = _ohv(REFERENCE, scale="raw")
    assert np.array_equal(default.values, explicit.values)
    assert default.metric_name == "Ordinal Hypervolume (Raw)"
    assert default.diagnostics["ohv_scale"] == "raw"
    assert default.diagnostics["ohv_scale_denominator"] is None


@pytest.mark.parametrize("scale", ["RAW", "Rel"])
def test_scale_is_case_insensitive(scale):
    result = _ohv(REFERENCE, scale=scale)
    assert result.diagnostics["ohv_scale"] == scale.lower()


@pytest.mark.parametrize("scale", ["abs", "foo", None])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="Unknown scale parameter"):
        _ohv(REFERENCE, scale=scale)


def test_relative_single_run_uses_one_fixed_final_denominator():
    run = _run([REFERENCE[[0, 2]], REFERENCE])
    raw = evaluator.ordinal_hypervolume(
        run, mode="exact", progress=False, scale="raw"
    )
    relative = evaluator.ordinal_hypervolume(
        run, mode="exact", progress=False, scale="rel"
    )
    denominator = raw.values[-1, 0]
    np.testing.assert_allclose(relative.values[:, 0], raw.values[:, 0] / denominator)
    assert relative.values[-1, 0] == pytest.approx(1.0)
    assert relative.diagnostics["ohv_scale_denominator"] == pytest.approx(denominator)


def test_relative_multi_run_self_reference_uses_best_final_run():
    first = _run([REFERENCE[[0, 2]]])
    second = _run([REFERENCE])
    raw = evaluator.ordinal_hypervolume(
        [first, second], mode="exact", progress=False, scale="raw"
    )
    relative = evaluator.ordinal_hypervolume(
        [first, second], mode="exact", progress=False, scale="rel"
    )
    denominator = np.max(raw.values[-1])
    np.testing.assert_allclose(relative.values, raw.values / denominator)
    assert np.max(relative.values[-1]) == pytest.approx(1.0)


def test_external_denominator_is_best_individual_front_not_pooled_union():
    left = _run([[[0.0, 1.0]]])
    right = _run([[[1.0, 0.0]]])
    pooled = _run([[[0.0, 1.0], [1.0, 0.0]]])
    references = [left, right]

    left_raw = evaluator.ordinal_hypervolume(
        left, ref=references, mode="exact", progress=False
    ).values[-1, 0]
    right_raw = evaluator.ordinal_hypervolume(
        right, ref=references, mode="exact", progress=False
    ).values[-1, 0]
    union_raw = evaluator.ordinal_hypervolume(
        pooled, ref=references, mode="exact", progress=False
    ).values[-1, 0]
    result = evaluator.ordinal_hypervolume(
        pooled, ref=references, mode="exact", progress=False, scale="rel"
    )

    assert union_raw > max(left_raw, right_raw)
    assert result.diagnostics["ohv_scale_denominator"] == max(left_raw, right_raw)
    assert result.diagnostics["ohv_scale_denominator"] != union_raw
    assert result.values[-1, 0] > 1.0


def test_external_experiment_runs_are_individual_denominator_candidates():
    weak = _run([REFERENCE[[0, 2]]])
    strong = _run([REFERENCE])
    reference_exp = mb.experiment()
    reference_exp._runs = [weak, strong]
    result = evaluator.ordinal_hypervolume(
        weak, ref=reference_exp, mode="exact", progress=False, scale="rel"
    )
    strong_raw = evaluator.ordinal_hypervolume(
        strong, ref=reference_exp, mode="exact", progress=False
    ).values[-1, 0]
    assert result.diagnostics["ohv_scale_denominator"] == pytest.approx(strong_raw)


def test_relative_gens_zero_keeps_reference_denominator():
    run = _run([REFERENCE])
    result = evaluator.ordinal_hypervolume(
        run, mode="exact", gens=0, progress=False, scale="rel"
    )
    assert result.values.shape == (0, 1)
    assert result.diagnostics["ohv_scale_denominator"] == pytest.approx(6.0)
    report = result.report(show=False, markdown=False)
    assert "Scale: Relative" in report
    assert "Scale denominator: 6.0000" in report


def test_nonpositive_relative_denominator_is_rejected(monkeypatch):
    monkeypatch.setattr(
        evaluator, "_evaluate_ordinal_history", lambda *args, **kwargs: np.array([0.0])
    )
    with pytest.raises(ValueError, match="positive best-final-reference OHV"):
        _ohv(REFERENCE, scale="rel")


def test_shared_multi_run_and_external_reference_contexts_are_identical():
    first = _run([REFERENCE[[0, 2]]])
    second = _run([REFERENCE])
    shared = evaluator.ordinal_hypervolume(
        [first, second], mode="exact", progress=False
    )
    assert shared.values.shape == (1, 2)

    left = evaluator.ordinal_hypervolume(
        first, ref=[first, second], mode="exact", progress=False
    )
    right = evaluator.ordinal_hypervolume(
        second, ref=[first, second], mode="exact", progress=False
    )
    assert np.array_equal(
        left.diagnostics["ordinal_level_counts"],
        right.diagnostics["ordinal_level_counts"],
    )
    assert all(np.array_equal(a, b) for a, b in zip(
        left.diagnostics["ordinal_levels"], right.diagnostics["ordinal_levels"]
    ))

    left_relative = evaluator.ordinal_hypervolume(
        first, ref=[first, second], mode="exact", progress=False, scale="rel"
    )
    right_relative = evaluator.ordinal_hypervolume(
        second, ref=[first, second], mode="exact", progress=False, scale="rel"
    )
    assert left_relative.diagnostics["ohv_scale_denominator"] == pytest.approx(
        right_relative.diagnostics["ohv_scale_denominator"]
    )


def test_external_reference_isolation_and_outside_reference_behavior():
    outside = np.array([[-100.0, 2.0], [100.0, 0.0]])
    result = _ohv(outside)
    assert all(np.array_equal(axis, expected) for axis, expected in zip(
        result.diagnostics["ordinal_levels"], ([0, 1, 2], [0, 1, 2])
    ))
    transformed = evaluator._ordinal_transform(outside, result.diagnostics["ordinal_levels"])
    assert transformed[0, 0] == 0
    assert transformed[1, 0] == 3


def test_zero_span_objective_is_valid():
    reference = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    result = evaluator.ordinal_hypervolume(
        reference, ref=reference, mode="exact", progress=False
    )
    assert np.array_equal(result.diagnostics["ordinal_level_counts"], [1, 3])
    assert np.isfinite(result.values).all()


@pytest.mark.parametrize("where", ["reference", "evaluated"])
def test_nonfinite_values_are_rejected_with_context(where):
    bad = REFERENCE.copy()
    bad[0, 0] = np.nan
    kwargs = {"ref": bad, "exp": REFERENCE} if where == "reference" else {
        "ref": REFERENCE, "exp": bad
    }
    with pytest.raises(ValueError, match=where + ".*NaN or infinite"):
        evaluator.ordinal_hypervolume(
            mode="exact", progress=False, **kwargs
        )


@pytest.mark.parametrize(
    ("objectives", "mode", "expected"),
    [(8, "auto", "exact"), (9, "auto", "monte_carlo"), (2, "fast", "monte_carlo")],
)
def test_backend_selection(objectives, mode, expected):
    reference = np.vstack([np.zeros(objectives), np.ones(objectives)])
    result = evaluator.ordinal_hypervolume(
        reference, ref=reference, mode=mode, n_samples=100, mc_seed=5, progress=False
    )
    assert result.diagnostics["ohv_backend"] == expected
    assert result.diagnostics["ohv_mode_requested"] == mode


def test_exact_ignores_sampling_arguments_but_monte_carlo_validates_them():
    exact = evaluator.ordinal_hypervolume(
        REFERENCE, mode="exact", n_samples="ignored", mc_seed="ignored", progress=False
    )
    assert exact.diagnostics["ohv_n_samples"] is None
    assert exact.diagnostics["ohv_mc_seed"] is None
    with pytest.raises(ValueError, match="n_samples"):
        evaluator.ordinal_hypervolume(
            REFERENCE, mode="fast", n_samples=0, progress=False
        )
    with pytest.raises(ValueError, match="mc_seed"):
        evaluator.ordinal_hypervolume(
            REFERENCE, mode="fast", mc_seed=1.5, progress=False
        )


def test_monte_carlo_is_reproducible():
    first = evaluator.ordinal_hypervolume(
        REFERENCE, mode="fast", n_samples=2000, mc_seed=23, progress=False
    )
    second = evaluator.ordinal_hypervolume(
        REFERENCE, mode="fast", n_samples=2000, mc_seed=23, progress=False
    )
    assert np.array_equal(first.values, second.values)


def test_relative_monte_carlo_denominator_is_reproducible():
    first = evaluator.ordinal_hypervolume(
        REFERENCE, mode="fast", scale="rel", n_samples=2000,
        mc_seed=23, progress=False
    )
    second = evaluator.ordinal_hypervolume(
        REFERENCE, mode="fast", scale="rel", n_samples=2000,
        mc_seed=23, progress=False
    )
    assert first.diagnostics["ohv_scale_denominator"] == pytest.approx(
        second.diagnostics["ohv_scale_denominator"]
    )
    np.testing.assert_array_equal(first.values, second.values)


def test_empty_generation_is_zero_for_both_evaluators():
    empty = np.empty((0, 2))
    assert GEN_ordinal_hypervolume([empty], [3, 3]).evaluate()[0] == 0.0
    assert GEN_mc_ordinal_hypervolume(
        [empty], [3, 3], n_samples=10, seed=1
    ).evaluate()[0] == 0.0


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_progress_advances_once_per_generation_and_closes(monkeypatch, mode):
    class RecordingProgress:
        def __init__(self):
            self.current_val = 0
            self.updates = []
            self.closed = False

        def update_to(self, value):
            self.current_val = value
            self.updates.append(value)

        def close(self):
            self.closed = True

    progress = RecordingProgress()
    monkeypatch.setattr(evaluator, "get_progress_bar", lambda **kwargs: progress)
    run = _run([REFERENCE[[0]], REFERENCE])
    evaluator.ordinal_hypervolume(
        run, mode=mode, n_samples=100, progress=True
    )
    assert progress.updates == [1, 2]
    assert progress.closed


def test_relative_denominator_does_not_advance_progress(monkeypatch):
    class RecordingProgress:
        current_val = 0
        updates = []

        def update_to(self, value):
            self.current_val = value
            self.updates.append(value)

        def close(self):
            pass

    progress = RecordingProgress()
    monkeypatch.setattr(evaluator, "get_progress_bar", lambda **kwargs: progress)
    run = _run([REFERENCE[[0, 2]], REFERENCE])
    evaluator.ordinal_hypervolume(
        run, mode="exact", scale="rel", progress=True
    )
    assert progress.updates == [1, 2]


def test_raw_geometry_diagnostics_are_scale_independent():
    run = _run([REFERENCE[[0, 2]], REFERENCE])
    raw = evaluator.ordinal_hypervolume(
        run, mode="exact", scale="raw", progress=False
    )
    relative = evaluator.ordinal_hypervolume(
        run, mode="exact", scale="rel", progress=False
    )
    for key in (
        "ordinal_level_counts",
        "ordinal_reference_point",
        "raw_ohv_fraction_of_ordinal_box",
    ):
        np.testing.assert_array_equal(raw.diagnostics[key], relative.diagnostics[key])
    for left, right in zip(
        raw.diagnostics["ordinal_levels"], relative.diagnostics["ordinal_levels"]
    ):
        np.testing.assert_array_equal(left, right)
    for key in ("ordinal_box_volume", "reference_points", "reference_names"):
        assert raw.diagnostics[key] == relative.diagnostics[key]


def test_report_is_ordinal_and_excludes_conventional_hv_geometry():
    report = _ohv(REFERENCE).report(show=False, markdown=False)
    assert "Ordinal Hypervolume (Raw)" in report
    assert "Ordinal Reference" in report
    assert "Ordinal levels" in report
    assert "Scale" in report
    assert "Raw OHV/OBox" in report
    for forbidden in ("nbox", "bbox", "HV/BBox", "Reference expansion"):
        assert forbidden not in report


def test_relative_report_explains_scale_without_claiming_a_ceiling():
    report = _ohv(REFERENCE, scale="rel").report(show=False, markdown=False)
    assert "Ordinal Hypervolume (Relative)" in report
    assert "Scale" in report and "Relative" in report
    assert "Scale denominator" in report
    assert "Raw OHV/OBox" in report
    assert "historical values may exceed 1.0" in report
    assert "1.0 ceiling" not in report


def test_public_history_view_accepts_ohv():
    _, ax = mb.view.history(
        _run([REFERENCE[[0, 2]], REFERENCE]),
        metric=mb.metrics.ohv,
        mode="static",
        progress=False,
    )
    assert ax.get_ylabel() == "Ordinal Hypervolume (Raw)"


def test_public_history_view_accepts_relative_ohv_with_shared_context():
    first = _run([REFERENCE[[0, 2]], REFERENCE])
    second = _run([REFERENCE])
    _, ax = mb.view.history(
        first,
        second,
        metric=mb.metrics.ohv,
        scale="rel",
        mode="static",
        progress=False,
    )
    assert ax.get_ylabel() == "Ordinal Hypervolume (Relative)"

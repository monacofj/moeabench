# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np

from moeabench.plotting.scatter2d import Scatter2D
from moeabench.plotting.scatter3d import Scatter3D

def detect_out_of_scale(pts):
    """
    Simulation of the logic used in generate_visual_report.py
    Input: pts (N x M) normalized coordinates.
    Output: True if any point is significantly > 1.0 (with tolerance).
    """
    # Logic mirror: if max(pts) > 1.2 -> Flagged
    return np.max(pts) > 1.2

def test_visual_safety_scale_audit():
    """ Verify the logic that flags non-converged populations. """
    rng = np.random.default_rng(7)
    
    # 1. Safe Population (Inside Unit Cube)
    pop_safe = rng.random((100, 3))
    assert not detect_out_of_scale(pop_safe), "Safe population flagged as Out-of-Scale"
    
    # 2. Borderline (1.1 is allowed as ref point boundary)
    pop_border = np.array([[1.1, 1.1, 1.1]])
    assert not detect_out_of_scale(pop_border), "Borderline (1.1) flagged incorrectly"
    
    # 3. Unsafe (Exploded)
    pop_exploded = np.array([[0.5, 0.5, 50.0]])
    assert detect_out_of_scale(pop_exploded), "Exploded population NOT flagged!"

def test_visual_jitter_mechanism():
    """ Verify that Jitter adds variance but preserves mean approximately. """
    rng = np.random.default_rng(7)
    original = np.ones((100, 3)) * 0.5
    
    # Apply Jitter (epsilon = 0.003)
    epsilon = 0.003
    jittered = original + rng.normal(0, epsilon, original.shape)
    
    # Check 1: Shapes match
    assert jittered.shape == original.shape
    
    # Check 2: Values are different (Entropy added)
    assert not np.array_equal(jittered, original)
    
    # Check 3: Deviation is controlled (3-sigma < 0.01)
    diffs = np.abs(jittered - original)
    assert np.all(diffs < 0.015), "Jitter magnitude too high!"
    
    # Check 4: Mean is preserved
    assert np.allclose(np.mean(jittered), 0.5, atol=1e-3)


class _PrettyRecorder:
    def __init__(self):
        self.parts = []

    def text(self, value):
        self.parts.append(value)


def test_scatter2d_pretty_repr_is_silent():
    plot = Scatter2D(["A"], [np.array([[0.1, 0.2]])], [0, 1], mode="static")
    recorder = _PrettyRecorder()

    plot._repr_pretty_(recorder, cycle=False)

    assert recorder.parts == [""]


def test_scatter3d_pretty_repr_is_silent():
    plot = Scatter3D(["A"], [np.array([[0.1, 0.2, 0.3]])], [0, 1, 2], mode="static")
    recorder = _PrettyRecorder()

    plot._repr_pretty_(recorder, cycle=False)

    assert recorder.parts == [""]

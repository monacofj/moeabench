# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostic only: evaluate frozen DPF1 inputs independent of evolutionary search.

Inputs (generation-1 decision vectors) and expected objective values came
from the passing CI run 44307380840. Both sets are lexicographically sorted
independently for comparison. They are observation records, NOT accepted
scientific baselines. This script never alters production code or regression
tests.
"""
import base64
import hashlib
import json
import math

import numpy as np

from moeabench.mops import DPF1

_X_DATA = "AO0IcUySdD+zzeBGhx3kP0oyVuuOEtI/6GWc3zFusT8vUpEfEL3jP7QRhGHckcY/QLjndb3qdj/9q4RrvxztP1j/pIdSY74/WVfVbF7u5z+HcsNPj2jSP/4oYMuVJsU/ALMv1h3CeD+1MYJkyBnkPxjzZ5mXxco/dPocCJvluj/eMS1n+3LRP+PHWn4OUdc/gF8MuvPjgz/4DsWg88rKPxW9gXBsUOk/3kvLJMaa5D9K4TYA9ATZP6NMY6/mGtU/gF8MuvPjgz/MRaUKsOLKPzZO2qkL1+s/neQ792sh7z9sQWhkU0bcP9RDZYVuPdg/YJt6eHoKnD+sAn/0OV7tPxg3zUc7HL8/WVfVbF7u5z89QGlETLDsPxiAYEO5fsU/mELSEwG0tD+Q5RpamgDsP5JdLI+6b9s/OZOon+LJ4z80DSur6wnUPzRAVz1B6MY/4H/ZxlgZzT8weXLFxEXUP52IRpAEhOk/3ZDU1eCj5T/eQj9a8AfZPyK59cnSTNU/cEsiFbpV0D/8w8cxTwvTP3RMP1Y83NE/RLnqblSt0D+moDXhkOXeP2CeSAohIss/YxAiULL40D/bNVRsqMTSP3RMP1Y83NE/Ipryyar/yj+moDXhkOXePwAnEaEzTs4/cm/eyAoN0T+I27Ed1xrqP3jUP9zevcg/zB8DVHGSwD/o2hRWV3e3P2LrYxt4J+M/obiN1ETw3z84Dx1SWh/qP0D/DzAIXMQ/IsjvgGT5xj+EC9ZxZHK5P9Jb3QbzAeM/zIrxSdzT4j8/cQJtNXTIPxNV5643h+U/aXKEiZlV7z9UY9r8g8bPP2DPxZnvhu0/obst5B7H4z/PsJg12C7hPxx1I/YKTuQ/9KAMC+RRxj9kIvuf2sPPPxppUuQR6uU/GHD3WwJa5T+wcCFhxIy4P0gLEeEZR9w/KFHCIgte7D82K48AilHmP/6ajXDu5NQ/5MVfWO/o5T/JhSZaZHvOP0gLEeEZR9w/nERdErNW7D82K48AilHmP/MRva5KudQ/rWapSFoE6D/h24tgTI7jPxjzZ5mXxco/S4S7999Q6D8s2io1xefPP+gmWG4H6LU/SBwyOZy66D/fd+pCC5LjP0oyVuuOEtI/zhu5sW8/6T8pVZv14HTjP1L3Kg3sGbY/IIMyCdP16D+sh7x8DhXKP9ynjcNmquE/Et97/9R81z/ko2fgpjvgP+DkA3wLV9U/DuWKZtVs6z/NkS0Jvc/jPxjzZ5mXxco/S4S7999Q6D//x52SGnXrP+ozbV5OmLY/MrCqqcUV7D9A1QAhL2C2P8Y73uI++94/asi2YW/M3j/tlbZd0wrpP9pqyAqs3e4/p21lP/FH7z9x0i0rFrzrP4DE+T14m6c/Gr096amU0j/c9B03tZjrPxF7ojwlOuM/p21lP/FH7z9x0i0rFrzrP4DE+T14m6c/XOU6Zs+h0j9rZWiqcTzrPxPySe9x5ck/0m5u+qB07z9x0i0rFrzrP4DE+T14m6c/Gr096amU0j/c9B03tZjrPxF7ojwlOuM/"
_F_DATA = "KZ8oLFcZ9D8IGJM9pRtvQGJJ3AbBp2pAuTTMK+XT9T/9zHlxrwpsQJeM6CqwCWhAAZxUoF099z9OcsqBciJwQH/FuEYXqGtAcs/4yVCf/j+9R2+2wGRoQG+pA0cQ8WRAoJuNRCtTBkBcQFiLq8hxQE8fmhbaiG5AyrjVh6/RHUAOJCtgdoxwQFQNZEYYn2xAJrE7mJ//MEBW559yLCZoQHDpi9WiYGVAhC3C9e53QEC5NxdcMqxWQHB4HoxONFZAZRqTikOQQkCSKFQI8ItfQIeouSerJl5AuSZ0/NPkRkDjhN3UeLNgQGS1oNg2PWBAUZi4uWBZR0Ahsws/4SxgQCRYCkwrqF9ALdLzAgLMUUCYhsDFae40QDLb/MNlH0VAWXH5R42NVkBZP/67xKNWQNZnYsu6EltA6mWfMeg6X0DXHQusYyZPQPV0IOfa/1dAR3Eh6ykyZEC/X0OBckJcQOb82cNq/WJAKX1dqnH2ZUBTBFUU/H04QOJYY+QsQlRAcxRKJExfZkBi3i34JZtUQAjIRfI4dmBAy7aCN990Z0BpCCvL0i9PQNwO0Yf2YF1A/ehs57yfaUANvZsgZOhMQFzfDrwI411AgJY8u4nIakA4oD5cKo1gQAwafMhKUGdAB+I7LFX4a0ADZaomv6lCQCibzyd7HFtAvW1onCIDckBJ7ZqQ+34aQH/UQnyODlpA1fjY1t0cckAvJ3j0LBAUQOxdcOO92VlAZkG7L7MKc0D2PKvDrwIcQFy+TpvVi1tA"
_X_SHA = "9fa990368319ac8aa104fbc047ef16cfe2f885632c348423b4bcffbdad7c2157"
_F_SHA = "9f9c75a7b40d7ad1fcab553ebafd0b2ef51768584fccd75ee56a7bec5eff1027"


def _load(raw, sha, shape):
    decoded = base64.b64decode(raw, validate=True)
    assert hashlib.sha256(decoded).hexdigest() == sha
    arr = np.frombuffer(decoded, dtype="<f8").reshape(shape).copy()
    assert np.isfinite(arr).all()
    return arr


def _sig(a):
    array = np.asarray(a, dtype="<f8")
    array = array[np.lexsort(array.T[::-1])]
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    x = _load(_X_DATA, _X_SHA, (24, 6))
    expected = _load(_F_DATA, _F_SHA, (24, 3))
    mop = DPF1(M=3, D=2)
    actual = np.asarray(mop.evaluation(x)["F"], dtype=np.float64)
    actual = actual[np.lexsort(actual.T[::-1])]
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected)

    xm = x[:, 1:]
    cvec = np.cos(20 * np.pi * (xm - 0.5))
    cscalar = np.array([[math.cos(float(t)) for t in row]
                        for row in (20 * np.pi * (xm - 0.5))])
    g = 100 * (xm.shape[1] + np.sum((xm - 0.5)**2 - cvec, axis=1))
    gscalar = 100 * (xm.shape[1] + np.sum((xm - 0.5)**2 - cscalar, axis=1))
    base = np.column_stack((0.5 * (1 + g) * x[:, 0],
                             0.5 * (1 + g) * (1 - x[:, 0])))
    w = mop._chaos_pool[:2]
    dot = np.dot(base, w)
    manual = base[:, 0] * w[0] + base[:, 1] * w[1]
    print(json.dumps({
        "x_sha": _sig(x),
        "objective_sha": _sig(actual),
        "pass_reference_objectives_sha": _F_SHA,
        "max_abs_delta_from_pass": float(diff.max()),
        "num_values_different_from_pass": int(np.count_nonzero(diff)),
        "num_values_different_gt_1e_12": int(np.count_nonzero(diff > 1e-12)),
        "cos_vs_scalar_max_abs": float(np.max(np.abs(cvec - cscalar))),
        "g_vs_scalar_cos_max_abs": float(np.max(np.abs(g - gscalar))),
        "projection_dot_vs_manual_max_abs": float(np.max(np.abs(dot - manual))),
        "g_signature": _sig(g.reshape(-1, 1)),
        "dot_signature": _sig(dot.reshape(-1, 1)),
        "manual_signature": _sig(manual.reshape(-1, 1)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Display helpers for robust plotting across interactive and headless backends."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


def is_interactive_backend() -> bool:
    """Return True when matplotlib backend supports interactive windows."""
    backend = str(matplotlib.get_backend()).lower()
    backend_name = backend.split("://")[-1]
    try:
        from matplotlib.backends import backend_registry, BackendFilter
        interactive = {b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
        non_interactive = {b.lower() for b in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)}
    except Exception:
        # Backward compatibility with older matplotlib.
        from matplotlib import rcsetup
        interactive = {b.lower() for b in rcsetup.interactive_bk}
        non_interactive = {b.lower() for b in rcsetup.non_interactive_bk}

    # Jupyter inline backends should still call plt.show() for proper display.
    if "matplotlib_inline" in backend:
        return True

    if backend in interactive or backend_name in interactive:
        return True
    if backend in non_interactive or backend_name in non_interactive:
        return False

    # Conservative fallback: keep standard Matplotlib behavior.
    return True


def show_matplotlib(fig=None) -> bool:
    """
    Show a matplotlib figure safely.

    Returns True when an interactive window show was attempted.
    Returns False on headless backends after forcing a canvas render.
    """
    if is_interactive_backend():
        plt.show()
        return True

    target = fig if fig is not None else plt.gcf()
    try:
        target.canvas.draw()
    except Exception:
        pass
    return False

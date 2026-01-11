# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from tqdm.auto import tqdm

_active_pbar = None

def set_active_pbar(pbar):
    global _active_pbar
    _active_pbar = pbar

def get_active_pbar():
    return _active_pbar

class MoeaProgress:
    """
    Manages progress bars for MoeaBench experiments.
    Supports both discrete steps (generations) and fractional progress (0-1).
    """
    def __init__(self, total=None, desc="MoeaBench", leave=True, position=0):
        self.total = total
        self.current_val = 0
        
        # tqdm.auto automatically detects if we are in Jupyter or Terminal
        self.pbar = tqdm(total=100 if total is None else total, 
                         desc=desc, 
                         leave=leave, 
                         position=position,
                         unit="gen" if total is not None else "it",
                         ascii=" -")
        
        self.is_fractional = total is None

    def update_to(self, value):
        """
        Updates the progress bar to a specific value.
        If total was None, value is expected to be a fraction 0.0-1.0.
        If total was set, value is expected to be the current step (integer).
        """
        if self.is_fractional:
            # Map 0.0-1.0 to 0-100
            new_val = int(min(1.0, max(0.0, value)) * 100)
            diff = new_val - self.current_val
            if diff > 0:
                self.pbar.update(diff)
                self.current_val = new_val
        else:
            diff = int(value) - self.current_val
            if diff > 0:
                self.pbar.update(diff)
                self.current_val = int(value)

    def close(self):
        # Ensure it hits 100% on close if it was close
        if self.is_fractional and self.current_val < 100:
             self.pbar.update(100 - self.current_val)
        self.pbar.close()

    def set_description(self, desc):
        self.pbar.set_description(desc)

def get_progress_bar(total=None, desc="Optimizing", position=0, leave=True):
    """Factory to create a progress bar."""
    return MoeaProgress(total=total, desc=desc, position=position, leave=leave)

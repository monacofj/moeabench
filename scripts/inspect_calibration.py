# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

def inspect():
    print("CRITICAL: This script depends on the deprecated 'MoeaBench.clinic' module.")
    print("Please use 'tests/calibration/generate_visual_report.py' for the new Clinical Metrology (Fair/Q-Score).")
    sys.exit(1)

if __name__ == "__main__":
    inspect()

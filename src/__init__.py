"""FinSight expense prediction package."""

import os
import sys
import warnings

# Silence Windows Python 3.13 joblib loky wmic warning and numpy 2.5 pickle deprecation warning
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 4) - 1))

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

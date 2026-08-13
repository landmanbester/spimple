"""
Radio astronomy image post-processing tools

author - Landman Bester
email  - lbester@sarao.ac.za
date   - 16/06/2022
"""

import os

# Ray >= 2.43 auto-injects a runtime_env when it detects it is running under
# `uv run`, handing itself the project directory as a URI. ray.init then dies
# with "<project dir> is not a valid URI", which takes `spimple mosaic` out
# entirely. setdefault so an explicit value in the environment still wins.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

__version__ = "0.0.6"

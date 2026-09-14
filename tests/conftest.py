"""Test-wide setup.

Kept to the one thing every test in the suite depends on and none of them
should have to think about: the console Kong writes through must not colour its
output while the tests are reading it.
"""

from __future__ import annotations

import os

# rich decides whether to emit escape sequences when the Console is built, and
# Kong builds its Console at import time — so this has to happen before any test
# module imports kong.__main__, which is exactly what a conftest is for.
#
# Without it a developer whose terminal exports FORCE_COLOR (or any of rich's
# other colour-forcing variables) gets escape codes wrapped around every number
# and word in the captured output, and assertions on what a command printed fail
# on machines where nothing is wrong. TTY_COMPATIBLE is rich's own way of saying
# "this is not a terminal", and it outranks the rest.
os.environ["TTY_COMPATIBLE"] = "0"
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)

"""Browser front-end for Kong.

`kong.webui.server` serves a small single-page interface over loopback and
drives the same `AnalysisController` the desktop window does. Importing it
costs nothing but the standard library, so unlike `kong.gui.app` it works on
a machine with no Tk.
"""

from kong.webui.server import KongSession, launch

__all__ = ["KongSession", "launch"]

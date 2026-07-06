#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "May 5, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters

class FrictionalZones:
    """
    Assigns depth-dependent rate-and-state parameters a(y) and b(y).

    The heterogeneous stratigraphy matches the Groningen / Slochteren
    reservoir setting.  You can subclass or replace `build()` to supply
    any profile you like.
    """

    # Layer depths relative to surface (positive downward convention)
    # These are absolute depths [m from surface].  ysize is subtracted to
    # convert to the model's coordinate system where y=0 is the top.
    LAYERS = {
        "Rocksalt":   {"top": 1, "bot": 2, "a": 0.012,  "b": 0.0135}, # Zechstein rocksalt (halite)
    }

    def __init__(self, p: ModelParameters, y: np.ndarray):
        self.p = p
        self.y = y
        self.a, self.b = self.build()

    # ------------------------------------------------------------------
    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (a, b) arrays of shape (Ny,) using the layer definitions above.
        Override this method to supply a custom depth profile.
        """
        p = self.p
        y = self.y
        a = np.zeros_like(y)
        b = np.zeros_like(y)

        #layers = list(self.LAYERS.items())
        layers = self.p.layers.layers

        #for i, (name, layer) in enumerate(layers):
        for i, layer in enumerate(layers):

            # Convert absolute depth [m] to model y-coordinate
            top_y = layer.top - p.ysize
            bot_y = layer.bottom - p.ysize

            # First layer
            if i == 0:
                mask = y <= bot_y

            # Last layer
            elif i == len(layers) - 1:
                mask = y > top_y

            # Middle layers
            else:
                mask = (y > top_y) & (y <= bot_y)

            if p.case_type == "california":
                # Linear interpolation of a(y)
                a[mask] = p.a0 + (p.a_max - p.a0) * (y[mask] - p.H) / p.h
                b[mask] = layer.b
            else:
                a[mask] = layer.a
                b[mask] = layer.b

        return a, b
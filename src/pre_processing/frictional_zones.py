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

from src.pre_processing.model_parameters import ModelParameters

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
        "Rocksalt":   {"top": 2000, "bot": 2730, "a": 0.00447,  "b": -0.00590}, # Zechstein rocksalt (halite)
        "BasalZech":  {"top": 2730, "bot": 2780, "a": 0.06895,  "b":  0.07209}, # Basal zechstein
        "TenBoer":    {"top": 2780, "bot": 2850, "a": 0.00305,  "b": -0.00093}, # Ten Boer
        "Sandstone":  {"top": 2850, "bot": 3050, "a": 0.04065,  "b":  0.03796}, # Slochteren Sandstone
        "Carbonif":   {"top": 3050, "bot": 4000, "a": 0.02538,  "b":  0.02347}, # Carboniferous member
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

        layers = list(self.LAYERS.items())

        for i, (name, layer) in enumerate(layers):

            # Convert absolute depth [m] to model y-coordinate
            top_y = layer["top"] - p.ysize
            bot_y = layer["bot"] - p.ysize

            # First layer
            if i == 0:
                mask = y <= bot_y

            # Last layer
            elif i == len(layers) - 1:
                mask = y > top_y

            # Middle layers
            else:
                mask = (y > top_y) & (y <= bot_y)

            a[mask] = layer["a"]
            b[mask] = layer["b"]

        return a, b
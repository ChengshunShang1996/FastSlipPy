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
    Assigns depth-dependent rate-and-state parameters a(y), b(y), and D_rs(y).

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
        profiles = self.build()
        if len(profiles) == 2:
            # Preserve custom ``build()`` implementations written before
            # layer-specific characteristic distances were supported.
            self.a, self.b = profiles
            self.D_rs = np.full_like(y, p.D_rs, dtype=float)
        elif len(profiles) == 3:
            self.a, self.b, self.D_rs = profiles
        else:
            raise ValueError("FrictionalZones.build() must return 2 or 3 profiles.")

        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.D_rs = np.asarray(self.D_rs, dtype=float)
        for name, values in (
            ("a", self.a),
            ("b", self.b),
            ("D_rs", self.D_rs),
        ):
            if values.shape != y.shape:
                raise ValueError(
                    f"Friction profile {name} has shape {values.shape}; "
                    f"expected {y.shape}."
                )

        if np.any(self.a <= 0.0):
            missing = np.flatnonzero(self.a <= 0.0)
            raise ValueError(
                "Friction layers must cover every fault node with a positive "
                f"direct-effect coefficient a; uncovered indices: {missing.tolist()}."
            )
        invalid_D_rs = ~np.isfinite(self.D_rs) | (self.D_rs <= 0.0)
        if np.any(invalid_D_rs):
            invalid = np.flatnonzero(invalid_D_rs)
            raise ValueError(
                "Rate-and-state characteristic distances must be finite and "
                f"positive; invalid fault indices: {invalid.tolist()}."
            )

    # ------------------------------------------------------------------
    def build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(a, b, D_rs)`` arrays of shape ``(Ny,)``.

        A layer-specific ``D_rs`` overrides the model-wide value. Layers that
        omit it inherit :attr:`ModelParameters.D_rs`.
        """
        p = self.p
        y = self.y
        a = np.zeros_like(y)
        b = np.zeros_like(y)
        D_rs = np.full_like(y, p.D_rs, dtype=float)

        if p.case_type == "california":
            a.fill(p.a_max)
            shallow = y < p.H
            transition = (y >= p.H) & (y < p.H + p.h)
            a[shallow] = p.a0
            a[transition] = (
                p.a0
                + (p.a_max - p.a0)
                * (y[transition] - p.H)
                / p.h
            )
            b.fill(p.b0)
            return a, b, D_rs

        #layers = list(self.LAYERS.items())
        layers = self.p.layers.layers

        # Homogeneous fallback keeps the general/default cases usable without
        # requiring callers to construct a one-layer profile explicitly.
        if not layers:
            a.fill(p.a0)
            b.fill(p.b0)
            return a, b, D_rs

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

            a[mask] = layer.a
            b[mask] = layer.b
            if layer.D_rs is not None:
                D_rs[mask] = layer.D_rs

        return a, b, D_rs

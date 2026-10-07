#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "June 26, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////


from dataclasses import dataclass, field
import math
from typing import Optional

@dataclass
class Layer:
    name: str
    top: float
    bottom: float
    a: float
    b: float
    D_rs: Optional[float] = None

@dataclass
class LayerParameters:
    """Depth layers with optional rate-and-state distance overrides.

    When ``D_rs`` is omitted, the layer inherits ``ModelParameters.D_rs``.
    """

    layers: list[Layer] = field(default_factory=list)

    def add(
        self,
        name: str,
        top: float,
        bottom: float,
        a: float,
        b: float,
        D_rs: Optional[float] = None,
    ):
        if D_rs is not None and (not math.isfinite(D_rs) or D_rs <= 0.0):
            raise ValueError("Layer D_rs must be finite and positive.")
        self.layers.append(
            Layer(
                name=name,
                top=top,
                bottom=bottom,
                a=a,
                b=b,
                D_rs=D_rs,
            )
        )

    def clear(self):

        self.layers.clear()

    def set_groningen(self):

        self.clear()

        self.add(
            "Rocksalt",
            2000,
            2730,
            0.00447,
            -0.00590,
        )

        self.add(
            "BasalZech",
            2730,
            2780,
            0.06895,
            0.07209,
        )

        self.add(
            "TenBoer",
            2780,
            2850,
            0.00305,
            -0.00093,
        )

        self.add(
            "Sandstone",
            2850,
            3050,
            0.04065,
            0.03796,
        )

        self.add(
            "Carbonif",
            3050,
            4000,
            0.02538,
            0.02347,
        )

    def set_homogeneous(
        self,
        top: float,
        bottom: float,
        a: float,
        b: float,
        D_rs: Optional[float] = None,
    ):
        self.clear()
        self.add("Homogeneous", top, bottom, a, b, D_rs)

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

@dataclass
class Layer:
    name: str
    top: float
    bottom: float
    a: float
    b: float

@dataclass
class LayerParameters:

    layers: list[Layer] = field(default_factory=list)

    def add(self, name: str, top: float, bottom: float, a: float, b: float):
        self.layers.append(
            Layer(name=name, top=top, bottom=bottom, a=a, b=b)
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

    def set_homogeneous(self, a, b, top=-1e30, bottom=1e30):
        self.clear()
        self.add("Homogeneous", top, bottom, a, b)
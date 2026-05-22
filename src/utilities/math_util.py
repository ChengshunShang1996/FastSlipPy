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

class MathUtil:
    """
    A collection of math utility functions.
    """

    @staticmethod
    def cosd(angle_deg, tol=1e-15):

        angle = angle_deg % 360

        if np.isclose(angle, 0, atol=tol):
            return 1.0

        if np.isclose(angle, 90, atol=tol):
            return 0.0

        if np.isclose(angle, 180, atol=tol):
            return -1.0

        if np.isclose(angle, 270, atol=tol):
            return 0.0

        return np.cos(np.deg2rad(angle))


    @staticmethod
    def sind(angle_deg, tol=1e-15):

        angle = angle_deg % 360

        if np.isclose(angle, 0, atol=tol):
            return 0.0

        if np.isclose(angle, 90, atol=tol):
            return 1.0

        if np.isclose(angle, 180, atol=tol):
            return 0.0

        if np.isclose(angle, 270, atol=tol):
            return -1.0

        return np.sin(np.deg2rad(angle))
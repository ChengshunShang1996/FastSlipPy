from pathlib import Path

import numpy as np

from fastslippy.utilities.diagnose_bp3_interface_bc import (
    _case_parameters,
    _numeric_rows,
)


def test_case_parameter_parser_handles_numeric_expressions(tmp_path: Path):
    case_file = tmp_path / "run_case_bp3.py"
    case_file.write_text(
        "from somewhere import ModelParameters\n"
        "p = ModelParameters(alpha=60.0, xsize=400e3, Nx=2 * 480 + 1, "
        "x_stretch_enabled=True, motion_sign=-1)\n",
        encoding="utf-8",
    )
    values = _case_parameters(case_file)
    assert values["alpha"] == 60.0
    assert values["xsize"] == 400e3
    assert values["Nx"] == 961
    assert values["x_stretch_enabled"] is True
    assert values["motion_sign"] == -1


def test_numeric_rows_skips_headers(tmp_path: Path):
    station = tmp_path / "station"
    station.write_text(
        "# header\nt disp_1 disp_2 vel_1 vel_2\n"
        "0 1 2 3 4\n1 5 6 7 8 extra\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(
        _numeric_rows(station, 5),
        [[0, 1, 2, 3, 4], [1, 5, 6, 7, 8]],
    )

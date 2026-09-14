from types import SimpleNamespace
from xml.etree import ElementTree as ET

import meshio
import numpy as np
import pytest

from fastslippy.post_processing.output_manager import OutputManager
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters


def _vtk_fixture(tmp_path, *, alpha=60.0):
    params = ModelParameters(
        alpha=alpha,
        Nx=5,
        Ny=5,
        xsize=4.0,
        ysize=4.0,
        Nt=2,
        output_interval=1,
        checkpoint_interval=10,
        output_vtk_option=False,
    )
    grid = Grid(params)
    mid = params.Nx // 2

    ux = np.full((params.Ny + 1, params.Nx), 2.0)
    vx = np.full((params.Ny + 1, params.Nx), 4.0)
    uy = np.empty((params.Ny, params.Nx + 1))
    vy = np.empty_like(uy)
    uy[:, :mid + 1] = -1.0
    uy[:, mid + 1:] = 3.0
    vy[:, :mid + 1] = -2.0
    vy[:, mid + 1:] = 6.0

    tauqs = np.arange(params.Ny * params.Nx, dtype=float).reshape(
        params.Ny, params.Nx
    )
    sigmaqs = 100.0 + np.arange(
        (params.Ny - 1) * (params.Nx - 1), dtype=float
    ).reshape(params.Ny - 1, params.Nx - 1)
    fault = SimpleNamespace(
        U=np.linspace(0.0, 1.0, params.Ny),
        V=np.linspace(1.0, 2.0, params.Ny),
        tau=np.linspace(2.0, 3.0, params.Ny),
        sigma=np.linspace(3.0, 4.0, params.Ny),
        theta=np.linspace(4.0, 5.0, params.Ny).astype(complex),
    )
    output = OutputManager(params, tmp_path)
    return params, grid, output, ux, uy, vx, vy, tauqs, sigmaqs, fault


def test_vtk_preserves_fault_jump_and_complete_global_vectors(tmp_path):
    (
        params,
        grid,
        output,
        ux,
        uy,
        vx,
        vy,
        tauqs,
        sigmaqs,
        fault,
    ) = _vtk_fixture(tmp_path)
    try:
        output.write_vtk(
            7, grid, ux, uy, vx, vy, tauqs, sigmaqs, fault, 12.5
        )
    finally:
        output.close()

    vtk_dir = tmp_path / "vtu_results"
    left = meshio.read(vtk_dir / "left_it00007.vtu")
    right = meshio.read(vtk_dir / "right_it00007.vtu")
    fault_mesh = meshio.read(vtk_dir / "fault_it00007.vtu")
    mid = params.Nx // 2

    assert left.cells[0].type == "quad"
    assert right.cells[0].type == "quad"
    assert not (vtk_dir / "sigma_it00007.vtu").exists()

    left_points = left.points.reshape(params.Ny, mid + 1, 3)
    right_points = right.points.reshape(params.Ny, params.Nx - mid, 3)
    np.testing.assert_allclose(left_points[:, -1], right_points[:, 0])
    np.testing.assert_allclose(
        left_points[:, -1, :2],
        np.column_stack((grid.y * grid.cosa, grid.y * grid.sina)),
    )

    left_u = left.point_data["displacement_m"].reshape(
        params.Ny, mid + 1, 3
    )
    right_u = right.point_data["displacement_m"].reshape(
        params.Ny, params.Nx - mid, 3
    )
    left_v = left.point_data["velocity_m_per_s"].reshape(
        params.Ny, mid + 1, 3
    )
    right_v = right.point_data["velocity_m_per_s"].reshape(
        params.Ny, params.Nx - mid, 3
    )
    np.testing.assert_allclose(
        left_u[:, -1],
        np.tile([2.0 - grid.cosa, -grid.sina, 0.0], (params.Ny, 1)),
    )
    np.testing.assert_allclose(
        right_u[:, 0],
        np.tile(
            [2.0 + 3.0 * grid.cosa, 3.0 * grid.sina, 0.0],
            (params.Ny, 1),
        ),
    )
    np.testing.assert_allclose(
        left_v[:, -1],
        np.tile(
            [4.0 - 2.0 * grid.cosa, -2.0 * grid.sina, 0.0],
            (params.Ny, 1),
        ),
    )
    np.testing.assert_allclose(
        right_v[:, 0],
        np.tile(
            [4.0 + 6.0 * grid.cosa, 6.0 * grid.sina, 0.0],
            (params.Ny, 1),
        ),
    )

    left_tau = left.point_data["shear_stress_quasistatic_Pa"].reshape(
        params.Ny, mid + 1
    )
    right_tau = right.point_data["shear_stress_quasistatic_Pa"].reshape(
        params.Ny, params.Nx - mid
    )
    np.testing.assert_array_equal(left_tau[:, -1], tauqs[:, mid - 1])
    np.testing.assert_array_equal(right_tau[:, 0], tauqs[:, mid + 1])
    np.testing.assert_array_equal(
        left.cell_data["normal_stress_quasistatic_Pa"][0],
        sigmaqs[:, :mid].ravel(order="C"),
    )
    np.testing.assert_array_equal(
        right.cell_data["normal_stress_quasistatic_Pa"][0],
        sigmaqs[:, mid:].ravel(order="C"),
    )

    assert fault_mesh.cells[0].type == "line"
    assert "slip_rate_V_m_per_s" in fault_mesh.point_data
    np.testing.assert_allclose(fault_mesh.point_data["time_s"], 12.5)

    datasets = ET.parse(vtk_dir / "results.pvd").findall(".//DataSet")
    assert len(datasets) == 3
    assert {dataset.get("file") for dataset in datasets} == {
        "left_it00007.vtu",
        "right_it00007.vtu",
        "fault_it00007.vtu",
    }
    assert {float(dataset.get("timestep")) for dataset in datasets} == {12.5}


def test_vtk_collection_retains_multiple_physical_times(tmp_path):
    fixture = _vtk_fixture(tmp_path, alpha=90.0)
    params, grid, output, *fields = fixture
    ux, uy, vx, vy, tauqs, sigmaqs, fault = fields
    output.write_vtk(
        2, grid, ux, uy, vx, vy, tauqs, sigmaqs, fault, 0.25
    )
    output.close()

    restarted_output = OutputManager(params, tmp_path, append_log=True)
    try:
        restarted_output.write_vtk(
            5, grid, ux, uy, vx, vy, tauqs, sigmaqs, fault, 3.75
        )
    finally:
        restarted_output.close()

    datasets = ET.parse(
        tmp_path / "vtu_results" / "results.pvd"
    ).findall(".//DataSet")
    assert len(datasets) == 6
    assert [
        float(dataset.get("timestep"))
        for dataset in datasets[::3]
    ] == [0.25, 3.75]

    truncated_output = OutputManager(params, tmp_path, append_log=True)
    try:
        truncated_output.write_vtk(
            3, grid, ux, uy, vx, vy, tauqs, sigmaqs, fault, 1.5
        )
    finally:
        truncated_output.close()
    datasets = ET.parse(
        tmp_path / "vtu_results" / "results.pvd"
    ).findall(".//DataSet")
    assert [dataset.get("file") for dataset in datasets[::3]] == [
        "left_it00002.vtu",
        "left_it00003.vtu",
    ]


def test_vtk_uses_coordinate_aware_interpolation_on_stretched_grid(tmp_path):
    params = ModelParameters(
        alpha=55.0,
        Nx=9,
        Ny=7,
        xsize=8.0,
        ysize=6.0,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=2.0,
        y_stretch_inner_size=1.5,
        x_stretch_inner_points=5,
        y_stretch_inner_points=4,
        allow_nonuniform_solver=True,
        Nt=1,
        output_interval=1,
        checkpoint_interval=1,
        output_vtk_option=False,
    )
    grid = Grid(params)
    assert grid.is_nonuniform

    ux = 2.0 * grid.yp[:, None] + 3.0 * grid.x[None, :]
    uy = -grid.y[:, None] + 4.0 * grid.xp[None, :]
    vx = -0.5 * ux
    vy = 0.25 * uy
    tauqs = np.zeros((params.Ny, params.Nx))
    sigmaqs = np.zeros((params.Ny - 1, params.Nx - 1))
    fault = SimpleNamespace(
        U=np.zeros(params.Ny),
        V=np.zeros(params.Ny),
        tau=np.zeros(params.Ny),
        sigma=np.zeros(params.Ny),
        theta=np.ones(params.Ny),
    )
    output = OutputManager(params, tmp_path)
    try:
        output.write_vtk(
            1, grid, ux, uy, vx, vy, tauqs, sigmaqs, fault, 1.0
        )
    finally:
        output.close()

    mid = params.Nx // 2
    vtk_dir = tmp_path / "vtu_results"
    left = meshio.read(vtk_dir / "left_it00001.vtu")
    right = meshio.read(vtk_dir / "right_it00001.vtu")
    normal_expected = 2.0 * grid.y[:, None] + 3.0 * grid.x[None, :]
    tangential_expected = -grid.y[:, None] + 4.0 * grid.x[None, :]
    tangential_left = tangential_expected[:, :mid + 1].copy()
    tangential_right = tangential_expected[:, mid:].copy()
    tangential_left[:, -1] = uy[:, mid]
    tangential_right[:, 0] = uy[:, mid + 1]

    np.testing.assert_allclose(
        left.point_data["ux_on_tau_m"].reshape(
            params.Ny, mid + 1
        ),
        normal_expected[:, :mid + 1],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        right.point_data["ux_on_tau_m"].reshape(
            params.Ny, params.Nx - mid
        ),
        normal_expected[:, mid:],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        left.point_data["uy_on_tau_m"].reshape(
            params.Ny, mid + 1
        ),
        tangential_left,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        right.point_data["uy_on_tau_m"].reshape(
            params.Ny, params.Nx - mid
        ),
        tangential_right,
        atol=1e-14,
    )


def test_vtk_rejects_invalid_field_shapes_and_values(tmp_path):
    fixture = _vtk_fixture(tmp_path)
    _, grid, output, ux, uy, vx, vy, tauqs, sigmaqs, fault = fixture
    try:
        with pytest.raises(ValueError, match="uy has shape"):
            output.write_vtk(
                1,
                grid,
                ux,
                uy[:, :-1],
                vx,
                vy,
                tauqs,
                sigmaqs,
                fault,
                1.0,
            )
        bad_tau = tauqs.copy()
        bad_tau[0, 0] = np.nan
        with pytest.raises(ValueError, match="tauqs contains non-finite"):
            output.write_vtk(
                1, grid, ux, uy, vx, vy, bad_tau, sigmaqs, fault, 1.0
            )
    finally:
        output.close()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_vtk_interval_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="vtk_interval"):
        ModelParameters(vtk_interval=value)


def test_vtk_interval_defaults_to_checkpoint_interval():
    assert ModelParameters(checkpoint_interval=17).vtk_interval == 17
    assert ModelParameters(checkpoint_interval=17, vtk_interval=3).vtk_interval == 3

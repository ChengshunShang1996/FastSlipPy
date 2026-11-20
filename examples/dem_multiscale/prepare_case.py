"""Generate the deterministic 2-D gouge and rough-wall Kratos input files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CASE_DIRECTORY = Path(__file__).resolve().parent
LENGTH = 0.12
HEIGHT = 0.052
COARSE_RADIUS = 0.5 * LENGTH / 20.0
FINE_RADIUS = 0.5 * COARSE_RADIUS
FINE_LAYER_WIDTH = 0.012


def _periodic_grid(radius: float) -> tuple[int, float]:
    columns = max(1, int(np.floor(LENGTH / (2.0 * radius))))
    return columns, LENGTH / columns


def _regular_layer(y_min: float, y_max: float, radius: float):
    if y_max < y_min:
        return []
    columns, spacing_x = _periodic_grid(radius)
    spacing_y = np.sqrt(max((2.0 * radius) ** 2 - (0.5 * spacing_x) ** 2, 0.0))
    rows = max(1, int(np.floor((y_max - y_min) / spacing_y)) + 1)
    elevations = (
        np.array([0.5 * (y_min + y_max)])
        if rows == 1
        else np.linspace(y_min, y_max, rows)
    )
    particles = []
    for row, y in enumerate(elevations):
        offset = 0.5 * spacing_x if row % 2 else 0.0
        for column in range(columns):
            x = ((column + 0.5) * spacing_x + offset) % LENGTH
            particles.append((float(x), float(y), radius))
    return particles


def _packing():
    wall_count, wall_spacing = _periodic_grid(COARSE_RADIUS)
    first_y = np.sqrt((2.0 * COARSE_RADIUS) ** 2 - (0.5 * wall_spacing) ** 2)
    last_y = HEIGHT - first_y
    centre = 0.5 * HEIGHT
    fine_bottom = centre - 0.5 * FINE_LAYER_WIDTH
    fine_top = centre + 0.5 * FINE_LAYER_WIDTH
    gouge = (
        _regular_layer(first_y, fine_bottom - COARSE_RADIUS, COARSE_RADIUS)
        + _regular_layer(
            fine_bottom + FINE_RADIUS,
            fine_top - FINE_RADIUS,
            FINE_RADIUS,
        )
        + _regular_layer(fine_top + COARSE_RADIUS, last_y, COARSE_RADIUS)
    )
    bottom = [
        (index * wall_spacing, 0.0, COARSE_RADIUS)
        for index in range(wall_count)
    ]
    top_row_x = min(x for x, y, _ in gouge if np.isclose(y, last_y))
    top_offset = (top_row_x + 0.5 * wall_spacing) % wall_spacing
    top = [
        ((top_offset + index * wall_spacing) % LENGTH, HEIGHT, COARSE_RADIUS)
        for index in range(wall_count)
    ]
    return gouge, bottom, top


def _submodel_part(name: str, identifiers: list[int]) -> list[str]:
    lines = [f"Begin SubModelPart {name}", "  Begin SubModelPartNodes"]
    lines.extend(f"    {identifier}" for identifier in identifiers)
    lines.extend(["  End SubModelPartNodes", "  Begin SubModelPartElements"])
    lines.extend(f"    {identifier}" for identifier in identifiers)
    lines.extend(
        [
            "  End SubModelPartElements",
            "  Begin SubModelPartConditions",
            "  End SubModelPartConditions",
            "End SubModelPart",
            "",
        ]
    )
    return lines


def write_case(directory: Path = CASE_DIRECTORY) -> None:
    gouge, bottom, top = _packing()
    fine_ids = [
        identifier
        for identifier, (_, y, _) in enumerate(gouge, start=1)
        if abs(y - 0.5 * HEIGHT) <= 0.5 * FINE_LAYER_WIDTH
    ]
    write_packing(directory, gouge, bottom, top, fine_ids)


def write_packing(directory, gouge, bottom, top, fine_ids):
    """Write free cylinders and two velocity-constrained rough walls."""
    directory.mkdir(parents=True, exist_ok=True)
    particles = gouge + bottom + top
    gouge_ids = list(range(1, len(gouge) + 1))
    bottom_ids = list(range(gouge_ids[-1] + 1, gouge_ids[-1] + len(bottom) + 1))
    top_ids = list(range(bottom_ids[-1] + 1, bottom_ids[-1] + len(top) + 1))

    lines = [
        "Begin ModelPartData",
        "End ModelPartData",
        "",
        "Begin Properties 0",
        "End Properties",
        "",
        "Begin Nodes",
    ]
    for identifier, (x, y, _) in enumerate(particles, start=1):
        lines.append(f"  {identifier} {x:.16e} {y:.16e} 0.0")
    lines.extend(["End Nodes", "", "Begin Elements CylinderParticle2D"])
    for identifier in range(1, len(particles) + 1):
        lines.append(f"  {identifier} 0 {identifier}")
    lines.extend(["End Elements", "", "Begin NodalData RADIUS"])
    for identifier, (_, _, radius) in enumerate(particles, start=1):
        lines.append(f"  {identifier} 0 {radius:.16e}")
    lines.extend(["End NodalData", ""])

    wall_ids = bottom_ids + top_ids
    for component in ("X", "Y", "Z"):
        lines.append(f"Begin NodalData VELOCITY_{component}")
        lines.extend(f"  {identifier} 1 0.0" for identifier in wall_ids)
        lines.extend(["End NodalData", ""])
    for component in ("X", "Y", "Z"):
        lines.append(f"Begin NodalData ANGULAR_VELOCITY_{component}")
        lines.extend(f"  {identifier} 1 0.0" for identifier in wall_ids)
        lines.extend(["End NodalData", ""])

    lines.extend(_submodel_part("Gouge", gouge_ids))
    if fine_ids:
        lines.extend(_submodel_part("FineLayer", fine_ids))
    lines.extend(_submodel_part("BottomWall", bottom_ids))
    lines.extend(_submodel_part("TopWall", top_ids))
    (directory / "dem_shearDEM.mdpa").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (directory / "dem_shearDEM_FEM_boundary.mdpa").write_text(
        "Begin ModelPartData\nEnd ModelPartData\n\n"
        "Begin Properties 0\nEnd Properties\n",
        encoding="utf-8",
    )
    print(
        f"Generated {len(gouge)} gouge particles and "
        f"{len(bottom) + len(top)} wall particles in {directory}."
    )


if __name__ == "__main__":
    write_case()


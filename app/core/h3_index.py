"""Thin wrapper around Uber H3 v4 so the rest of the code never touches h3 directly."""
import h3

from ..config import settings


def to_cell(lat: float, lng: float, res: int | None = None) -> str:
    return h3.latlng_to_cell(lat, lng, res or settings.h3_res)


def boundary(cell: str) -> list[list[float]]:
    return [list(p) for p in h3.cell_to_boundary(cell)]


def center(cell: str) -> tuple[float, float]:
    return h3.cell_to_latlng(cell)


def ring(cell: str, k: int = 1) -> list[str]:
    return list(h3.grid_disk(cell, k))


def ring_distance(a: str, b: str, default: int = 10**6) -> int:
    """Grid distance in hex steps; huge value if the cells are too far apart for H3 to path."""
    try:
        return h3.grid_distance(a, b)
    except Exception:
        return default

"""Road distance/duration. Uses the free OSRM demo server if enabled, else haversine * road factor."""
import httpx

from ..config import settings
from .geo import ROAD_FACTOR, haversine_km


def route(lat1, lng1, lat2, lng2) -> tuple[float, float]:
    """Returns (km, minutes)."""
    if settings.use_osrm:
        try:
            url = f"{settings.osrm_url}/route/v1/driving/{lng1},{lat1};{lng2},{lat2}?overview=false"
            r = httpx.get(url, timeout=4).json()["routes"][0]
            return r["distance"] / 1000, r["duration"] / 60
        except Exception:
            pass
    km = float(haversine_km(lat1, lng1, lat2, lng2)) * ROAD_FACTOR
    return km, km / 20 * 60

import math

import numpy as np

ROAD_FACTOR = 1.3  # straight-line -> road distance fudge when OSRM is off


def haversine_km(lat1, lng1, lat2, lng2):
    """Vectorised great-circle distance in km (broadcasts like numpy)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def offset_point(lat, lng, dx_km, dy_km):
    return lat + dy_km / 111.0, lng + dx_km / (111.0 * math.cos(math.radians(lat)))

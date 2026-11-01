from dataclasses import dataclass


@dataclass
class RiderState:
    id: str
    lat: float
    lng: float
    earnings: float = 0.0
    free_at: float = 0.0  # simulator only
    n: int = 0


@dataclass
class OrderState:
    id: int
    rlat: float
    rlng: float
    clat: float
    clng: float
    created: float = 0.0  # simulator only
    wait_min: float = 0.0

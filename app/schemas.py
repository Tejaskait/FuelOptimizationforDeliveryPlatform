from pydantic import BaseModel, Field


class PingIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class OrderIn(BaseModel):
    rlat: float
    rlng: float
    clat: float
    clng: float


class WeightsIn(BaseModel):
    time: float = Field(ge=0, le=10)
    pay: float = Field(ge=0, le=10)
    fair: float = Field(ge=0, le=10)

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, utcnow


class Rider(Base):
    __tablename__ = "riders"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    lat: Mapped[float] = mapped_column(Float, default=0.0)
    lng: Mapped[float] = mapped_column(Float, default=0.0)
    h3: Mapped[str] = mapped_column(String(20), default="", index=True)
    status: Mapped[str] = mapped_column(String(16), default="idle", index=True)  # idle|busy
    is_virtual: Mapped[bool] = mapped_column(Boolean, default=False)
    earnings: Mapped[float] = mapped_column(Float, default=0.0)
    n_orders: Mapped[int] = mapped_column(Integer, default=0)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rlat: Mapped[float] = mapped_column(Float)
    rlng: Mapped[float] = mapped_column(Float)
    clat: Mapped[float] = mapped_column(Float)
    clng: Mapped[float] = mapped_column(Float)
    rh3: Mapped[str] = mapped_column(String(20), default="", index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    rider_id: Mapped[str | None] = mapped_column(ForeignKey("riders.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    km: Mapped[float] = mapped_column(Float, default=0.0)
    eta_min: Mapped[float] = mapped_column(Float, default=0.0)
    payout: Mapped[float] = mapped_column(Float, default=0.0)


class Ping(Base):
    __tablename__ = "pings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rider_id: Mapped[str] = mapped_column(String(64), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lng: Mapped[float] = mapped_column(Float)
    h3: Mapped[str] = mapped_column(String(20), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

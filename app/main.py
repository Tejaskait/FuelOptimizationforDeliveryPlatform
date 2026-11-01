import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .core import h3_index
from .core.geo import offset_point
from .core.metrics import gini
from .db import Base, SessionLocal, engine, get_db, utcnow
from .ml import demand
from .models import Order, Ping, Rider
from .schemas import OrderIn, PingIn, WeightsIn
from .services import dispatcher

STATIC = Path(__file__).parent / "static"


class Hub:
    def __init__(self):
        self.conns: set[WebSocket] = set()

    async def broadcast(self, msg: dict):
        for ws in list(self.conns):
            try:
                await ws.send_json(msg)
            except Exception:
                self.conns.discard(ws)


hub = Hub()


async def dispatch_loop():
    while True:
        await asyncio.sleep(settings.batch_seconds)
        with SessionLocal() as db:
            res = dispatcher.run_batch(db)
        if res:
            await hub.broadcast({"type": "assign", "n": len(res)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    task = asyncio.create_task(dispatch_loop())
    yield
    task.cancel()


app = FastAPI(title="Delivery Platform", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def root():
    return RedirectResponse("/dashboard")


@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/tracker")
def tracker():
    return FileResponse(STATIC / "tracker.html")


@app.get("/health")
def health():
    return {"ok": True}


# ---------- riders (phones are riders) ----------
@app.put("/riders/{rid}/ping")
async def ping(rid: str, p: PingIn, db: Session = Depends(get_db)):
    cell = h3_index.to_cell(p.lat, p.lng)
    r = db.get(Rider, rid)
    if r is None:
        r = Rider(id=rid, name=rid, status="idle", is_virtual=False, earnings=0.0, n_orders=0)
        db.add(r)
    r.lat, r.lng, r.h3, r.last_seen = p.lat, p.lng, cell, utcnow()
    db.add(Ping(rider_id=rid, lat=p.lat, lng=p.lng, h3=cell))
    db.commit()
    await hub.broadcast({"type": "ping", "id": rid, "lat": p.lat, "lng": p.lng, "h3": cell})
    return {"h3": cell}


@app.get("/riders")
def riders(db: Session = Depends(get_db)):
    return [dict(id=r.id, lat=r.lat, lng=r.lng, h3=r.h3, status=r.status, virtual=r.is_virtual,
                 earnings=round(r.earnings, 1), n_orders=r.n_orders) for r in db.scalars(select(Rider))]


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    hub.conns.add(sock)
    try:
        while True:
            await sock.receive_text()
    except WebSocketDisconnect:
        hub.conns.discard(sock)


# ---------- orders ----------
@app.post("/orders")
def create_order(o: OrderIn, db: Session = Depends(get_db)):
    row = Order(**o.model_dump(), rh3=h3_index.to_cell(o.rlat, o.rlng))
    db.add(row)
    db.commit()
    return {"id": row.id}


@app.get("/orders")
def orders(db: Session = Depends(get_db)):
    rows = db.scalars(select(Order).order_by(Order.id.desc()).limit(50))
    return [dict(id=o.id, status=o.status, rider_id=o.rider_id, km=round(o.km, 2), eta_min=round(o.eta_min, 1),
                 payout=round(o.payout, 1), rlat=o.rlat, rlng=o.rlng, clat=o.clat, clng=o.clng) for o in rows]


@app.post("/orders/{oid}/complete")
def complete(oid: int, db: Session = Depends(get_db)):
    o = db.get(Order, oid)
    if not o or o.status != "assigned":
        raise HTTPException(409, "order not in assigned state")
    return dispatcher.complete_order(db, o)


# ---------- dispatch / config / analytics ----------
@app.post("/dispatch/run")
async def dispatch_now(db: Session = Depends(get_db)):
    res = dispatcher.run_batch(db)
    await hub.broadcast({"type": "assign", "n": len(res)})
    return res


@app.get("/config/weights")
def get_weights():
    return dispatcher.weights.__dict__


@app.post("/config/weights")
def set_weights(w: WeightsIn):
    dispatcher.weights.time, dispatcher.weights.pay, dispatcher.weights.fair = w.time, w.pay, w.fair
    return dispatcher.weights.__dict__


@app.get("/hexes")
def hexes(db: Session = Depends(get_db)):
    cells: dict[str, dict] = {}
    for r in db.scalars(select(Rider)):
        if r.h3:
            cells.setdefault(r.h3, {"riders": 0, "pending": 0})["riders"] += 1
    for o in db.scalars(select(Order).where(Order.status == "pending")):
        cells.setdefault(o.rh3, {"riders": 0, "pending": 0})["pending"] += 1
    return [{"h3": c, **v, "boundary": h3_index.boundary(c)} for c, v in cells.items()]


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    earn = list(db.scalars(select(Rider.earnings)))
    done = db.execute(select(func.count(Order.id), func.avg(Order.payout)).where(Order.status == "delivered")).one()
    return {"riders": len(earn), "delivered": done[0], "avg_payout": float(done[1] or 0),
            "gini": gini(earn), "weights": dispatcher.weights.__dict__}


@app.get("/hotspots")
def hotspots(hour: int | None = None, dow: int | None = None):
    now = utcnow()
    return demand.hotspots(Path(settings.model_dir) / "demand.joblib",
                           hour if hour is not None else dispatcher.local_hour(),
                           dow if dow is not None else now.weekday())


# ---------- demo helpers (so 2 phones feel like a city) ----------
def _center(db: Session):
    real = [(r.lat, r.lng) for r in db.scalars(select(Rider).where(Rider.is_virtual.is_(False))) if r.lat]
    return tuple(np.mean(real, axis=0)) if real else (settings.center_lat, settings.center_lng)


def _pt(c, radius, rng):
    t, d = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform()) * radius
    return offset_point(c[0], c[1], d * np.cos(t), d * np.sin(t))


@app.post("/demo/virtual-riders")
def demo_riders(n: int = 8, db: Session = Depends(get_db)):
    rng, c = np.random.default_rng(), _center(db)
    base = db.scalar(select(func.count(Rider.id))) or 0
    for i in range(n):
        lat, lng = _pt(c, 4, rng)
        db.add(Rider(id=f"bot-{base + i}", name=f"bot-{base + i}", lat=lat, lng=lng, h3=h3_index.to_cell(lat, lng),
                     is_virtual=True, status="idle", earnings=0.0, n_orders=0, last_seen=utcnow()))
    db.commit()
    return {"added": n}


@app.post("/demo/orders")
def demo_orders(n: int = 5, db: Session = Depends(get_db)):
    rng, c = np.random.default_rng(), _center(db)
    for _ in range(n):
        r, cu = _pt(c, 2.5, rng), _pt(c, 5, rng)
        db.add(Order(rlat=r[0], rlng=r[1], clat=cu[0], clng=cu[1], rh3=h3_index.to_cell(*r)))
    db.commit()
    return {"added": n}

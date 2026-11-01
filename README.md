# Delivery Platform — H3 dispatch, fair rider payouts, ML ETA

A Swiggy/Zomato-style dispatch backend. Phones are real riders (live GPS), bots fill out the city,
and a batch optimizer assigns orders by trading off **speed, platform payout cost, and rider earnings fairness**.

## Architecture
```
 phone (/tracker) ──PUT /riders/{id}/ping──▶ FastAPI ──▶ SQLite / Postgres (Neon free)
 dashboard (/dashboard) ◀── WebSocket + REST ──┘   │
                                                   ├─ core/h3_index   H3 cells, rings, boundaries
                                                   ├─ core/assignment scipy Hungarian, H3 feasibility mask
                                                   ├─ core/payout     slab payout + capped late penalty
                                                   ├─ ml/eta_model    GradientBoosting ETA (vs naive)
                                                   ├─ ml/demand       per-hex hourly demand forecast
                                                   └─ services/dispatcher  30s batch loop
 simulator/ + scripts/  — same assign()/payout() code, offline experiments (Pareto sweep)
```

## Tech stack (all free)
FastAPI · SQLAlchemy 2 · SQLite→Postgres · Uber H3 · NumPy/Pandas/SciPy · scikit-learn · Leaflet + CARTO tiles
(no OSM-blocking) · OSRM demo server (optional) · GitHub Actions CI · Docker · Render + Neon free tiers · ngrok/cloudflared for phone testing.

## Quickstart
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python -m scripts.train_models                      # ETA + demand models
pytest -q
uvicorn app.main:app --reload                       # http://localhost:8000/dashboard
```
Dashboard: **+8 bot riders → +5 orders → Dispatch now → done** on assigned orders; move the α/β/δ sliders and watch earnings/Gini.

## Using your 2 phones
Phone GPS needs **HTTPS**. Pick one:
- `cloudflared tunnel --url http://localhost:8000` (free) or `ngrok http 8000`, then open `<url>/tracker` on each phone (ids `phone1`, `phone2`) → Start.
- Or deploy (below) and open `<deployed-url>/tracker`.

Bots and demo orders spawn around your phones' centroid, so the "city" is wherever you are.

## Experiments
```bash
python -m scripts.sweep        # reports/pareto.png + sweep.csv  (beta x delta grid vs nearest baseline)
python -m scripts.compare_eta  # naive vs learned ETA inside the simulator
```
Observed with defaults (40 riders, 0.5 orders/min): switching from nearest to fairness weight δ=3 cuts earnings
Gini ~0.22 → ~0.07 with ≈ same average delivery time. Note the fairness term only changes outcomes when there are
**more idle riders than orders** (otherwise every rider is matched anyway) — a genuine property of batch matching.

## Cost function
`cost(r,o) = α·(ETA − 0.3·wait) + β·payout/10 + δ·(earnings_r − fleet_mean)/10`, solved globally with
`linear_sum_assignment`; riders more than `MAX_RING` H3 hexes from the restaurant are infeasible.

## API
| Method | Path | Purpose |
|---|---|---|
| PUT | `/riders/{id}/ping` | phone GPS → H3 cell, stored |
| GET | `/riders` `/orders` `/hexes` `/metrics` `/hotspots` | state, hex heatmap, Gini, demand forecast |
| POST | `/orders` | create order (rlat,rlng,clat,clng) |
| POST | `/dispatch/run` | run one batch now (also runs every `BATCH_SECONDS`) |
| POST | `/orders/{id}/complete` | deliver → payout with lateness penalty |
| GET/POST | `/config/weights` | live-tune α/β/δ |
| POST | `/demo/virtual-riders?n=` `/demo/orders?n=` | seed bots/orders |
| WS | `/ws` | live ping/assign events |

## Free deployment
1. Neon/Supabase: create free Postgres, copy URL as `postgresql+psycopg://...`.
2. Push to GitHub; on Render create a Web Service from the repo (Docker), set `DATABASE_URL`.
3. Open `https://<app>.onrender.com/tracker` on the phones.
(Free tier sleeps when idle; first request takes ~30s.)

## Roadmap to production
Replace synthetic training data with logged orders (`ml/` already takes DataFrames) · multi-order batching per rider ·
Alembic migrations · auth for rider endpoints · OSRM self-hosted (`USE_OSRM=true`) · Redis for hot rider positions · rider-acceptance model.

## Train on your own phones (real data)
1. Run the server (see Quickstart), open `/tracker` on the phone(s), and **carry them while you actually move** (walk/ride/drive).
   Every ping is stored in the `pings` table.
2. After ~30+ minutes of movement with a few stops: `python -m scripts.train_from_pings`
   (or `--csv pings.csv` with columns `rider_id,lat,lng,ts`). It splits pings into trips, writes `data/trips.csv`,
   and retrains `models/eta.joblib`. Restart the server to load it.
3. With <30 trips it fits on all data and reports in-sample error only — treat that as a calibration, not validation.
The demand forecaster needs real order logs, which GPS alone cannot provide; it stays synthetic until you have orders.
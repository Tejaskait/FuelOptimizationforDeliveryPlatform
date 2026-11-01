from fastapi.testclient import TestClient
from app.main import app


def test_full_flow():
    with TestClient(app) as c:
        assert c.put("/riders/phone1/ping", json={"lat": 12.97, "lng": 77.59}).status_code == 200
        c.post("/demo/virtual-riders?n=4")
        c.post("/demo/orders?n=3")
        res = c.post("/dispatch/run").json()
        assert 1 <= len(res) <= 3
        oid = res[0]["order_id"]
        assert c.post(f"/orders/{oid}/complete").json()["payout"] > 0
        m = c.get("/metrics").json()
        assert m["delivered"] == 1 and 0 <= m["gini"] <= 1
        assert c.get("/hexes").json()
        assert c.post("/config/weights", json={"time": 1, "pay": 1, "fair": 2}).json()["fair"] == 2
        assert c.get("/dashboard").status_code == 200

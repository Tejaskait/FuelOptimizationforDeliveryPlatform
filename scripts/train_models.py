"""python -m scripts.train_models  -> trains ETA + demand models on synthetic data (swap in real logs later)."""
from pathlib import Path

from app.config import settings
from app.ml import demand, eta_model
from simulator.synth import eta_training_data, order_log

if __name__ == "__main__":
    d = Path(settings.model_dir)
    print("ETA   :", eta_model.train(eta_training_data(), d / "eta.joblib"))
    print("Demand:", demand.train(order_log(), d / "demand.joblib"))

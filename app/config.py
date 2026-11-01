from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"
    h3_res: int = 8
    center_lat: float = 12.9716
    center_lng: float = 77.5946
    tz_offset_hours: float = 5.5
    batch_seconds: int = 30
    w_time: float = 1.0
    w_pay: float = 0.5
    w_fair: float = 1.0
    max_ring: int = 25
    use_osrm: bool = False
    osrm_url: str = "https://router.project-osrm.org"
    model_dir: str = "models"


settings = Settings()

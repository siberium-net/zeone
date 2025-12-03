import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import geoip2.database
    _GEO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GEO_AVAILABLE = False

DEFAULT_RULES = {
    "DE": {"banned_topics": ["nazi_symbolism", "holocaust_denial"]},
    "RU": {"banned_topics": ["lgbt_propaganda", "extremism"]},
    "US": {"banned_topics": ["dmca_violation", "csam"]},
}


class GeoRules:
    def __init__(self, rules_path: str = "legal_rules.json", geo_db: str = None):
        self.rules = DEFAULT_RULES
        path = Path(rules_path)
        if path.exists():
            try:
                self.rules = json.loads(path.read_text())
            except Exception as e:
                logger.warning(f"[COMPLIANCE] Failed to load legal_rules: {e}")
        self.geo_db = geo_db
        self.reader = None
        if geo_db and Path(geo_db).exists() and _GEO_AVAILABLE:
            try:
                self.reader = geoip2.database.Reader(geo_db)
            except Exception as e:
                logger.warning(f"[COMPLIANCE] Geo DB load failed: {e}")

    def get_country(self, ip: str) -> str:
        if self.reader:
            try:
                resp = self.reader.country(ip)
                return resp.country.iso_code or "US"
            except Exception:
                return "US"
        return "US"

    def get_rules_for_country(self, country: str) -> Dict[str, Any]:
        return self.rules.get(country, {})

from src.constants import validate_gkx_schema
from src.features.gkx_registry import GKX_94


def test_validate_gkx_schema_detects_complete_set():
    cols = ["ticker", "date"] + GKX_94
    out = validate_gkx_schema(cols)
    assert len(out["missing"]) == 0
    assert len(out["present"]) == 94


def test_validate_gkx_schema_detects_missing_set():
    cols = ["ticker", "date", "mom12m", "bm"]
    out = validate_gkx_schema(cols)
    assert "mom12m" in out["present"]
    assert len(out["missing"]) == 92

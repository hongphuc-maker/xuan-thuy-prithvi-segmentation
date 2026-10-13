from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "experiments"


def load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_completed_wce_contract():
    config = load("p2_prithvi_unet_wce_completed.json")
    assert config["experiment_id"] == "P2_multitemporal6_sep21_UNet_WCE"
    assert config["target_date"] == "2026-09-21"
    assert config["bands"] == ["B02", "B03", "B04", "B8A", "B11", "B12"]
    assert len(config["active_dates"]) == 6
    assert config["radiometry"]["dn_offset"] == -1000.0
    assert config["model"]["decoder"] == "UNetDecoder"
    assert config["schema"] == "prithvi-xtseg-v1"
    assert config["training"]["selection"].startswith("max validation macro-F1")

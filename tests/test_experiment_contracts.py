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


def test_staged_branch_is_separate_and_uninitialized_from_wce():
    config = load("p5_frozen_prithvi_wce_warmup_dmi_morphology.json")
    assert config["experiment_id"].startswith("P5_")
    assert config["initialization"]["WCE_checkpoint"] is None
    assert config["warmup"]["encoder_policy"] == "frozen_eval"
    assert config["warmup"]["objective"] == "weighted_cross_entropy"
    assert config["objective"]["name"] == "exact_dmi"
    assert config["model"]["morphology"]["name"] == "LogitAdditiveClosing2d"
    assert config["selection"]["primary"] == "validation_macro_F1_11_after_train_permutation"

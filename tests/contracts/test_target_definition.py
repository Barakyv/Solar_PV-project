"""docs/target_definition.md quotes the current decisions of stage 6 (configs/target.yaml,
configs/derived_constants.yaml, configs/panel_orientation.yaml)."""

import yaml

from solarpv.config.paths import CONFIGS_DIR, REPO_ROOT

DOC = REPO_ROOT / "docs" / "target_definition.md"


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))


def test_target_definition_matches_the_configuration():
    text = DOC.read_text(encoding="utf-8")
    target, derived, plane = (_yaml("target.yaml"), _yaml("derived_constants.yaml"),
                              _yaml("panel_orientation.yaml"))
    assert f"**{target['k_denominator'].upper()}_cs**" in text
    assert f"| {derived['mean_daytime_k']} |" in text
    assert f"| {target['training_min_theo_kw']:.0f} kW |" in text
    assert f"tilt {plane['tilt']:.0f}°, azimuth {plane['azimuth']:.0f}°" in text
    crit = target["criteria"]
    for den in ("ghi", "poa"):
        assert f"{crit[den]['val_nrmse_kw_pct']:.2f} ± {crit[den]['val_nrmse_kw_seed_std']:.2f} %" \
            in text

from __future__ import annotations

from pathlib import Path

from synthrain.config import load_scenario_config


def test_load_config_supports_observation_and_faults(tmp_path: Path):
    cfg_path = tmp_path / "config.ini"
    cfg_path.write_text(
        """
[io]
out = demo

[observation]
noise_mmph = 1.5
link_path_samples = 11

[faults]
outage_fraction = 0.2
bias_fraction = 0.1
""".strip(),
        encoding="utf-8",
    )

    cfg = load_scenario_config(cfg_path)

    assert cfg.io.out == "demo"
    assert cfg.observation.noise_mmph == 1.5
    assert cfg.observation.link_path_samples == 11
    assert cfg.observation.outage_fraction == 0.2
    assert cfg.observation.bias_fraction == 0.1

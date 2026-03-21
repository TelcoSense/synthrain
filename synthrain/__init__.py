from __future__ import annotations

__all__ = [
    "ScenarioConfig",
    "ScenarioResult",
    "FieldMetrics",
    "compute_field_metrics",
    "load_scenario_config",
    "run_scenario",
    "write_scenario_outputs",
]


def __getattr__(name: str):
    if name in {"ScenarioConfig", "load_scenario_config"}:
        from synthrain.config import ScenarioConfig, load_scenario_config

        return {
            "ScenarioConfig": ScenarioConfig,
            "load_scenario_config": load_scenario_config,
        }[name]
    if name in {"FieldMetrics", "compute_field_metrics"}:
        from synthrain.metrics import FieldMetrics, compute_field_metrics

        return {
            "FieldMetrics": FieldMetrics,
            "compute_field_metrics": compute_field_metrics,
        }[name]
    if name in {"ScenarioResult", "run_scenario"}:
        from synthrain.scenario import ScenarioResult, run_scenario

        return {
            "ScenarioResult": ScenarioResult,
            "run_scenario": run_scenario,
        }[name]
    if name == "write_scenario_outputs":
        from synthrain.outputs import write_scenario_outputs

        return write_scenario_outputs
    raise AttributeError(name)

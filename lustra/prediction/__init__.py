"""Fire spread prediction (Rothermel surface model + Huygens propagation).

Submodules:
- weather:     Open-Meteo client, wind reduction, fuel moisture estimation.
- fuel_model:  Scott and Burgan (2005) fuel-model catalog and drone-class map.
- terrain:     Open-Topo-Data SRTM client, slope/aspect via Horn (1981).
- rothermel:   SI adapter over pyretechnics' Rothermel implementation.
- propagation: Huygens wavelet (Finney 1998) and Richards (1990) normal march.
- pipeline:    PredictionEngine that ties everything together.
- validation:  Reference scenarios and Sorensen perimeter agreement metric.
"""

from lustra.prediction.fuel_model import (
    FUEL_MODEL_CATALOG,
    FuelModel,
    classify_drone_class,
    get_fuel_model,
    list_fuel_models,
)
from lustra.prediction.pipeline import PredictionEngine, build_dry_zone_fuel_classifier
from lustra.prediction.propagation import (
    PredictedPerimeter,
    PropagationConfig,
    propagate,
    propagate_geojson,
    propagate_huygens,
)
from lustra.prediction.rothermel import SpreadResult, compute_spread
from lustra.prediction.terrain import TerrainProvider, TerrainSample
from lustra.prediction.validation import (
    REFERENCE_SCENARIOS,
    ReferenceResult,
    ReferenceScenario,
    format_report,
    run_all_reference_scenarios,
    sorensen_index,
)
from lustra.prediction.weather import (
    WeatherProvider,
    WeatherSnapshot,
    midflame_wind_speed,
    one_hour_dead_fuel_moisture,
)

__all__ = [
    "FUEL_MODEL_CATALOG",
    "FuelModel",
    "PredictedPerimeter",
    "PredictionEngine",
    "PropagationConfig",
    "REFERENCE_SCENARIOS",
    "ReferenceResult",
    "ReferenceScenario",
    "SpreadResult",
    "TerrainProvider",
    "TerrainSample",
    "WeatherProvider",
    "WeatherSnapshot",
    "build_dry_zone_fuel_classifier",
    "classify_drone_class",
    "compute_spread",
    "format_report",
    "get_fuel_model",
    "list_fuel_models",
    "midflame_wind_speed",
    "one_hour_dead_fuel_moisture",
    "propagate",
    "propagate_geojson",
    "propagate_huygens",
    "run_all_reference_scenarios",
    "sorensen_index",
]

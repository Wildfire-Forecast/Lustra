"""Scott and Burgan (2005) standard fire behavior fuel models.

Defines :class:`FuelModel` and a curated subset of the 40 standard
fuel models. The actual Rothermel parameter values used at runtime
come from pyretechnics' built-in canonical table (keyed by the
:attr:`FuelModel.number`); the values in this catalog exist for
fuel-bed-depth lookups (wind reduction) and for documentation.

Reference: Scott, J. H. and Burgan, R. E. (2005). Standard fire
behavior fuel models. USDA Forest Service GTR RMRS-GTR-153.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


_TONS_PER_ACRE_TO_KG_PER_M2 = 0.2241702
_FT_TO_M = 0.3048
_PER_FT_TO_PER_M = 1.0 / _FT_TO_M
_BTU_PER_LB_TO_KJ_PER_KG = 2.326


@dataclass(frozen=True)
class FuelModel:
    code: str
    number: int
    name: str
    dynamic: bool
    fuel_bed_depth_ft: float
    dead_moisture_of_extinction_pct: float
    load_1hr_t_per_ac: float
    load_10hr_t_per_ac: float
    load_100hr_t_per_ac: float
    load_live_herb_t_per_ac: float
    load_live_woody_t_per_ac: float
    sav_1hr_per_ft: int
    sav_live_herb_per_ft: int
    sav_live_woody_per_ft: int
    heat_content_btu_per_lb: float = 8000.0

    @property
    def is_burnable(self) -> bool:
        return self.total_load_t_per_ac > 0.0 and self.fuel_bed_depth_ft > 0.0

    @property
    def total_load_t_per_ac(self) -> float:
        return (
            self.load_1hr_t_per_ac + self.load_10hr_t_per_ac + self.load_100hr_t_per_ac
            + self.load_live_herb_t_per_ac + self.load_live_woody_t_per_ac
        )

    @property
    def fuel_bed_depth_m(self) -> float:
        return self.fuel_bed_depth_ft * _FT_TO_M


FUEL_MODEL_CATALOG: Dict[str, FuelModel] = {
    "NB1": FuelModel("NB1", 91, "Urban/developed (nonburnable)", False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0),
    "GR1": FuelModel("GR1", 101, "Short, sparse dry climate grass", True, 0.4, 15.0, 0.10, 0.0, 0.0, 0.30, 0.0, 2200, 2000, 0),
    "GR2": FuelModel("GR2", 102, "Low load, dry climate grass", True, 1.0, 15.0, 0.10, 0.0, 0.0, 1.00, 0.0, 2000, 1800, 0),
    "GR4": FuelModel("GR4", 104, "Moderate load, dry climate grass", True, 2.0, 15.0, 0.25, 0.0, 0.0, 1.90, 0.0, 2000, 1800, 0),
    "GS2": FuelModel("GS2", 122, "Moderate load, dry climate grass-shrub", True, 1.5, 15.0, 0.50, 0.50, 0.0, 0.60, 1.00, 2000, 1800, 1800),
    "SH2": FuelModel("SH2", 142, "Moderate load, dry climate shrub", False, 1.0, 15.0, 1.35, 2.40, 0.75, 0.0, 3.85, 2000, 0, 1600),
    "SH5": FuelModel("SH5", 145, "High load, dry climate shrub", True, 6.0, 15.0, 3.60, 2.10, 0.0, 0.0, 2.90, 750, 0, 1600),
    "SH7": FuelModel("SH7", 147, "Very high load, dry climate shrub", False, 6.0, 15.0, 3.50, 5.30, 2.20, 0.0, 3.40, 750, 0, 1600),
    "TL3": FuelModel("TL3", 183, "Moderate load conifer litter", False, 0.3, 20.0, 0.50, 2.20, 2.80, 0.0, 0.0, 2000, 0, 0),
}

DEFAULT_DRONE_CLASS_TO_FUEL_CODE: Dict[str, str] = {
    "fire": "GR4",
    "dry_zone": "GR4",
    "dry_grass": "GR4",
    "dry_shrub": "SH2",
    "grass_shrub": "GS2",
    "timber_litter": "TL3",
    "nonburnable": "NB1",
    "barren": "NB1",
    "default": "GR1",
}


def get_fuel_model(code: str) -> FuelModel:
    key = code.strip().upper()
    if key not in FUEL_MODEL_CATALOG:
        raise KeyError(f"Fuel model {code!r} not in catalog. Available: {sorted(FUEL_MODEL_CATALOG)}")
    return FUEL_MODEL_CATALOG[key]


def list_fuel_models() -> Iterable[FuelModel]:
    return FUEL_MODEL_CATALOG.values()


def classify_drone_class(class_name: str, *, overrides: Optional[Dict[str, str]] = None) -> FuelModel:
    mapping = dict(DEFAULT_DRONE_CLASS_TO_FUEL_CODE)
    if overrides:
        mapping.update({k.lower(): v for k, v in overrides.items()})
    key = (class_name or "").strip().lower()
    return get_fuel_model(mapping.get(key) or mapping["default"])

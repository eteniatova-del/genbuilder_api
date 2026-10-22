from pathlib import Path
import contextvars
import contextlib

from typing import Dict, Any, Iterator, Literal, Optional
from pydantic import BaseModel, ConfigDict


class GenParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    INNER_BORDER: int = 3
    """width of inner plot border restricted for construction"""
    MAX_COVERAGE: float = 0.9
    """control parameter for building generation"""
    rectangle_finder_step: int = 5
    """parameter for maximum rectangle inscription, lower - higher quality and lower speed"""
    minimal_rectangle_side: int = 40
    """minimal side length for rectangle segments"""
    jobs_number: int = 5
    """jobs number for MIR"""
    la_per_person: int = 18
    """normative square meters for person, used in service deman calculation"""
    max_service_attempts: int = 200
    """limit for attempts for service placement"""
    max_sites_per_service_per_block: int = 10
    """limit for number of service of one type in block"""
    physical_objects_exclusion_buffer_m: float = 5.0
    """Buffer (in meters) around physical objects that will be excluded from generation."""
    physical_objects_exclusion_dynamic: bool = True
    """If True, buffer is chosen per physical object based on its geometry size and building_params."""
    physical_objects_exclusion_min_buffer_m: float = 3.0
    """Lower bound for dynamic buffer (in meters)."""
    physical_objects_exclusion_max_buffer_m: float = 60.0
    """Upper bound for dynamic buffer (in meters)."""
    landscape_enabled: bool = False
    """If True, greenery (lawn, trees) and surface parking are computed after buildings and services (app/logic/landscape.py)."""
    housing_class: Literal["standard", "business"] = "standard"
    """Housing comfort class (SP 42.13330.2026, tables 6.1 and 8.16): drives total flat area per resident and parking per apartment."""
    persons_per_apartment: Optional[float] = None
    """Average residents per apartment; if None, permanent parking is 350 cars per 1000 residents (SP 42, 8.113 note 2)."""
    parking_type: Literal["underground", "surface", "mixed"] = "mixed"
    """Parking type for permanent places, chosen by the user: underground only, surface only, or mixed (minimal underground share)."""
    tree_crown_d: float = 15.0
    """Mature crown diameter (m) of the planted tree; assumption, not a norm (Acer platanoides: 15-20 m)."""
    service_projects_file: str = str( Path(__file__).resolve().parent / "service_projects.geojson")
    """path to service projects file with geometry and plot/building parameters"""
    seed: Optional[int] = None
    """random seed for service placement; None - non-deterministic"""

    def patched(self, patch: Dict[str, Any]) -> "GenParams":
        def deep_merge(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                c = dict(a)
                for k, v in b.items():
                    c[k] = deep_merge(c.get(k), v)
                return c
            return b if b is not None else a

        data = self.model_dump()
        merged = deep_merge(data, patch)
        return self.__class__.model_validate(merged)


class ParamsProvider:
    def __init__(self, base: GenParams):
        self._var: contextvars.ContextVar[GenParams] = contextvars.ContextVar(
            "gen_params", default=base
        )

    def current(self) -> GenParams:
        return self._var.get()

    @contextlib.contextmanager
    def override(self, new_params: GenParams) -> Iterator[None]:
        token = self._var.set(new_params)
        try:
            yield
        finally:
            self._var.reset(token)

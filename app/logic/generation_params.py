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
    roads_enabled: bool = False
    """If True (requires landscape_enabled), intra-block roads are generated (app/logic/roads.py) and parking/greenery are placed around them."""
    floor_height_m: float = 3.0
    """Assumed storey height (m) used to convert floors to building height for fire-access road width; not a norm."""
    tree_crown_d: float = 15.0
    """Mature crown diameter (m) of the planted tree; assumption, not a norm (Acer platanoides: 15-20 m)."""
    road_style: Literal["minimal", "loop"] = "minimal"
    """Intra-block road scheme: minimal = shortest tree of fire access roads, dead ends only up to 150 m (SP 42, 8.49); loop = every dead end is closed by a second exit to the street (looped scheme, methodical guideline, secondary source)."""
    road_turn_penalty_m: float = 20.0
    """Assumption (not a norm): penalty for a 90-degree turn of a road, metres of equivalent length; larger value gives straighter roads."""
    road_junction_penalty_m: float = 10.0
    """Assumption (not a norm): penalty for a new junction, metres of equivalent length; larger value gives fewer branches."""
    road_reuse_cost: float = 0.15
    """Assumption (not a norm): relative cost of using an already built road, 0..1; smaller value gives more shared roads."""
    parking_reach_m: float = 12.0
    """Assumption (not a norm): a parking patch farther than this from a road gets its own access road; nearer patches use the existing road."""
    parking_max_lot_places: int = 50
    """Assumption (not a norm): a surface lot is split into lots of at most this number of places; the window setback is taken by lot size."""
    norm_mode: Literal["extended", "strict"] = "extended"
    """Reporting mode: extended = checks based on secondary sources and assumptions count as passed; strict = only checks based on a verified SP 42 text count as passed, the rest are reported as unverified."""
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

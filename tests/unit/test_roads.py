"""Intra-block roads: reachability, normative widths, wall gap, dead ends, turnarounds."""
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from app.logic.landscape import LandscapeParams, generate_landscape
from app.logic.roads import RoadParams, fire_width_for, generate_roads

BLOCK = box(0, 0, 400, 300)
HOUSES = [box(60, 60, 160, 78), box(220, 60, 320, 78), box(60, 180, 160, 198), box(220, 180, 320, 198)]


def _union(res):
    return unary_union([c["geometry"] for c in res["carriageways"]])


def test_fire_width_by_height_thresholds():
    p = RoadParams()
    assert fire_width_for(12.9, p) == 3.5
    assert fire_width_for(13.1, p) == 4.2
    assert fire_width_for(46.0, p) == 4.2
    assert fire_width_for(46.1, p) == 6.0


def test_every_building_is_reachable_and_has_a_road_near_the_wall():
    res = generate_roads(BLOCK, HOUSES, [5] * 4)
    assert res["diagnostics"]["unreachable"] == []
    roads = _union(res)
    for h in HOUSES:
        d = h.distance(roads)
        assert d <= 10.5  # внутренний край проезда не дальше 10 м (верх диапазона 8-10 м)


def test_road_edge_keeps_min_gap_to_walls():
    res = generate_roads(BLOCK, HOUSES, [8] * 4)
    roads = _union(res)
    for h in HOUSES:
        assert h.distance(roads) >= 5.0 - 1e-6  # СП 4.13130 (вторичный): не ближе 5 м


def test_width_grows_with_building_height():
    low = generate_roads(BLOCK, HOUSES, [4] * 4)       # 12 м -> 3,5 м
    high = generate_roads(BLOCK, HOUSES, [17] * 4)     # 51 м -> 6,0 м
    assert min(c["width"] for c in low["carriageways"]) == 3.5
    assert max(c["width"] for c in high["carriageways"]) == 6.0


def test_turnaround_is_placed_at_dead_ends():
    res = generate_roads(BLOCK, HOUSES, [8] * 4)
    assert res["diagnostics"]["turnarounds"] >= 1
    assert all(t.area > 0 for t in res["turnarounds"])


def test_education_service_gets_two_accesses():
    plain = generate_roads(BLOCK, [box(150, 120, 250, 138)], [3])
    school = generate_roads(BLOCK, [box(150, 120, 250, 138)], [3], ["Школа"])
    assert school["diagnostics"]["road_length_m"] > plain["diagnostics"]["road_length_m"]


def test_empty_input_is_safe():
    assert generate_roads(BLOCK, [], [])["carriageways"] == []


def test_landscape_with_roads_keeps_parking_and_trees_off_the_road():
    roads = generate_roads(BLOCK, HOUSES, [8] * 4)
    res = generate_landscape(BLOCK, HOUSES, 600, LandscapeParams(parking_type="surface"), roads=roads)
    ru = unary_union([c["geometry"] for c in roads["carriageways"]] + list(roads["turnarounds"]))
    assert res["roads"] is not None and res["roads"]["diagnostics"]["buildings"] == len(HOUSES)
    for c in res["parking_cells"]:
        assert c.intersection(ru).area < 1e-6
    for t in res["trees"]:
        assert t.distance(ru) >= 2.0 - 1e-6   # СП 42 табл. 6.3: 2,0 м от проезжей части

"""Layout checks must catch the defects they exist for (missing roads, overlaps) and pass on a clean layout."""
from shapely.geometry import box

from app.logic.landscape import LandscapeParams, generate_landscape
from app.logic.layout_checks import check_layout
from app.logic.roads import RoadParams, generate_roads

BLOCK = box(0, 0, 400, 300)
HOUSES = [box(60, 60, 160, 78), box(220, 60, 320, 78), box(60, 180, 160, 198), box(220, 180, 320, 198)]
P = LandscapeParams(parking_type="surface", roads_enabled=True)


def _run(houses=HOUSES, floors=None):
    floors = floors or [8] * len(houses)
    roads = generate_roads(BLOCK, houses, floors)
    inputs = {"floors": floors, "services": None, "params": RoadParams()}
    res = generate_landscape(BLOCK, houses, 600, P, roads=roads, road_inputs=inputs)
    return res, floors


def _by_id(checks):
    return {c["id"]: c for c in checks}


def test_clean_layout_passes_road_and_parking_checks():
    res, floors = _run()
    checks = _by_id(check_layout(BLOCK, HOUSES, floors, None, res["roads"], res, P))
    for cid in ("road_access_distance", "road_width_fire", "dead_end_limit", "roads_no_overlap",
                "parking_window_setback", "parking_balance", "trees_buffers"):
        assert checks[cid]["status"] == "pass", (cid, checks[cid]["violations"])


def test_missing_road_is_detected():
    res, floors = _run()
    empty = {"carriageways": [], "turnarounds": [], "diagnostics": {"dead_end_violations": [], "turnarounds": 0}}
    checks = _by_id(check_layout(BLOCK, HOUSES, floors, None, empty, res, P))
    assert checks["road_access_distance"]["status"] == "fail"
    assert checks["road_access_distance"]["violations_count"] == len(HOUSES)


def test_parking_on_a_road_is_detected():
    res, floors = _run()
    roads = res["roads"]
    road_geom = roads["carriageways"][0]["geometry"]
    res2 = dict(res)
    res2["parking_patches"] = res["parking_patches"] + [{"house_index": 0, "kind": "surface", "places": 10, "geometry": road_geom.buffer(1)}]
    checks = _by_id(check_layout(BLOCK, HOUSES, floors, None, roads, res2, P))
    assert checks["roads_no_overlap"]["status"] == "fail"

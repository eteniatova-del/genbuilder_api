"""СП 4.13130 пп. 8.1.9, 8.1.10, 8.1.12, 8.1.15: универсальные проверки."""
from shapely.geometry import box

from app.logic.fire_checks import continuous_fronts, find_courtyards, long_side
from app.logic.layout_checks import check_layout
from app.logic.landscape import LandscapeParams, generate_landscape
from app.logic.roads import RoadParams, generate_roads

BLOCK = box(0, 0, 500, 400)


def _checks(houses, floors, services=None):
    lp = LandscapeParams(parking_type="surface", roads_enabled=True)
    roads = generate_roads(BLOCK, houses, floors, services)
    res = generate_landscape(BLOCK, houses, 500, lp, roads=roads, road_inputs={"floors": floors, "services": services, "params": RoadParams()})
    return {c["id"]: c for c in check_layout(BLOCK, houses, floors, services, res["roads"], res, lp)}


def test_u_shaped_group_is_a_semi_closed_courtyard_and_parallel_slabs_are_not():
    u = [box(100, 100, 120, 200), box(100, 100, 158, 120), box(138, 100, 158, 200)]       # П-образная группа
    assert len(find_courtyards(u)) == 1
    slabs = [box(100, 100, 200, 118), box(100, 133, 200, 151)]                              # две параллельные секции, улица между ними
    assert find_courtyards(slabs) == []
    ring = [box(100, 100, 200, 120), box(100, 180, 200, 200), box(100, 100, 120, 200), box(180, 100, 200, 200)]
    assert len(find_courtyards(ring)) == 1                                                    # замкнутый двор


def test_continuous_front_and_long_side():
    a, b = box(0, 0, 200, 18), box(202, 0, 400, 18)                  # зазор 2 м < 3,5 м: один фронт
    fr = continuous_fronts([a, b])
    assert len(fr) == 1 and fr[0][1] > 300
    assert len(continuous_fronts([a, box(210, 0, 400, 18)])) == 2     # зазор 10 м: проход есть
    assert abs(long_side(box(0, 0, 114, 20)) - 114) < 1e-6


def test_checks_report_long_building_courtyard_and_low_rise_reach():
    long_house = box(60, 160, 180, 178)                               # 120 м > 100 м
    far_low = box(420, 330, 440, 345)                                # дом до 3 этажей
    ch = _checks([long_house, far_low], [9, 3])
    assert ch["fire_passages_100m"]["violations_count"] == 1 and ch["fire_passages_100m"]["status"] == "warn"
    assert ch["fire_low_rise_50m"]["checked"] == 1 and ch["fire_low_rise_50m"]["status"] == "pass"
    assert ch["fire_courtyard_access"]["checked"] == 0

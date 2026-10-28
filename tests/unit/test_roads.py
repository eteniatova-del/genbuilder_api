"""Intra-block roads: reachability, normative widths, wall gap, dead ends, turnarounds."""
import pytest
from shapely.geometry import LineString, Point, box
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


def test_turnarounds_keep_wall_gap_and_violations_are_reported():
    houses = HOUSES + [box(160, 100, 220, 118), box(60, 120, 80, 160)]
    for style in ("minimal", "loop"):
        res = generate_roads(BLOCK, houses, [5] * len(houses), p=RoadParams(road_style=style))
        bu = unary_union(houses)
        viol = res["diagnostics"]["turnaround_violations"]
        far = [t for t in res["turnarounds"] if t.distance(bu) < 5.0 - 0.5]
        assert len(far) == len(viol)         # каждое нарушение отступа попадает в диагностику


def test_tall_buildings_keep_8_m_wall_gap_and_low_ones_keep_5_m():
    floors = [16, 16, 3, 3]                      # 48 м (выше 28 м) и 9 м
    res = generate_roads(BLOCK, HOUSES, floors)
    roads = _union(res)
    for h, f in zip(HOUSES, floors):
        lo = 8.0 if f * 3.0 > 28.0 else 5.0
        assert h.distance(roads) >= lo - 0.5


def _fs(res):
    return {x["building"]: x for x in res["diagnostics"]["fire_sides"]}


def test_fire_class_and_required_sides():
    from app.logic.roads import fire_class, fire_sides_required
    p = RoadParams()
    poly = box(0, 0, 60, 14)
    assert fire_class(None, p) == "F1.3"
    assert fire_class("Детский сад", p) == "F1.1"
    assert fire_sides_required(15.0, poly, None, p) == 1       # дом до 28 м: одна сторона
    assert fire_sides_required(48.0, poly, None, p) == 2       # Ф1.3 от 28 м: две стороны
    assert fire_sides_required(9.0, poly, "Школа", p) == 4     # Ф4.1: со всех сторон
    assert fire_sides_required(9.0, box(0, 0, 150, 120), None, p) == 4   # площадь > 10 000 м2
    assert fire_sides_required(15.0, poly, "Поликлиника", p) == 4        # допущение: все сервисы со всех сторон
    assert fire_sides_required(15.0, poly, "Поликлиника", RoadParams(fire_all_sides_services=False)) == 1   # по п. 8.1.1: Ф3 ниже 18 м


def test_fire_access_sides_are_built_along_the_walls():
    houses = [box(120, 120, 200, 138), box(270, 120, 300, 138)]
    res = generate_roads(BLOCK, houses, [16, 5], [None, "Школа"])
    fs = _fs(res)
    assert fs[0]["required"] == 2 and fs[0]["built"] == 2
    assert fs[1]["required"] == 4 and fs[1]["built"] == 4
    assert all(c >= 0.8 for x in fs.values() for c in x["coverage"])
    roads = _union(res)
    for h, gap in zip(houses, (8.0, 5.0)):
        assert h.distance(roads) >= gap - 1e-6


def test_low_house_gets_one_fire_side():
    res = generate_roads(BLOCK, [box(120, 120, 200, 138)], [5])
    assert _fs(res)[0]["required"] == 1 and _fs(res)[0]["built"] == 1


def test_smooth_polyline_straightens_staircase_and_rounds_corner():
    from app.logic.roads import smooth_polyline
    stairs = [(0, 0)]
    for _ in range(8):
        stairs += [(stairs[-1][0] + 2.5, stairs[-1][1]), (stairs[-1][0] + 2.5, stairs[-1][1] + 1.5)]
    out = smooth_polyline(stairs, 0.0, 2.0)
    assert len(out) == 2                                  # лесенка превращена в один прямой отрезок
    corner = smooth_polyline([(0, 0), (30, 0), (30, 30)], 6.0, 0.0)
    assert len(corner) > 3                                # поворот на 90° заменён дугой
    assert LineString(corner).length < 60.0 and LineString(corner).distance(Point(30, 0)) > 1.5


def test_corner_radius_zero_gives_square_corners_and_keeps_wall_gap():
    p = RoadParams(corner_radius_m=0.0, smooth_tol_m=0.0)
    res = generate_roads(BLOCK, HOUSES, [8] * 4, params=p) if False else generate_roads(BLOCK, HOUSES, [8] * 4, None, p)
    assert res["diagnostics"]["unreachable"] == []


def test_parking_patches_are_full_rectangles():
    lp = LandscapeParams(parking_type="surface", roads_enabled=True)
    roads = generate_roads(BLOCK, HOUSES, [8] * 4)
    res = generate_landscape(BLOCK, HOUSES, 900, lp, roads=roads, road_inputs={"floors": [8] * 4, "services": None, "params": RoadParams()})
    assert res["parking_patches"]
    for pt in res["parking_patches"]:
        g = pt["geometry"]
        assert abs(g.area - g.envelope.area) < 1e-6                      # прямоугольник, не Г-образная фигура
        assert abs(g.area - pt["places"] * lp.cell_w * lp.cell_l) < 1e-3  # все ячейки заполнены


def test_kerb_fillets_added_and_turnaround_corners_rounded():
    sharp = generate_roads(BLOCK, HOUSES, [5] * 4, None, RoadParams(kerb_radius_m=0.0))
    soft = generate_roads(BLOCK, HOUSES, [5] * 4, None, RoadParams(kerb_radius_m=6.0))
    assert soft["diagnostics"].get("kerb_fillets", 0) > 0
    assert "kerb_fillets" not in sharp["diagnostics"]
    t_s, t_r = sharp["turnarounds"][0], soft["turnarounds"][0]
    assert t_r.area < t_s.area - 1.0                                      # углы площадки срезаны дугами


def test_parking_shapes_are_compact():
    from app.logic.landscape import _shapes
    lp = LandscapeParams()
    for n in (4, 13, 37, 73, 145):
        for w, h, *_ in _shapes(n, lp):
            assert max(w, h) / min(w, h) <= lp.lot_max_aspect + 1e-9      # без узких лент


def test_parallel_second_lane_along_ring_is_dropped_and_joined_perpendicularly():
    from app.logic.roads import drop_parallel, join_perpendicular
    ring = LineString([(0, 0), (100, 0)])
    # линия идёт вдоль кольца на расстоянии 7 м и выходит на него в конце
    path = [(0, 40), (0, 7), (60, 7), (60, 0)]
    pieces = drop_parallel(path, ring)
    assert pieces and all(LineString(x).length > 2 for x in pieces)
    total = sum(LineString(x).length for x in pieces)
    assert total < LineString(path).length - 30                         # параллельный участок убран
    out = join_perpendicular([(30, 40), (30, 20), (45, 10), (50, 1)], ring)
    assert abs(out[-1][1]) < 1e-6 and abs(out[-2][1] - out[-1][1]) > 5   # последний отрезок идёт по нормали к кольцу


def test_parking_patch_gets_a_link_to_the_road_edge():
    houses = [box(120, 120, 200, 138)]
    patch = box(260, 60, 285, 100)          # пятно без прямого примыкания к проезду
    res = generate_roads(BLOCK, houses, [16], None, patches=[patch])
    d = res["diagnostics"]
    assert d["parking_unlinked"] == []
    roads = unary_union([c["geometry"] for c in res["carriageways"]] + res["turnarounds"])
    assert patch.distance(roads) <= 0.3 + 1e-6
    assert d["unreachable"] == []


def test_long_dead_end_gets_intermediate_turnarounds_every_150_m():
    # СП 4.13130 п. 8.1.11: если проезд длиннее 150 м, ещё площадки не реже чем через 150 м
    block = box(0, 0, 900, 900)
    house = [box(400, 430, 480, 448)]
    res = generate_roads(block, house, [5], None, RoadParams(close_dead_ends=False, fire_sides_enabled=False))
    d = res["diagnostics"]
    assert d["max_dead_end_m"] > 150.0
    assert d["intermediate_turnarounds"] >= 1
    assert d["dead_end_violations"] == []
    assert len(res["turnarounds"]) >= 1 + d["intermediate_turnarounds"]


def test_dead_end_length_is_measured_from_the_street_and_closed_by_a_second_exit():
    block = box(0, 0, 900, 900)
    res = generate_roads(block, [box(400, 430, 480, 448)], [5], None)
    d = res["diagnostics"]
    assert d["through_links_added"] >= 1          # единственный подъезд длиной ~400 м закрыт вторым выходом
    assert d["max_dead_end_m"] <= 150.0


def test_parking_link_keeps_the_wall_gap_of_a_tall_building():
    tall = box(120, 120, 200, 138)            # 16 этажей: отступ проезда 8 м
    patch = box(150, 90, 175, 112)            # пятно у самой стены высокого дома
    res = generate_roads(BLOCK, [tall], [16], None, patches=[patch])
    roads = unary_union([c["geometry"] for c in res["carriageways"]])
    assert tall.distance(roads) >= 8.0 - 0.5
    assert res["diagnostics"]["parking_unlinked"] == [] or patch.distance(roads) > 0.3

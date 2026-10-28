"""Единый список предупреждений собирается из полей ответа."""
from app.logic.result_warnings import build_warnings


def test_empty_when_everything_fits():
    cap = {"status": "ok"}
    srv = {"status": "ok"}
    blocks = [{"block_index": 0, "green_deficit_m2": 0.0, "parking_deficit_surface": 0,
               "roads": {"unreachable": [], "turnaround_violations": [], "dead_end_violations": []},
               "checks": [{"id": "x", "status": "pass"}]}]
    assert build_warnings(cap, srv, blocks) == []


def test_reports_shortage_of_residents_services_parking_green_and_roads():
    cap = {"status": "insufficient_area", "warning": "мало места", "residents_requested": 15000, "residents_placed": 9000}
    srv = {"status": "partial", "warning": "не все сервисы", "unplaced_by_reason": {"no_space": ["Школа"]}, "capacity_unplaced": 698}
    blocks = [{"block_index": 1, "green_deficit_m2": 120.0, "parking_deficit_surface": 30,
               "roads": {"unreachable": [3], "turnaround_violations": [{}], "dead_end_violations": []},
               "checks": [{"id": "road_access_distance", "title": "Проезд у здания", "status": "fail", "violations_count": 2}]}]
    codes = {w["code"] for w in build_warnings(cap, srv, blocks)}
    assert {"residents_not_placed", "services_not_placed", "green_deficit", "parking_deficit",
            "buildings_without_road", "turnaround_too_close", "check_road_access_distance"} <= codes
    school = [w for w in build_warnings(cap, srv, blocks) if w["code"] == "services_not_placed"][0]
    assert "Школа" in school["message"] and school["severity"] == "error"


def test_handles_missing_inputs():
    assert build_warnings(None, None, None) == []


def test_unplaced_parking_is_reported_per_house_and_unlinked_patches_and_blockers():
    blocks = [{"block_index": 0, "green_deficit_m2": 0.0, "parking_deficit_surface": 12,
               "unplaced": [{"object": "parking_permanent", "house_index": 3, "required": 40, "placed": 30, "short": 10},
                            {"object": "parking_guest", "house_index": 3, "required": 5, "placed": 3, "short": 2}],
               "roads": {"unreachable": [], "turnaround_violations": [], "dead_end_violations": [],
                         "parking_unlinked": [{"patch": 2, "gap_m": 7.5}]},
               "checks": [{"id": "fire_access_sides", "title": "Подъезд", "status": "fail", "violations_count": 1,
                           "violations": [{"building": 7, "coverage": [1.0, 0.7],
                                           "blocked_by": [{"building": 12, "distance_m": 13.3}]}]}]}]
    ws = {w["code"]: w for w in build_warnings(None, None, blocks)}
    assert "дом 3" in ws["parking_deficit"]["message"] and "постоянные 30 из 40" in ws["parking_deficit"]["message"]
    assert "гостевые 3 из 5" in ws["parking_deficit"]["message"]
    assert ws["parking_not_linked"]["severity"] == "error"
    assert "мешает здание 12 (13.3 м)" in ws["check_fire_access_sides"]["message"]

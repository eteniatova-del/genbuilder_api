"""Территориальная доступность сервисов по прямой (СП 42, прил. Л, табл. Л.1)."""
from shapely.geometry import box, mapping

from app.logic.result_warnings import build_warnings
from app.logic.service_access import check_service_access

LON, LAT = 30.3, 59.9
M_LAT = 1 / 111_000
M_LON = M_LAT / 0.5          # на 59.9° долгота ≈ 0,5 от широты


def _b(x0, y0, service=None, w=20, h=20):
    g = box(LON + x0 * M_LON, LAT + y0 * M_LAT, LON + (x0 + w) * M_LON, LAT + (y0 + h) * M_LAT)
    return {"type": "Feature", "properties": {"service": service}, "geometry": mapping(g)}


def test_nearby_services_pass():
    fc = {"type": "FeatureCollection", "features": [_b(0, 0), _b(300, 0, "Детский сад"), _b(0, 300, "Школа"), _b(600, 0, "Поликлиника")]}
    r = check_service_access(fc)
    assert r["violations"] == []
    assert r["services"]["Детский сад"]["limit_m"] == 500.0 and r["services"]["Поликлиника"]["limit_m"] == 1000.0


def test_far_home_is_reported_with_its_limit():
    fc = {"type": "FeatureCollection", "features": [_b(0, 0), _b(900, 0), _b(600, 0, "Школа")]}
    r = check_service_access(fc)
    v = r["violations"]
    assert [x["service"] for x in v] == ["Школа"] and v[0]["building_index"] == 0 and v[0]["distance_m"] > 500


def test_constrained_conditions_use_800_m():
    fc = {"type": "FeatureCollection", "features": [_b(0, 0), _b(0, 0), _b(650, 0, "Детский сад")]}
    assert check_service_access(fc)["violations"]
    assert check_service_access(fc, constrained=True)["violations"] == []


def test_missing_service_and_empty_input_are_safe():
    assert check_service_access(None)["violations"] == []
    assert check_service_access({"features": [_b(0, 0)]})["services"] == {}


def test_violations_become_a_warning():
    acc = {"violations": [{"service": "Школа", "building_index": 1, "distance_m": 620.0, "limit_m": 500.0}]}
    ws = build_warnings(None, {"access": acc}, None)
    assert ws[0]["code"] == "service_access_distance" and "Школа" in ws[0]["message"] and "по прямой" in ws[0]["message"]

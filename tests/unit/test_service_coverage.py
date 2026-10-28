"""Достройка сервисов, если часть жилых домов дальше нормативной доступности (СП 42, прил. Л, табл. Л.1)."""
import asyncio

import geopandas as gpd
from shapely.geometry import box

from app.logic.service_access import SERVICE_ACCESS_M
from tests.unit.test_service_placement import CRS, _generator, _normatives, _projects

LENGTH = 1400.0


def _run(project_capacity=100.0, normative=61.0, homes=None):
    homes = homes or [box(5, 90, 25, 110), box(LENGTH - 25, 90, LENGTH - 5, 110)]
    block = gpd.GeoDataFrame([{"zone": "residential", "geometry": box(0, 0, LENGTH, 200)}], geometry="geometry", crs=CRS)
    plots = gpd.GeoDataFrame([{"src_index": 0, "geometry": box(0, 0, LENGTH, 200)}], geometry="geometry", crs=CRS)
    bld = gpd.GeoDataFrame([{"src_index": 0, "living_area": 18_000.0, "geometry": g} for g in homes], geometry="geometry", crs=CRS)
    gen = _generator(max_sites_per_service_per_block=1)
    projects = _projects("Детский сад", capacity=project_capacity, plot_size=30.0)
    gen.load_service_projects = lambda: projects
    res = asyncio.run(gen.generate_services(block, plots, bld, _normatives("Детский сад", normative), CRS))
    return res, bld


def test_far_homes_get_an_additional_service_within_radius():
    res, bld = _run()
    assert len(res) >= 2 and res.attrs["service_diagnostics"]["coverage_services_added"] >= 1
    r = SERVICE_ACCESS_M["Детский сад"]
    for h in bld.geometry:
        assert res.geometry.distance(h).min() <= r + 1e-6
    assert res.attrs["service_diagnostics"]["access_unserved"] == []


def test_added_service_capacity_covers_the_demand_of_its_homes():
    res, _ = _run(project_capacity=100.0, normative=61.0)
    # норматив 61 на 1000 жителей: один дом (≈1000 жителей по la_per_person) требует не более 100 мест
    assert (res["capacity"] >= 61.0).all()


"""Greenery and parking module: demand formulas, underground switch, tree buffers."""
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from app.logic.landscape import (
    DISCLAIMER,
    LandscapeParams,
    apply_landscape,
    demand,
    generate_landscape,
)

BLOCK = box(0, 0, 200, 150)
HOUSES = [box(20, 20, 80, 35), box(100, 20, 160, 35), box(20, 70, 80, 85), box(100, 70, 160, 85)]


def test_demand_uses_residents_only_without_apartments():
    d = demand(1000, LandscapeParams())
    assert d["permanent_places"] == 350  # SP 42, 8.113 note 2
    assert d["guest_places"] == 30  # SP 42, 8.117
    assert d["green_required_m2"] == pytest.approx(3000.0)  # SP 476, table 7.1
    assert d["flat_area_m2"] == pytest.approx(30000.0)  # SP 42, table 6.1, standard


def test_business_class_changes_flat_area_and_apartment_based_parking():
    p = LandscapeParams(housing_class="business", persons_per_apartment=2.5)
    d = demand(1000, p)
    assert d["flat_area_m2"] == pytest.approx(40000.0)
    assert d["permanent_places"] == 800  # 400 apartments x 2.0, SP 42, table 8.16
    assert d["permanent_basis"] == "СП 42 табл. 8.16"


def test_standard_class_uses_1_2_places_per_apartment():
    d = demand(1000, LandscapeParams(persons_per_apartment=2.5))
    assert d["permanent_places"] == 480  # 400 x 1.2


def test_trees_keep_to_wall_and_parking_buffers():
    res = generate_landscape(BLOCK, HOUSES, 600, LandscapeParams())
    houses = [h for h in HOUSES]
    wall = res["diagnostics"]["tree_wall_buffer_m"]
    assert res["trees"], "expected some trees on a spacious block"
    for t in res["trees"]:
        assert all(t.distance(h) >= wall - 1e-6 for h in houses)
        assert all(t.distance(c) >= 0.7 - 1e-6 for c in res["parking_cells"])


def test_crown_wider_than_5m_enlarges_wall_buffer():
    narrow = generate_landscape(BLOCK, HOUSES, 300, LandscapeParams(tree_crown_d=4.0))
    wide = generate_landscape(BLOCK, HOUSES, 300, LandscapeParams(tree_crown_d=20.0))
    assert narrow["diagnostics"]["tree_wall_buffer_m"] == pytest.approx(5.0)
    assert wide["diagnostics"]["tree_wall_buffer_m"] == pytest.approx(10.0)


def test_parking_moves_underground_when_surface_is_too_tight():
    small_block = box(0, 0, 90, 60)
    houses = [box(10, 10, 80, 25)]
    res = generate_landscape(small_block, houses, 600, LandscapeParams())
    d = res["diagnostics"]
    assert d["underground_share"] > 0
    assert d["permanent_underground"] + d["permanent_surface_placed"] == d["permanent_places"] or d[
        "parking_deficit_surface"
    ] > 0


def test_diagnostics_carry_the_sketch_disclaimer():
    d = generate_landscape(BLOCK, HOUSES, 300, LandscapeParams())["diagnostics"]
    assert d["disclaimer"] == DISCLAIMER


def test_apply_landscape_works_per_block_and_skips_blocks_without_residents():
    blocks = gpd.GeoDataFrame(
        {"zone": ["residential", "residential"]},
        geometry=[BLOCK, box(300, 0, 500, 150)],
        crs=32636,
    )
    houses = gpd.GeoDataFrame(
        {"living_area": [9000.0] * len(HOUSES)},
        geometry=[h for h in HOUSES],
        crs=32636,
    )
    out = apply_landscape(blocks, houses, 18.0, LandscapeParams())
    assert len(out["diagnostics"]) == 1  # the second block has no buildings
    assert out["diagnostics"][0]["residents"] == 2000
    assert set(out["greenery"]["kind"]) <= {"lawn", "tree", "shrub"}
    assert set(out["parking"]["kind"]) <= {"surface", "guest", "underground"}
    assert out["greenery"].crs == blocks.crs


def test_empty_result_is_still_a_geodataframe():
    blocks = gpd.GeoDataFrame({"zone": ["residential"]}, geometry=[BLOCK], crs=32636)
    houses = gpd.GeoDataFrame({"living_area": []}, geometry=[], crs=32636)
    out = apply_landscape(blocks, houses, 18.0, LandscapeParams())
    assert out["diagnostics"] == []
    assert len(out["greenery"]) == 0 and len(out["parking"]) == 0


def test_each_house_gets_its_own_compact_parking_patch():
    res = generate_landscape(BLOCK, HOUSES, 400, LandscapeParams(parking_type="surface"))
    d = res["diagnostics"]
    assert len(d["houses"]) == len(HOUSES)
    assert all(h["permanent_surface"] > 0 and h["guest_places"] > 0 for h in d["houses"])
    by_house = {}
    for pt in res["parking_patches"]:
        by_house.setdefault(pt["house_index"], []).append(pt)
    assert set(by_house) == set(range(len(HOUSES)))


def test_parking_type_underground_has_only_guest_on_surface():
    res = generate_landscape(BLOCK, HOUSES, 400, LandscapeParams(parking_type="underground"))
    d = res["diagnostics"]
    assert d["permanent_surface_required"] == 0
    assert d["permanent_underground"] == d["permanent_places"]
    assert {pt["kind"] for pt in res["parking_patches"]} == {"guest"}


def test_parking_type_surface_reports_deficit_instead_of_going_underground():
    small_block = box(0, 0, 90, 60)
    res = generate_landscape(small_block, [box(10, 10, 80, 25)], 600, LandscapeParams(parking_type="surface"))
    d = res["diagnostics"]
    assert d["underground_share"] == 0
    assert d["parking_deficit_surface"] > 0
    items = d["unplaced"]                                         # что именно не разместилось: дом, сколько нужно и сколько размещено
    assert items and all(u["short"] > 0 and u["required"] > u["placed"] for u in items)
    assert sum(u["short"] for u in items if u["object"].startswith("parking")) == d["parking_deficit_surface"]
    assert {"object", "required", "placed", "short", "reason"} <= set(items[0])


def test_unplaced_is_empty_when_everything_fits():
    res = generate_landscape(BLOCK, HOUSES, 600, LandscapeParams(parking_type="mixed"))
    assert res["diagnostics"]["unplaced"] == []


def test_apply_landscape_returns_parking_attributes_per_building():
    blocks = gpd.GeoDataFrame({"zone": ["residential"]}, geometry=[BLOCK], crs=32636)
    houses = gpd.GeoDataFrame({"living_area": [9000.0] * len(HOUSES)}, geometry=[h for h in HOUSES], crs=32636)
    out = apply_landscape(blocks, houses, 18.0, LandscapeParams(parking_type="mixed"))
    attrs = out["buildings"]
    assert set(attrs.index) <= set(houses.index)
    assert {"parking_class", "has_underground_parking", "underground_parking_places",
            "surface_parking_places", "guest_parking_places"} <= set(attrs.columns)
    assert (attrs["has_underground_parking"] == (attrs["underground_parking_places"] > 0)).all()
    out_u = apply_landscape(blocks, houses, 18.0, LandscapeParams(parking_type="underground"))
    assert out_u["buildings"]["has_underground_parking"].all()
    assert set(out_u["buildings"]["parking_class"]) == {"underground"}

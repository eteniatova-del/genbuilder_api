"""Управление деревьями: плотность посадки (tree_density) и зоны (tree_zones)."""
from shapely.geometry import box

from app.logic.landscape import LandscapeParams, _place_trees

LAWN = box(0, 0, 200, 150)
HOUSES = box(20, 20, 80, 35).union(box(100, 20, 160, 35))
ROAD = box(0, 60, 200, 66)
PARK = box(170, 100, 190, 120)


def _trees(**kw):
    p = LandscapeParams(tree_crown_d=10.0, **kw)
    st = {}
    pts, _, _ = _place_trees(LAWN.difference(ROAD).difference(PARK), HOUSES, PARK, p, ROAD, None, None, st)
    return pts, st


def test_default_is_full_grid():
    pts, st = _trees()
    assert len(pts) == st["trees_full_grid"] > 50


def test_density_keeps_share_and_is_subset_of_full_grid():
    full, _ = _trees()
    half, st = _trees(tree_density=0.5, tree_seed=3)
    assert len(half) == round(len(full) * 0.5)
    assert st["trees_full_grid"] == len(full)
    assert {(t.x, t.y) for t in half} <= {(t.x, t.y) for t in full}


def test_density_seed_reproducible_and_varies():
    a, _ = _trees(tree_density=0.3, tree_seed=1)
    b, _ = _trees(tree_density=0.3, tree_seed=1)
    c, _ = _trees(tree_density=0.3, tree_seed=2)
    assert [(t.x, t.y) for t in a] == [(t.x, t.y) for t in b]
    assert [(t.x, t.y) for t in a] != [(t.x, t.y) for t in c]


def test_density_zero_gives_no_trees():
    pts, _ = _trees(tree_density=0.0)
    assert pts == []


def test_roads_zone_only_near_road():
    pts, _ = _trees(tree_zones=("roads",), tree_zone_reach=10.0)
    assert pts
    assert all(ROAD.distance(t) <= 10.0 + 1e-6 for t in pts)
    assert len(pts) < len(_trees()[0])


def test_parking_zone_only_near_parking():
    pts, _ = _trees(tree_zones=("parking",), tree_zone_reach=15.0)
    assert pts and all(PARK.distance(t) <= 15.0 + 1e-6 for t in pts)


def test_courtyards_zone_near_walls_and_union_of_zones():
    yard, _ = _trees(tree_zones=("courtyards",), tree_yard_reach=30.0)
    assert yard and all(HOUSES.distance(t) <= 30.0 + 1e-6 for t in yard)
    both, _ = _trees(tree_zones=("courtyards", "roads"), tree_yard_reach=30.0, tree_zone_reach=10.0)
    roads, _ = _trees(tree_zones=("roads",), tree_zone_reach=10.0)
    assert len(both) >= max(len(yard), len(roads))


def test_zone_without_source_gives_no_trees_and_buffers_kept():
    p = LandscapeParams(tree_crown_d=10.0, tree_zones=("parking",))
    pts, _, _ = _place_trees(LAWN, HOUSES, None, p, None, None, None)
    assert pts == []
    pts, _ = _trees(tree_zones=("all",))
    assert all(HOUSES.distance(t) >= 5.0 - 1e-6 for t in pts)  # отступ от стен сохраняется

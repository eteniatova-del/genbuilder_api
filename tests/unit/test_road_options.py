"""Параметры управления проездами: стиль сети, штрафы, норм-режим отчёта, масштаб QGIS."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from app.logic.generation_params import GenParams
from app.logic.landscape import LandscapeParams, generate_landscape, landscape_params_from_generation, road_params
from app.logic.layout_checks import check_layout
from app.logic.roads import RoadParams, generate_roads

BLOCK = box(0, 0, 400, 300)
HOUSES = [box(60, 60, 160, 78), box(220, 60, 320, 78), box(60, 180, 160, 198), box(220, 180, 320, 198)]
ROOT = Path(__file__).resolve().parents[2]


def test_gen_params_flow_into_road_params():
    gp = GenParams(road_style="loop", road_turn_penalty_m=30.0, road_junction_penalty_m=5.0, road_reuse_cost=0.3,
                   parking_reach_m=8.0, parking_max_lot_places=30, norm_mode="strict")
    lp = landscape_params_from_generation(gp)
    rp = road_params(lp)
    assert (rp.road_style, rp.turn_penalty, rp.junction_penalty, rp.reuse_cost, rp.patch_reach) == ("loop", 30.0, 5.0, 0.3, 8.0)
    assert lp.max_lot_places == 30 and lp.norm_mode == "strict"


def test_loop_style_has_no_longer_dead_ends_than_minimal():
    a = generate_roads(BLOCK, HOUSES, [5] * 4, p=RoadParams(road_style="minimal"))["diagnostics"]
    b = generate_roads(BLOCK, HOUSES, [5] * 4, p=RoadParams(road_style="loop"))["diagnostics"]
    assert b["unreachable"] == []
    assert b["dead_ends"] <= a["dead_ends"]


def test_turn_penalty_does_not_break_reachability():
    res = generate_roads(BLOCK, HOUSES, [5] * 4, p=RoadParams(turn_penalty=60.0, junction_penalty=40.0))
    assert res["diagnostics"]["unreachable"] == []


def test_patch_near_existing_road_gets_no_extra_stub():
    base = generate_roads(BLOCK, HOUSES, [5] * 4)
    road = unary_union([c["geometry"] for c in base["carriageways"]])
    pt = road.representative_point()
    cands = [box(pt.x + dx - 2, pt.y + dy - 2, pt.x + dx + 2, pt.y + dy + 2) for dx, dy in ((8, 0), (-8, 0), (0, 8), (0, -8))]
    patch = next(c for c in cands if BLOCK.contains(c) and not any(c.intersects(h.buffer(8)) for h in HOUSES) and c.distance(road) > 3)
    assert patch.distance(road) <= 12.0
    with_patch = generate_roads(BLOCK, HOUSES, [5] * 4, patches=[patch])
    assert with_patch["diagnostics"]["road_length_m"] <= base["diagnostics"]["road_length_m"] + 10.0


def _checks(mode):
    lp = LandscapeParams(parking_type="surface", roads_enabled=True, norm_mode=mode)
    roads = generate_roads(BLOCK, HOUSES, [8] * 4)
    res = generate_landscape(BLOCK, HOUSES, 600, lp, roads=roads, road_inputs={"floors": [8] * 4, "services": None, "params": RoadParams()})
    return check_layout(BLOCK, HOUSES, [8] * 4, None, res["roads"], res, lp)


def test_every_check_has_basis_and_strict_mode_marks_unverified():
    ext = _checks("extended")
    assert all(c["basis"] in ("norm", "secondary", "assumption") for c in ext)
    st = _checks("strict")
    for c_e, c_s in zip(ext, st):
        if c_e["status"] == "pass" and c_e["basis"] != "norm":
            assert c_s["status"] == "unverified"
        if c_e["basis"] == "norm":
            assert c_s["status"] == c_e["status"]


def _export(tmp_path, scale):
    src = tmp_path / "r.json"
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [30, 60]}}]}
    src.write_text(json.dumps({"generated_buildings": fc}))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "export_qgis_project.py"), str(src), str(tmp_path / "o"), "--scale", str(scale)],
                          capture_output=True, text=True)


@pytest.mark.parametrize("scale", [500, 1000, 2000, 5000])
def test_qgis_script_accepts_scale_series(tmp_path, scale):
    r = _export(tmp_path, scale)
    assert r.returncode == 0, r.stderr
    code = (tmp_path / "o" / "open_in_qgis.py").read_text()
    assert "SCALE = %d" % scale in code
    compile(code, "open_in_qgis.py", "exec")


def test_qgis_script_rejects_scale_outside_series(tmp_path):
    assert _export(tmp_path, 750).returncode != 0


def _landscape(layout, services=None):
    lp = LandscapeParams(parking_type="surface", roads_enabled=True, parking_layout=layout)
    roads = generate_roads(BLOCK, HOUSES, [8] * 4)
    res = generate_landscape(BLOCK, HOUSES, 600, lp, roads=roads, services=services,
                             road_inputs={"floors": [8] * 4, "services": services, "params": RoadParams()})
    return lp, roads, res


def test_along_roads_patches_touch_the_road():
    lp, roads, res = _landscape("along_roads")
    road_u = unary_union([c["geometry"] for c in roads["carriageways"]])
    patches = [x for x in res["parking_patches"] if x["kind"] == "surface"]
    assert patches
    near = [x for x in patches if x["geometry"].distance(road_u) <= lp.touch_tol + 0.01]
    assert len(near) >= len(patches) * 0.5


def test_parking_keeps_distance_to_school_and_walk_limits():
    services = ["", "школа", "", ""]
    lp, roads, res = _landscape("along_roads", services)
    ch = {c["id"]: c for c in check_layout(BLOCK, HOUSES, [8] * 4, services, res["roads"], res, lp)}
    assert ch["parking_edu_setback"]["status"] == "pass", ch["parking_edu_setback"]["violations"]
    assert ch["parking_walk_distance"]["status"] == "pass", ch["parking_walk_distance"]["violations"]
    assert ch["parking_walk_distance"]["basis"] == "norm"


def test_shrubs_follow_norm_distances_and_use_listed_species():
    lp, roads, res = _landscape("along_roads")
    shrubs = res["shrubs"]
    assert len(shrubs) > 20
    assert {x["species_id"] for x in shrubs} <= {t[0] for t in lp.shrub_species}
    ch = {c["id"]: c for c in check_layout(BLOCK, HOUSES, [8] * 4, None, res["roads"], res, lp)}
    assert ch["shrubs_buffers"]["status"] == "pass", ch["shrubs_buffers"]["violations"]
    assert ch["shrubs_buffers"]["basis"] == "norm"


def test_shrubs_can_be_switched_off_and_limited_to_species():
    lp = LandscapeParams(parking_type="surface", roads_enabled=True, shrubs_enabled=False)
    roads = generate_roads(BLOCK, HOUSES, [8] * 4)
    res = generate_landscape(BLOCK, HOUSES, 600, lp, roads=roads, road_inputs={"floors": [8] * 4, "services": None, "params": RoadParams()})
    assert res["shrubs"] == []
    lp2 = LandscapeParams(parking_type="surface", roads_enabled=True, shrub_species_ids=("rose",))
    res2 = generate_landscape(BLOCK, HOUSES, 600, lp2, roads=roads, road_inputs={"floors": [8] * 4, "services": None, "params": RoadParams()})
    assert {x["species_id"] for x in res2["shrubs"]} == {"rose"}

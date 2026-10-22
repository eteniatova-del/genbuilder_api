"""
Модуль озеленения и парковок (федеральные нормы).

Вход: полигон квартала и полигоны зданий (метры, UTM), число жителей, класс жилья.
Выход: полигоны парковок (наземных), точки деревьев, диагностика (нормативная потребность, нехватка).

Нормативная основа (федеральная, без региональных норм):
  СП 42.13330.2026: табл. 6.1 (30/40 м2/чел.), табл. 8.16 (1,2/2,0 места на квартиру),
      п. 8.113 прим. 2 (350 легковых на 1000 жителей, допускается для крупных городов),
      п. 8.117 (гостевые не менее 30 мест на 1000 жителей), табл. 8.17 (25 м2 на наземное место),
      табл. 6.3 (деревья: 5,0 м от стены, 0,7 м от стоянки; кустарники: 1,5 / 0,5 м; примечание 1: для кроны > 5 м увеличивать).
  СП 476.1325800.2020 п. 7.29, табл. 7.1 (по странице tiflocentre.ru, сверить): зелёные насаждения 3,0 м2/чел.,
      детские площадки 0,4, площадки отдыха 0,1 (площадки здесь только резервируются по площади).
ДОПУЩЕНИЯ (не нормы, помечены ASSUMPTION): диаметр кроны и шаг деревьев, увеличенный отступ от стены,
  разрывы стоянок от окон (СанПиН 2.2.1/2.1.1.1200-03 табл. 7.1.1 по вторичному источнику), ячейка парковки 2,5 x 10 м.
Зачётная площадь озеленения = газон (решение автора); деревья/кустарники размещаются на газоне с буферами.
"""
from dataclasses import dataclass, field
from math import ceil, sqrt
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union


@dataclass
class LandscapeParams:
    housing_class: str = "standard"                    # standard | business
    total_area_per_person: dict = field(default_factory=lambda: {"standard": 30.0, "business": 40.0})  # СП 42 табл. 6.1
    spots_per_apartment: dict = field(default_factory=lambda: {"standard": 1.2, "business": 2.0})      # СП 42 табл. 8.16
    persons_per_apartment: float | None = None         # нужен источник; если None, места считаются по 350/1000
    cars_per_1000: float = 350.0                       # СП 42 п. 8.113 прим. 2
    guest_per_1000: float = 30.0                       # СП 42 п. 8.117
    area_per_surface_spot: float = 25.0                # СП 42 табл. 8.17
    green_per_person: float = 3.0                      # СП 476 табл. 7.1 (сверить)
    playground_per_person: float = 0.4                 # там же
    rest_per_person: float = 0.1                       # там же
    tree_crown_d: float = 15.0                         # ASSUMPTION: клён остролистный, крона 15-20 м (каталог питомника)
    tree_wall_buffer: float | None = None              # None -> max(5.0, crown_d/2), ASSUMPTION по прим. 1 табл. 6.3
    tree_carriageway_buffer: float = 2.0               # СП 42 табл. 6.3
    tree_parking_buffer: float = 0.7                   # СП 42 табл. 6.3
    tree_spacing: float | None = None                  # None -> crown_d, ASSUMPTION (нормы шага нет)
    cell_w: float = 2.5                                # ASSUMPTION: ячейка места с долей проезда 2,5 x 10 м = 25 м2
    cell_l: float = 10.0
    underground_step: float = 0.1
    max_lot_places: int = 50                           # ASSUMPTION: стоянки делятся на лоты не более N мест; разрыв от окон берётся по размеру лота
    # ASSUMPTION (вторичный источник): разрыв открытой стоянки до окон по числу мест
    window_setback: tuple = ((10, 10.0), (50, 15.0), (100, 25.0), (300, 35.0), (10**9, 50.0))


def _setback(n_places, table):
    for lim, s in table:
        if n_places <= lim:
            return s
    return table[-1][1]


def demand(residents, p: LandscapeParams):
    flat_area = residents * p.total_area_per_person[p.housing_class]
    if p.persons_per_apartment:
        apts = residents / p.persons_per_apartment
        permanent = ceil(apts * p.spots_per_apartment[p.housing_class])
        basis = "СП 42 табл. 8.16"
    else:
        permanent = ceil(residents * p.cars_per_1000 / 1000)
        basis = "СП 42 п. 8.113 прим. 2"
    guest = ceil(residents * p.guest_per_1000 / 1000)
    return {
        "flat_area_m2": flat_area, "permanent_places": permanent, "guest_places": guest,
        "permanent_basis": basis,
        "green_required_m2": residents * p.green_per_person,
        "platforms_reserved_m2": residents * (p.playground_per_person + p.rest_per_person),
    }


DISCLAIMER = (
    "Допущение: улично-дорожная сеть (проезды, дорожки, пожарные проезды) в работе пока не рассматривается, "
    "вся площадь квартала вне зданий считается свободной. Расчёт эскизный (предпроектный): размещение "
    "парковок и деревьев упрощено, результат носит оценочный характер и содержит погрешность; "
    "не заменяет проектную документацию."
)


def _place_cells(zone, n, p, taken):
    """Жадная раскладка ячеек 2,5 x 10 м по сетке в зоне; возвращает список прямоугольников."""
    if n <= 0 or zone.is_empty:
        return []
    minx, miny, maxx, maxy = zone.bounds
    out = []
    y = miny
    while y + p.cell_l <= maxy and len(out) < n:
        x = minx
        while x + p.cell_w <= maxx and len(out) < n:
            c = box(x, y, x + p.cell_w, y + p.cell_l)
            if zone.contains(c) and not any(c.intersects(t) for t in taken + out):
                out.append(c)
            x += p.cell_w
        y += p.cell_l
    return out


def _place_trees(lawn, buildings_u, parking_u, p, carriageway=None, walkways=None):
    wall = p.tree_wall_buffer if p.tree_wall_buffer is not None else max(5.0, p.tree_crown_d / 2)
    step = p.tree_spacing if p.tree_spacing is not None else p.tree_crown_d
    zone = lawn
    if buildings_u is not None:
        zone = zone.difference(buildings_u.buffer(wall))
    if parking_u is not None and not parking_u.is_empty:
        zone = zone.difference(parking_u.buffer(p.tree_parking_buffer))
    if carriageway is not None and not carriageway.is_empty:   # СП 42 табл. 6.3: 2,0 м от края проезжей части
        zone = zone.difference(carriageway.buffer(p.tree_carriageway_buffer))
    if walkways is not None and not walkways.is_empty:         # 0,7 м от края тротуара и дорожки
        zone = zone.difference(walkways.buffer(p.tree_parking_buffer))
    if zone.is_empty:
        return [], wall, step
    minx, miny, maxx, maxy = zone.bounds
    pts = []
    row = 0
    y = miny
    while y <= maxy:
        x = minx + (step / 2 if row % 2 else 0)
        while x <= maxx:
            pt = Point(x, y)
            if zone.contains(pt):
                pts.append(pt)
            x += step
        y += step * sqrt(3) / 2
        row += 1
    # жёсткая проверка шага (гекс-сетка даёт step, но у краёв после фильтрации оставляем как есть)
    return pts, wall, step


def generate_landscape(block: Polygon, buildings: list, residents: float, p: LandscapeParams | None = None, carriageway=None, walkways=None):
    p = p or LandscapeParams()
    d = demand(residents, p)
    bu = unary_union(buildings) if buildings else None
    free = block.difference(bu) if bu is not None else block
    free_area = free.area

    chosen = None
    # перебор доли подземных постоянных мест: 0, 0.1, ..., 1.0
    steps = int(round(1 / p.underground_step))
    for k in range(steps + 1):
        u = k * p.underground_step
        perm_surface = ceil(d["permanent_places"] * (1 - u))
        # гостевые: разрывы от окон не устанавливаются (СанПиН прим. 11, вторичный источник), ставим везде в свободной зоне
        zone_perm = free.difference(bu.buffer(_setback(min(perm_surface, p.max_lot_places), p.window_setback))) if bu is not None else free
        perm_cells = _place_cells(zone_perm, perm_surface, p, [])
        guest_cells = _place_cells(free, d["guest_places"], p, perm_cells)
        placed_perm, placed_guest = len(perm_cells), len(guest_cells)
        parking_u = unary_union(perm_cells + guest_cells) if (perm_cells or guest_cells) else None
        lawn = free.difference(parking_u) if parking_u is not None else free
        lawn_area = lawn.area - d["platforms_reserved_m2"]
        ok = placed_perm >= perm_surface and placed_guest >= d["guest_places"] and lawn_area >= d["green_required_m2"]
        chosen = (u, perm_cells, guest_cells, parking_u, lawn, lawn_area, perm_surface, placed_perm, placed_guest)
        if ok:
            break
    u, perm_cells, guest_cells, parking_u, lawn, lawn_area, perm_surface, placed_perm, placed_guest = chosen
    trees, wall, step = _place_trees(lawn, bu, parking_u, p, carriageway, walkways)

    diag = {
        **d,
        "free_area_after_buildings_m2": round(free_area, 1),
        "underground_share": round(u, 2),
        "permanent_surface_required": perm_surface,
        "permanent_surface_placed": placed_perm,
        "permanent_underground": d["permanent_places"] - perm_surface,
        "guest_placed": placed_guest,
        "lawn_available_m2": round(lawn_area, 1),
        "green_deficit_m2": round(max(0.0, d["green_required_m2"] - lawn_area), 1),
        "parking_deficit_surface": max(0, perm_surface - placed_perm) + max(0, d["guest_places"] - placed_guest),
        "trees": len(trees), "tree_wall_buffer_m": wall, "tree_spacing_m": step,
        "disclaimer": DISCLAIMER,
        "assumptions": ["tree_crown_d", "tree_wall_buffer", "tree_spacing", "cell 2.5x10", "window_setback (SanPiN, secondary)"],
    }
    return {"parking_cells": perm_cells + guest_cells, "lawn": lawn, "trees": trees, "diagnostics": diag}


# ---------------------------------------------------------------------------
# Интеграция с Genbuilder: расчёт по всем кварталам и сборка GeoDataFrame
# ---------------------------------------------------------------------------
def landscape_params_from_generation(gen_params) -> LandscapeParams:
    """Строит LandscapeParams из GenParams (поля landscape_* и housing_class)."""
    return LandscapeParams(
        housing_class=gen_params.housing_class,
        persons_per_apartment=gen_params.persons_per_apartment,
        tree_crown_d=float(gen_params.tree_crown_d),
    )


def apply_landscape(blocks_utm, buildings_utm, la_per_person: float, p: LandscapeParams):
    """Считает озеленение и парковки для кварталов с жильём.

    blocks_utm: GeoDataFrame кварталов (UTM, метры) с колонками geometry, zone.
    buildings_utm: GeoDataFrame сгенерированных зданий (UTM) с колонками geometry, living_area.
    Возвращает словарь {"greenery": GeoDataFrame, "parking": GeoDataFrame, "diagnostics": list[dict]}
    в той же системе координат (UTM); перепроецирование делает вызывающий код.
    """
    import geopandas as gpd

    greenery_rows, parking_rows, diag = [], [], []
    for bid, (_, block) in enumerate(blocks_utm.iterrows()):
        geom = block.geometry
        if geom is None or geom.is_empty:
            continue
        inside = buildings_utm[buildings_utm.geometry.representative_point().within(geom)]
        la = float(inside["living_area"].fillna(0).sum()) if "living_area" in inside.columns else 0.0
        residents = la / la_per_person if la_per_person else 0.0
        if residents <= 0:
            continue
        footprints = [g for g in inside.geometry if g is not None and not g.is_empty]
        res = generate_landscape(geom, footprints, residents, p)
        d = res["diagnostics"]
        d["block_index"] = bid
        d["residents"] = round(residents)
        diag.append(d)
        lawn = res["lawn"]
        if not lawn.is_empty:
            greenery_rows.append({"block_index": bid, "kind": "lawn", "geometry": lawn})
        for t in res["trees"]:
            greenery_rows.append({"block_index": bid, "kind": "tree", "geometry": t})
        for c in res["parking_cells"]:
            parking_rows.append({"block_index": bid, "kind": "surface", "geometry": c})
    crs = blocks_utm.crs
    return {
        "greenery": gpd.GeoDataFrame(greenery_rows, geometry="geometry", crs=crs) if greenery_rows
        else gpd.GeoDataFrame({"block_index": [], "kind": []}, geometry=[], crs=crs),
        "parking": gpd.GeoDataFrame(parking_rows, geometry="geometry", crs=crs) if parking_rows
        else gpd.GeoDataFrame({"block_index": [], "kind": []}, geometry=[], crs=crs),
        "diagnostics": diag,
    }

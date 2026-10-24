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
    parking_type: str = "mixed"                        # underground | surface | mixed (выбирает пользователь)
    guest_gap: float = 3.0                             # ASSUMPTION: минимальный зазор гостевого пятна от стены, м (норм нет)
    roads_enabled: bool = False                        # строить внутриквартальные проезды (app/logic/roads.py)
    floor_height: float = 3.0                          # ASSUMPTION: высота этажа, м (для пожарной ширины и отступа проезда)
    parking_road_reach: float = 12.0                   # ASSUMPTION: пятна парковки предпочтительно не дальше этого расстояния от проезда
    patch_gap: float = 2.5                             # ASSUMPTION: зазор между пятнами разных домов, м
    search_step: float = 5.0                           # шаг перебора позиций пятна, м
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


def _shapes(n, p):
    """Компактные варианты пятна из n ячеек: (ширина, высота, число столбцов, число рядов, повёрнуто)."""
    out = []
    for rows in range(1, 6):
        cols = ceil(n / rows)
        if cols < rows:
            continue
        out.append((cols * p.cell_w, rows * p.cell_l, cols, rows, False))
        out.append((rows * p.cell_l, cols * p.cell_w, cols, rows, True))
    return out or [(n * p.cell_w, p.cell_l, n, 1, False)]


def _patch_cells(x, y, n, cols, rows, rot, p):
    cells = []
    for r in range(rows):
        for c in range(cols):
            if len(cells) >= n:
                break
            if rot:
                cells.append(box(x + r * p.cell_l, y + c * p.cell_w, x + (r + 1) * p.cell_l, y + (c + 1) * p.cell_w))
            else:
                cells.append(box(x + c * p.cell_w, y + r * p.cell_l, x + (c + 1) * p.cell_w, y + (r + 1) * p.cell_l))
    return cells


def _find_patch(n, zone, target, p):
    """Одно компактное пятно из n ячеек в zone, ближайшее к target; None, если не помещается."""
    if n <= 0 or zone.is_empty:
        return None
    from shapely.prepared import prep
    pz = prep(zone)
    minx, miny, maxx, maxy = zone.bounds
    step = p.search_step
    best, best_d = None, 1e18
    for w, h, cols, rows, rot in _shapes(n, p):
        x = minx
        while x + w <= maxx:
            y = miny
            while y + h <= maxy:
                r = box(x, y, x + w, y + h)
                if pz.contains(r):
                    d = r.distance(target)
                    if d < best_d:
                        best, best_d = (x, y, cols, rows, rot), d
                y += step
            x += step
    if best is None:
        return None
    x, y, cols, rows, rot = best
    return _patch_cells(x, y, n, cols, rows, rot, p)


def _place_for_house(n, free, bu_u, taken, target, p, guest=False, depth=0, near=None):
    """Размещает n мест у дома одним пятном; если не помещается, делит пополам (не глубже 2 раз).
    Возвращает (список пятен-ячеек, размещено мест, отступ от окон, м)."""
    if n <= 0:
        return [], 0, 0.0
    sb = p.guest_gap if guest else _setback(n, p.window_setback)
    zone = free
    if bu_u is not None:
        zone = zone.difference(bu_u.buffer(sb))
    if taken is not None and not taken.is_empty:
        zone = zone.difference(taken.buffer(p.patch_gap))
    cells = None
    if near is not None and not near.is_empty:          # сначала пробуем у проезда, затем где угодно
        cells = _find_patch(n, zone.intersection(near.buffer(p.parking_road_reach)), target, p)
    if cells is None:
        cells = _find_patch(n, zone, target, p)
    if cells is not None:
        return [cells], n, sb
    if depth >= 2 or n < 2:
        return [], 0, sb
    h1 = n // 2
    p1, k1, s1 = _place_for_house(h1, free, bu_u, taken, target, p, guest, depth + 1, near)
    t2 = unary_union([taken] + [c for pc in p1 for c in pc]) if (taken is not None and p1) else (unary_union([c for pc in p1 for c in pc]) if p1 else taken)
    p2, k2, s2 = _place_for_house(n - h1, free, bu_u, t2, target, p, guest, depth + 1, near)
    return p1 + p2, k1 + k2, max(s1, s2)


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


def generate_landscape(block: Polygon, buildings: list, residents: float, p: LandscapeParams | None = None,
                       carriageway=None, walkways=None, house_residents: list | None = None, roads=None):
    """Парковки привязаны к домам: у каждого дома своё пятно постоянных мест и своё пятно гостевых.

    parking_type: underground (постоянные только под землёй), surface (только наземные; нехватка
    места показывается как дефицит), mixed (доля подземных подбирается минимальной).
    """
    p = p or LandscapeParams()
    d = demand(residents, p)
    bu = unary_union(buildings) if buildings else None
    free = block.difference(bu) if bu is not None else block
    road_u = None
    if roads is not None:
        parts = [c["geometry"] for c in roads.get("carriageways", [])] + list(roads.get("turnarounds", []))
        parts = [g for g in parts if g is not None and not g.is_empty]
        if parts:
            road_u = unary_union(parts)
            free = free.difference(road_u)
            carriageway = road_u if carriageway is None else unary_union([carriageway, road_u])
    free_area = free.area
    n_h = len(buildings)
    if house_residents is None or len(house_residents) != n_h:
        tot = sum(b.area for b in buildings) or 1.0
        house_residents = [residents * b.area / tot for b in buildings]
    hd = [demand(r, p) for r in house_residents]
    order = sorted(range(n_h), key=lambda i: -hd[i]["permanent_places"])

    if p.parking_type == "underground":
        u_values = [1.0]
    elif p.parking_type == "surface":
        u_values = [0.0]
    else:
        steps = int(round(1 / p.underground_step))
        u_values = [k * p.underground_step for k in range(steps + 1)]

    chosen = None
    for u in u_values:
        taken, patches, houses_diag = None, [], []
        deficit = 0
        for i in order:
            perm_surf = ceil(hd[i]["permanent_places"] * (1 - u))
            pc, placed, sb = _place_for_house(perm_surf, free, bu, taken, buildings[i], p, near=road_u)
            for c in pc:
                patches.append({"house_index": i, "kind": "surface", "cells": c})
            if pc:
                taken = unary_union(([taken] if taken is not None else []) + [x for c in pc for x in c])
            gc, gplaced, _ = _place_for_house(hd[i]["guest_places"], free, bu, taken, buildings[i], p, guest=True, near=road_u)
            for c in gc:
                patches.append({"house_index": i, "kind": "guest", "cells": c})
            if gc:
                taken = unary_union(([taken] if taken is not None else []) + [x for c in gc for x in c])
            deficit += (perm_surf - placed) + (hd[i]["guest_places"] - gplaced)
            houses_diag.append({
                "house_index": i, "residents": round(house_residents[i]),
                "permanent_places": hd[i]["permanent_places"], "permanent_underground": hd[i]["permanent_places"] - perm_surf,
                "permanent_surface": placed, "permanent_surface_required": perm_surf,
                "guest_places": gplaced, "guest_required": hd[i]["guest_places"],
                "window_setback_m": sb, "patches": len(pc) + len(gc),
            })
        parking_u = unary_union([x for pt in patches for x in pt["cells"]]) if patches else None
        lawn = free.difference(parking_u) if parking_u is not None else free
        lawn_area = lawn.area - d["platforms_reserved_m2"]
        ok = deficit == 0 and lawn_area >= d["green_required_m2"]
        chosen = (u, patches, parking_u, lawn, lawn_area, deficit, houses_diag)
        if ok:
            break
    u, patches, parking_u, lawn, lawn_area, deficit, houses_diag = chosen
    trees, wall, step = _place_trees(lawn, bu, parking_u, p, carriageway, walkways)
    houses_diag.sort(key=lambda h: h["house_index"])
    perm_surface = sum(h["permanent_surface_required"] for h in houses_diag)

    diag = {
        **d,
        "parking_type": p.parking_type,
        "free_area_after_buildings_m2": round(free_area, 1),
        "underground_share": round(u, 2),
        "permanent_surface_required": perm_surface,
        "permanent_surface_placed": sum(h["permanent_surface"] for h in houses_diag),
        "permanent_underground": d["permanent_places"] - perm_surface,
        "guest_placed": sum(h["guest_places"] for h in houses_diag),
        "lawn_available_m2": round(lawn_area, 1),
        "green_deficit_m2": round(max(0.0, d["green_required_m2"] - lawn_area), 1),
        "parking_deficit_surface": deficit,
        "trees": len(trees), "tree_wall_buffer_m": wall, "tree_spacing_m": step,
        "houses": houses_diag,
        "roads": roads["diagnostics"] if roads is not None else None,
        "disclaimer": DISCLAIMER,
        "assumptions": ["tree_crown_d", "tree_wall_buffer", "tree_spacing", "cell 2.5x10", "window_setback (SanPiN, secondary)",
                        "guest_gap", "patch_gap", "underground parking: places only, area norm not set"],
    }
    # parking_cells: по одному полигону на пятно (объединение ячеек) — для деревьев и вывода
    cells_out = [unary_union(pt["cells"]) for pt in patches]
    return {"parking_cells": cells_out, "parking_patches": [
        {"house_index": pt["house_index"], "kind": pt["kind"], "places": len(pt["cells"]), "geometry": unary_union(pt["cells"])}
        for pt in patches], "lawn": lawn, "trees": trees, "diagnostics": diag}


# ---------------------------------------------------------------------------
# Интеграция с Genbuilder: расчёт по всем кварталам и сборка GeoDataFrame
# ---------------------------------------------------------------------------
def landscape_params_from_generation(gen_params) -> LandscapeParams:
    """Строит LandscapeParams из GenParams (поля landscape_* и housing_class)."""
    return LandscapeParams(
        housing_class=gen_params.housing_class,
        persons_per_apartment=gen_params.persons_per_apartment,
        tree_crown_d=float(gen_params.tree_crown_d),
        parking_type=gen_params.parking_type,
        roads_enabled=gen_params.roads_enabled,
        floor_height=float(gen_params.floor_height_m),
    )


def apply_landscape(blocks_utm, buildings_utm, la_per_person: float, p: LandscapeParams):
    """Считает озеленение и парковки для кварталов с жильём.

    blocks_utm: GeoDataFrame кварталов (UTM, метры) с колонками geometry, zone.
    buildings_utm: GeoDataFrame сгенерированных зданий (UTM) с колонками geometry, living_area.
    Возвращает словарь {"greenery": GeoDataFrame, "parking": GeoDataFrame, "diagnostics": list[dict]}
    в той же системе координат (UTM); перепроецирование делает вызывающий код.
    """
    import geopandas as gpd

    greenery_rows, parking_rows, road_rows, diag = [], [], [], []
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
        las = [float(v) if v == v else 0.0 for v in inside["living_area"]] if "living_area" in inside.columns else []
        hres = [x / la_per_person for x in las] if las and len(las) == len(footprints) else None
        roads = None
        if p.roads_enabled:
            from app.logic.roads import RoadParams, generate_roads
            fl = [float(v) if v == v else 1.0 for v in inside["floors_count"]] if "floors_count" in inside.columns else [1.0] * len(footprints)
            sv = [(x if isinstance(x, str) else "") for x in inside["service"]] if "service" in inside.columns else None
            roads = generate_roads(geom, footprints, fl, sv, RoadParams(floor_height=p.floor_height))
            for c in roads["carriageways"]:
                road_rows.append({"block_index": bid, "kind": "carriageway", "width_m": c["width"],
                                  "road_class": c["class"], "geometry": c["geometry"]})
            for t in roads["turnarounds"]:
                road_rows.append({"block_index": bid, "kind": "turnaround", "width_m": 15.0,
                                  "road_class": "turnaround", "geometry": t})
        res = generate_landscape(geom, footprints, residents, p, house_residents=hres, roads=roads)
        d = res["diagnostics"]
        d["block_index"] = bid
        d["residents"] = round(residents)
        diag.append(d)
        lawn = res["lawn"]
        if not lawn.is_empty:
            greenery_rows.append({"block_index": bid, "kind": "lawn", "geometry": lawn})
        for t in res["trees"]:
            greenery_rows.append({"block_index": bid, "kind": "tree", "geometry": t})
        for pt in res["parking_patches"]:
            parking_rows.append({"block_index": bid, "house_index": pt["house_index"],
                                 "kind": "surface" if pt["kind"] == "surface" else "guest",
                                 "places": pt["places"], "geometry": pt["geometry"]})
        for hd_ in d.get("houses", []):                      # подземные места: контур дома, площадь по месту не нормируем
            if hd_["permanent_underground"] > 0:
                parking_rows.append({"block_index": bid, "house_index": hd_["house_index"], "kind": "underground",
                                     "places": hd_["permanent_underground"], "geometry": footprints[hd_["house_index"]]})
    crs = blocks_utm.crs
    return {
        "greenery": gpd.GeoDataFrame(greenery_rows, geometry="geometry", crs=crs) if greenery_rows
        else gpd.GeoDataFrame({"block_index": [], "kind": []}, geometry=[], crs=crs),
        "parking": gpd.GeoDataFrame(parking_rows, geometry="geometry", crs=crs) if parking_rows
        else gpd.GeoDataFrame({"block_index": [], "kind": []}, geometry=[], crs=crs),
        "roads": gpd.GeoDataFrame(road_rows, geometry="geometry", crs=crs) if road_rows
        else gpd.GeoDataFrame({"block_index": [], "kind": []}, geometry=[], crs=crs),
        "diagnostics": diag,
    }

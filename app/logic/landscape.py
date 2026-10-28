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
import pandas as pd
import shapely
from dataclasses import dataclass, field
from math import ceil, hypot, sqrt
from shapely.geometry import Polygon, Point, box
from shapely.ops import nearest_points, unary_union


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
    shrubs_enabled: bool = True
    # Виды: сообщение Смольного от 14.11.2024 (кустарники, высаженные Комитетом по благоустройству в 2024 г.: кизильник, барбарис, сирень, спирея, роза парковая).
    # Размеры куста и шаг — ASSUMPTION: в найденных источниках размеров по видам нет. (id, название, латинское название, диаметр куста м, шаг посадки м)
    shrub_species: tuple = (
        ("spirea", "Спирея", "Spiraea", 1.5, 1.5),
        ("cotoneaster", "Кизильник блестящий", "Cotoneaster lucidus", 1.5, 1.5),
        ("barberry", "Барбарис", "Berberis", 1.5, 1.5),
        ("lilac", "Сирень обыкновенная", "Syringa vulgaris", 3.0, 3.0),
        ("rose", "Роза парковая", "Rosa", 1.2, 1.2),
    )
    shrub_species_ids: tuple = ("spirea", "cotoneaster", "barberry", "lilac", "rose")
    shrub_wall_buffer: float = 1.5                     # СП 42 табл. 6.3: кустарник до наружной стены, м (до оси куста)
    shrub_road_buffer: float = 1.0                     # СП 42 табл. 6.3: до края проезжей части
    shrub_parking_buffer: float = 0.5                  # СП 42 табл. 6.3: до края тротуара, дорожки и площадки хранения транспортных средств
    shrub_belt_offset: float = 2.5                     # ASSUMPTION: пояс кустарников вдоль стен на этом расстоянии от стены (норма 1,5 м плюс запас)
    shrub_hedge_offset: float = 1.0                    # ASSUMPTION: живая изгородь вдоль пятна парковки на этом расстоянии от края пятна
    shrub_wall_belt: bool = True
    shrub_parking_hedge: bool = True
    parking_layout: str = "along_roads"                # along_roads: пятна примыкают к проезду (сначала проезды, потом парковки) | free
    edu_setback: tuple = ((10, 25.0), (10**9, 50.0))   # вторичный источник (СанПиН 2.2.1/2.1.1.1200-03 табл. 7.1.1): стоянки до школ и детских учреждений, по числу мест
    walk_permanent_m: float = 800.0                    # СП 42 п. 8.113: пешеходная доступность постоянного хранения (реконструкция: 1200 м)
    walk_guest_m: float = 200.0                        # СП 42 п. 8.125: от гостевых стоянок до входов в жилые дома
    touch_tol: float = 1.0                             # допуск «пятно примыкает к проезду», м
    road_style: str = "minimal"                        # minimal | loop
    road_turn_penalty: float = 20.0                    # ASSUMPTION, м эквивалентной длины
    road_junction_penalty: float = 10.0                # ASSUMPTION, м эквивалентной длины
    road_reuse_cost: float = 0.15                      # ASSUMPTION
    norm_mode: str = "extended"                        # extended | strict (влияет на отчёт проверок)
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
    "Допущение: внутриквартальные проезды строятся упрощённо (ортогональная растровая сеть, ширина по классу и "
    "высоте здания), дороги вне кварталов из площади не вычитаются. Расчёт эскизный (предпроектный): размещение "
    "парковок, деревьев и проездов упрощено, результат носит оценочный характер и содержит погрешность; "
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


def _find_patch(n, zone, target, p, touch=None):
    """Одно компактное пятно из n ячеек в zone, ближайшее к target; None, если не помещается."""
    if n <= 0 or zone.is_empty:
        return None
    if zone.area < n * p.cell_w * p.cell_l:       # свободной площади меньше, чем нужно местам: пятно заведомо не поместится
        return None
    from shapely.prepared import prep
    pz = prep(zone)
    minx, miny, maxx, maxy = zone.bounds
    step = p.search_step
    best, best_d = None, 1e18
    eroded = {}
    for w, h, cols, rows, rot in _shapes(n, p):
        # центр прямоугольника w x h, лежащего в зоне, отстоит от границы зоны не меньше чем на min(w, h) / 2:
        # если такой области нет, форма не помещается; иначе перебираем позиции только в границах этой области
        sd = min(w, h) / 2.0 - 1e-6
        if sd not in eroded:
            eroded[sd] = zone.buffer(-sd)
        ez = eroded[sd]
        if ez.is_empty:
            continue
        shapely.prepare(ez)
        tz = None
        if touch is not None:                      # пятно дальше reach от проезда отбрасывается: центр обязан быть не дальше reach + полудиагональ
            tz = touch.buffer(p.parking_road_reach + p.touch_tol + hypot(w, h) / 2.0)
            shapely.prepare(tz)
        ex0, ey0, ex1, ey1 = ez.bounds
        x = minx
        while x + w / 2.0 < ex0 - 1e-9:
            x += step
        while x + w <= maxx and x + w / 2.0 <= ex1 + 1e-9:
            y = miny
            while y + h / 2.0 < ey0 - 1e-9:
                y += step
            while y + h <= maxy and y + h / 2.0 <= ey1 + 1e-9:
                cx_, cy_ = x + w / 2.0, y + h / 2.0
                if not shapely.contains_xy(ez, cx_, cy_) or (tz is not None and not shapely.contains_xy(tz, cx_, cy_)):
                    y += step
                    continue
                r = box(x, y, x + w, y + h)
                if pz.contains(r):
                    if touch is not None:
                        # прижимаем пятно к проезду: сдвиг по оси наибольшей составляющей до касания
                        dt = r.distance(touch)
                        if dt > p.parking_road_reach:
                            y += step
                            continue
                        if dt > p.touch_tol:
                            a_, b_ = nearest_points(r, touch)
                            vx, vy = b_.x - a_.x, b_.y - a_.y
                            if abs(vx) >= abs(vy):
                                x2, y2 = x + vx, y
                            else:
                                x2, y2 = x, y + vy
                            r2 = box(x2, y2, x2 + w, y2 + h)
                            if not pz.contains(r2) or r2.distance(touch) > p.touch_tol:
                                y += step
                                continue
                            r = r2
                            xs_, ys_ = x2, y2
                        else:
                            xs_, ys_ = x, y
                        d = r.distance(target)
                        if d < best_d:
                            best, best_d = (xs_, ys_, cols, rows, rot), d
                        y += step
                        continue
                    d = r.distance(target)
                    if d < best_d:
                        best, best_d = (x, y, cols, rows, rot), d
                y += step
            x += step
    if best is None:
        return None
    x, y, cols, rows, rot = best
    return _patch_cells(x, y, n, cols, rows, rot, p)


def _place_for_house(n, free, bu_u, taken, target, p, guest=False, depth=0, near=None, edu_u=None):
    """Размещает n мест у дома одним пятном; если не помещается, делит пополам (не глубже 2 раз).
    Возвращает (список пятен-ячеек, размещено мест, отступ от окон, м)."""
    if n <= 0:
        return [], 0, 0.0
    sb = p.guest_gap if guest else _setback(n, p.window_setback)
    zone = free
    if bu_u is not None:
        zone = zone.difference(bu_u.buffer(sb))
    if edu_u is not None and not edu_u.is_empty and not guest:   # школы и детские сады: 25/50 м (вторичный источник)
        zone = zone.difference(edu_u.buffer(_setback(n, p.edu_setback)))
    # пешеходная доступность: гостевые ≤200 м до дома (СП 42 п. 8.125), постоянные ≤800 м (п. 8.113)
    zone = zone.intersection(target.buffer(p.walk_guest_m if guest else p.walk_permanent_m))
    if taken is not None and not taken.is_empty:
        zone = zone.difference(taken.buffer(p.patch_gap))
    cells = None
    if near is not None and not near.is_empty:
        if p.parking_layout == "along_roads":            # пятно примыкает к проезду: отдельный заезд не нужен
            cells = _find_patch(n, zone, target, p, touch=near)
        if cells is None:
            cells = _find_patch(n, zone.intersection(near.buffer(p.parking_road_reach)), target, p)
    if cells is None:
        cells = _find_patch(n, zone, target, p)
    if cells is not None:
        return [cells], n, sb
    if depth >= 2 or n < 2:
        return [], 0, sb
    h1 = n // 2
    p1, k1, s1 = _place_for_house(h1, free, bu_u, taken, target, p, guest, depth + 1, near, edu_u)
    t2 = unary_union([taken] + [c for pc in p1 for c in pc]) if (taken is not None and p1) else (unary_union([c for pc in p1 for c in pc]) if p1 else taken)
    p2, k2, s2 = _place_for_house(n - h1, free, bu_u, t2, target, p, guest, depth + 1, near, edu_u)
    return p1 + p2, k1 + k2, max(s1, s2)


def _place_trees(lawn, buildings_u, parking_u, p, carriageway=None, walkways=None, fire_zone=None):
    wall = p.tree_wall_buffer if p.tree_wall_buffer is not None else max(5.0, p.tree_crown_d / 2)
    step = p.tree_spacing if p.tree_spacing is not None else p.tree_crown_d
    zone = lawn
    if buildings_u is not None:
        zone = zone.difference(buildings_u.buffer(wall))
    if fire_zone is not None and not fire_zone.is_empty:       # СП 4.13130 п. 8.1.2: нет деревьев между проездом и стеной
        zone = zone.difference(fire_zone)
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


def _place_shrubs(lawn, buildings, patches_geoms, road_u, walkways, p):
    """Кустарники: пояс вдоль стен зданий и живая изгородь вдоль пятен парковки (СП 42 табл. 6.3 по расстояниям).

    Возвращает список {geometry, species, species_lat, crown_d}. Виды чередуются по зданиям и пятнам."""
    if not p.shrubs_enabled or lawn is None or lawn.is_empty:
        return []
    table = [t for t in p.shrub_species if t[0] in p.shrub_species_ids]
    if not table:
        return []
    zone = lawn
    if buildings:
        zone = zone.difference(unary_union(buildings).buffer(p.shrub_wall_buffer))
    if road_u is not None and not road_u.is_empty:
        zone = zone.difference(road_u.buffer(p.shrub_road_buffer))
    if patches_geoms:
        zone = zone.difference(unary_union(patches_geoms).buffer(p.shrub_parking_buffer))
    if walkways is not None and not walkways.is_empty:
        zone = zone.difference(walkways.buffer(p.shrub_parking_buffer))
    if zone.is_empty:
        return []
    out, grid = [], {}

    def free_of_neighbours(pt, step):
        cx, cy = int(pt.x // step), int(pt.y // step)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for q in grid.get((round(step, 2), gx, gy), ()):
                    if q.distance(pt) < 0.8 * step:
                        return False
        return True

    def along(line_geom, tab):
        sid, name, lat, crown, step = tab
        pieces = [line_geom] if line_geom.geom_type == "LineString" else list(getattr(line_geom, "geoms", []))
        for ln in pieces:
            if ln.geom_type != "LineString" or ln.length < step * 0.5:
                continue
            d = step / 2.0
            while d <= ln.length:
                pt = ln.interpolate(d)
                if zone.contains(pt) and free_of_neighbours(pt, step):
                    out.append({"geometry": pt, "species": name, "species_id": sid, "species_lat": lat, "crown_d": crown})
                    grid.setdefault((round(step, 2), int(pt.x // step), int(pt.y // step)), []).append(pt)
                d += step

    k = 0
    if p.shrub_wall_belt:
        for b in buildings:
            ring = b.buffer(p.shrub_belt_offset).boundary
            along(ring.intersection(zone), table[k % len(table)])
            k += 1
    if p.shrub_parking_hedge:
        for g in patches_geoms:
            ring = g.buffer(p.shrub_hedge_offset).boundary
            along(ring.intersection(zone), table[k % len(table)])
            k += 1
    return out


def generate_landscape(block: Polygon, buildings: list, residents: float, p: LandscapeParams | None = None,
                       carriageway=None, walkways=None, house_residents: list | None = None, roads=None, road_inputs: dict | None = None,
                       services: list | None = None):
    """Парковки привязаны к домам: у каждого дома своё пятно постоянных мест и своё пятно гостевых.

    parking_type: underground (постоянные только под землёй), surface (только наземные; нехватка
    места показывается как дефицит), mixed (доля подземных подбирается минимальной).
    """
    p = p or LandscapeParams()
    d = demand(residents, p)
    bu = unary_union(buildings) if buildings else None
    free = block.difference(bu) if bu is not None else block
    edu_idx = [i for i, nm in enumerate(services or []) if i < len(buildings) and any(k in (nm or "").lower() for k in ("школа", "детский сад"))]
    edu_u = unary_union([buildings[i] for i in edu_idx]) if edu_idx else None
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
            pc, placed, sb = _place_for_house(perm_surf, free, bu, taken, buildings[i], p, near=road_u, edu_u=edu_u)
            for c in pc:
                patches.append({"house_index": i, "kind": "surface", "cells": c})
            if pc:
                taken = unary_union(([taken] if taken is not None else []) + [x for c in pc for x in c])
            gc, gplaced, _ = _place_for_house(hd[i]["guest_places"], free, bu, taken, buildings[i], p, guest=True, near=road_u, edu_u=edu_u)
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
    if roads is not None and road_inputs is not None and patches:
        # второй проход: к каждому пятну парковки подводится проезд; газон и деревья считаются по итоговой сети
        from app.logic.roads import generate_roads
        roads = generate_roads(block, buildings, road_inputs["floors"], road_inputs.get("services"),
                               road_inputs["params"], patches=[pt["geometry"] if "geometry" in pt else unary_union(pt["cells"]) for pt in patches])
        parts = [c["geometry"] for c in roads["carriageways"]] + list(roads["turnarounds"])
        parts = [g for g in parts if g is not None and not g.is_empty]
        road_u = unary_union(parts) if parts else None
        free_nr = block.difference(bu) if bu is not None else block
        lawn = free_nr.difference(road_u) if road_u is not None else free_nr
        if parking_u is not None:
            lawn = lawn.difference(parking_u)
        lawn_area = lawn.area - d["platforms_reserved_m2"]
        carriageway = road_u
    fire_zone = None
    if roads is not None and road_inputs is not None and carriageway is not None and not carriageway.is_empty:
        rp_ = road_inputs["params"]
        zs_ = []
        for g_, f_ in zip(buildings, road_inputs["floors"]):
            lo_, hi_ = rp_.wall_gap_high if (float(f_) * rp_.floor_height) > rp_.high_building_m else rp_.wall_gap_low
            # полоса между стеной и проездом: не дальше hi от стены и не дальше (hi-lo)+запас от края проезда
            zs_.append(g_.buffer(hi_).intersection(carriageway.buffer(hi_ - lo_ + 1.0)))
        fire_zone = unary_union(zs_) if zs_ else None
    trees, wall, step = _place_trees(lawn, bu, parking_u, p, carriageway, walkways, fire_zone)
    shrubs = _place_shrubs(lawn, buildings, [unary_union(pt["cells"]) for pt in patches], carriageway, walkways, p)
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
        "shrubs": len(shrubs),
        "shrubs_by_species": {n_: sum(1 for x in shrubs if x["species"] == n_) for n_ in sorted({x["species"] for x in shrubs})},
        "houses": houses_diag,
        "disclaimer": DISCLAIMER,
        "assumptions": ["shrub sizes and step by species", "shrub belt/hedge offsets", "tree_crown_d", "tree_wall_buffer", "tree_spacing", "cell 2.5x10", "window_setback (SanPiN, secondary)",
                        "guest_gap", "patch_gap", "underground parking: places only, area norm not set"],
    }
    # parking_cells: по одному полигону на пятно (объединение ячеек) — для деревьев и вывода
    cells_out = [unary_union(pt["cells"]) for pt in patches]
    return {"parking_cells": cells_out, "parking_patches": [
        {"house_index": pt["house_index"], "kind": pt["kind"], "places": len(pt["cells"]), "geometry": unary_union(pt["cells"])}
        for pt in patches], "lawn": lawn, "trees": trees, "shrubs": shrubs, "diagnostics": diag, "roads": roads}


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
        road_style=gen_params.road_style,
        road_turn_penalty=float(gen_params.road_turn_penalty_m),
        road_junction_penalty=float(gen_params.road_junction_penalty_m),
        road_reuse_cost=float(gen_params.road_reuse_cost),
        parking_road_reach=float(gen_params.parking_reach_m),
        parking_layout=gen_params.parking_layout,
        shrubs_enabled=gen_params.shrubs_enabled,
        shrub_species_ids=tuple(gen_params.shrub_species),
        shrub_wall_belt=gen_params.shrub_wall_belt,
        shrub_parking_hedge=gen_params.shrub_parking_hedge,
        max_lot_places=int(gen_params.parking_max_lot_places),
        norm_mode=gen_params.norm_mode,
    )


def road_params(p: "LandscapeParams"):
    """RoadParams из LandscapeParams (единое место передачи параметров проездов)."""
    from app.logic.roads import RoadParams
    return RoadParams(floor_height=p.floor_height, road_style=p.road_style, turn_penalty=p.road_turn_penalty,
                      junction_penalty=p.road_junction_penalty, reuse_cost=p.road_reuse_cost,
                      patch_reach=p.parking_road_reach)


def apply_landscape(blocks_utm, buildings_utm, la_per_person: float, p: LandscapeParams):
    """Считает озеленение и парковки для кварталов с жильём.

    blocks_utm: GeoDataFrame кварталов (UTM, метры) с колонками geometry, zone.
    buildings_utm: GeoDataFrame сгенерированных зданий (UTM) с колонками geometry, living_area.
    Возвращает словарь {"greenery": GeoDataFrame, "parking": GeoDataFrame, "diagnostics": list[dict]}
    в той же системе координат (UTM); перепроецирование делает вызывающий код.
    """
    import geopandas as gpd

    greenery_rows, parking_rows, road_rows, diag = [], [], [], []
    building_rows = {}
    for bid, (_, block) in enumerate(blocks_utm.iterrows()):
        geom = block.geometry
        if geom is None or geom.is_empty:
            continue
        inside = buildings_utm[buildings_utm.geometry.representative_point().within(geom)]
        la = float(inside["living_area"].fillna(0).sum()) if "living_area" in inside.columns else 0.0
        residents = la / la_per_person if la_per_person else 0.0
        if residents <= 0:
            continue
        inside_valid = [(ix, g) for ix, g in zip(inside.index, inside.geometry) if g is not None and not g.is_empty]
        footprints = [g for _, g in inside_valid]
        b_index = [ix for ix, _ in inside_valid]
        las = [float(v) if v == v else 0.0 for v in inside["living_area"]] if "living_area" in inside.columns else []
        hres = [x / la_per_person for x in las] if las and len(las) == len(footprints) else None
        roads = None
        fl = [float(v) if v == v else 1.0 for v in inside["floors_count"]] if "floors_count" in inside.columns else [1.0] * len(footprints)
        sv = [(x if isinstance(x, str) else "") for x in inside["service"]] if "service" in inside.columns else None
        if p.roads_enabled:
            from app.logic.roads import generate_roads
            roads = generate_roads(geom, footprints, fl, sv, road_params(p))
            for c in roads["carriageways"]:
                road_rows.append({"block_index": bid, "kind": "carriageway", "width_m": c["width"],
                                  "road_class": c["class"], "geometry": c["geometry"]})
            for t in roads["turnarounds"]:
                road_rows.append({"block_index": bid, "kind": "turnaround", "width_m": 15.0,
                                  "road_class": "turnaround", "geometry": t})
        res = generate_landscape(geom, footprints, residents, p, house_residents=hres, roads=roads, services=sv,
                                 road_inputs=({"floors": fl, "services": sv, "params": road_params(p)} if roads is not None else None))
        roads = res.get("roads")
        if roads is not None:
            res["diagnostics"]["roads"] = roads["diagnostics"]
            road_rows[:] = [r_ for r_ in road_rows if r_["block_index"] != bid]
            for c in roads["carriageways"]:
                road_rows.append({"block_index": bid, "kind": "carriageway", "width_m": c["width"], "road_class": c["class"], "geometry": c["geometry"]})
            for t in roads["turnarounds"]:
                road_rows.append({"block_index": bid, "kind": "turnaround", "width_m": 15.0, "road_class": "turnaround", "geometry": t})
        d = res["diagnostics"]
        from app.logic.layout_checks import check_layout
        fl_ = [float(v) if v == v else 1.0 for v in inside["floors_count"]] if "floors_count" in inside.columns else [1.0] * len(footprints)
        d["checks"] = check_layout(geom, footprints, fl_, sv, roads, res, p)
        d["checks_summary"] = {k: sum(1 for c in d["checks"] if c["status"] == k) for k in ("pass", "unverified", "warn", "fail")}
        d["block_index"] = bid
        d["residents"] = round(residents)
        diag.append(d)
        lawn = res["lawn"]
        if not lawn.is_empty:
            greenery_rows.append({"block_index": bid, "kind": "lawn", "geometry": lawn})
        for t in res["trees"]:
            greenery_rows.append({"block_index": bid, "kind": "tree", "geometry": t})
        for sh in res.get("shrubs", []):
            greenery_rows.append({"block_index": bid, "kind": "shrub", "species": sh["species"],
                                  "species_lat": sh["species_lat"], "crown_d_m": sh["crown_d"], "geometry": sh["geometry"]})
        for pt in res["parking_patches"]:
            parking_rows.append({"block_index": bid, "house_index": pt["house_index"],
                                 "kind": "surface" if pt["kind"] == "surface" else "guest",
                                 "places": pt["places"], "geometry": pt["geometry"]})
        for hd_ in d.get("houses", []):                      # атрибуты зданий: парковка дома
            ix = b_index[hd_["house_index"]] if hd_["house_index"] < len(b_index) else None
            if ix is None:
                continue
            ug = int(hd_["permanent_underground"])
            sf = int(hd_["permanent_surface"])
            building_rows[ix] = {
                "parking_class": "underground" if ug > 0 and sf == 0 else ("mixed" if ug > 0 else ("surface" if sf > 0 else "none")),
                "has_underground_parking": ug > 0,
                "underground_parking_places": ug,
                "surface_parking_places": sf,
                "guest_parking_places": int(hd_["guest_places"]),
            }
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
        "buildings": pd.DataFrame.from_dict(building_rows, orient="index") if building_rows else pd.DataFrame(),
        "diagnostics": diag,
    }

"""
Проверки результата генерации проездов, парковок и озеленения (эскизный расчёт).

Каждая проверка возвращает: id, title, status (pass | fail | warn), checked, violations (до 10 примеров), note.
fail — нарушено правило из норм или внутренний инвариант; warn — информационное замечание (допущение или ограничение модели).
Используется в apply_landscape (поле diagnostics["checks"]) и в тестах.
"""
from shapely.ops import unary_union

from app.logic.landscape import _setback
from app.logic.roads import RoadParams, fire_width_for


# основание каждой проверки: norm — текст СП 42.13330 проверен; secondary — вторичный источник; assumption — допущение модели
BASIS = {
    "road_access_distance": "secondary", "road_width_fire": "secondary", "dead_end_limit": "norm",
    "roads_connected": "assumption", "turnaround_free": "secondary", "roads_no_overlap": "assumption",
    "parking_near_road": "assumption", "parking_window_setback": "secondary", "parking_balance": "norm", "shrubs_buffers": "norm", "parking_walk_distance": "norm", "parking_edu_setback": "secondary",
    "trees_buffers": "assumption", "green_norm": "secondary", "near_street_axis": "assumption",
}
BASIS_NOTE = {"norm": "норма (СП 42, текст проверен)", "secondary": "вторичный источник, первоисточник не сверен",
              "assumption": "допущение модели, не норма"}


def _res(cid, title, checked, viol, note="", warn_only=False):
    status = "pass" if not viol else ("warn" if warn_only else "fail")
    basis = BASIS.get(cid, "assumption")
    return {"id": cid, "title": title, "status": status, "checked": checked, "basis": basis,
            "basis_note": BASIS_NOTE[basis], "violations_count": len(viol), "violations": viol[:10], "note": note}


def _apply_norm_mode(out, mode):
    """strict: пройденная проверка, опирающаяся не на проверенный текст СП 42, помечается unverified."""
    if mode == "strict":
        for c in out:
            if c["status"] == "pass" and c["basis"] != "norm":
                c["status"] = "unverified"
    return out


def check_layout(block, footprints, floors, services, roads, res, p, rp: RoadParams | None = None):
    rp = rp or RoadParams(floor_height=p.floor_height)
    out = []
    n = len(footprints)
    heights = [(float(f) if f == f and f else 1.0) * rp.floor_height for f in floors]
    tol = rp.cell
    patches = res.get("parking_patches", [])
    trees = res.get("trees", [])
    diag = res["diagnostics"]
    bu = unary_union(footprints) if footprints else None

    road_u = None
    carr = []
    if roads is not None:
        carr = roads.get("carriageways", [])
        parts = [c["geometry"] for c in carr] + list(roads.get("turnarounds", []))
        parts = [g for g in parts if g is not None and not g.is_empty]
        road_u = unary_union(parts) if parts else None

        # 1. у каждого здания есть проезд на нормативном расстоянии от стены
        viol = []
        carr_u = unary_union([c["geometry"] for c in carr if c["geometry"] is not None and not c["geometry"].is_empty]) if carr else None
        for i, g in enumerate(footprints):
            hi = rp.wall_gap_high[1] if heights[i] > rp.high_building_m else rp.wall_gap_low[1]
            lo = rp.wall_gap_high[0] if heights[i] > rp.high_building_m else rp.wall_gap_low[0]
            d = g.distance(carr_u) if carr_u is not None else 1e9
            if d > hi + tol or d < lo - 0.5:
                viol.append({"building": i, "distance_to_road_m": round(d, 1), "allowed_m": [lo, hi]})
        out.append(_res("road_access_distance", "Проезд у каждого здания на расстоянии 5-8 м (выше 28 м: 8-10 м) от стены", n, viol,
                        "СП 4.13130 (вторичный источник); допуск на растр 2,5 м"))

        # 2. ширина проезда не меньше пожарной для ближайшего здания
        viol = []
        for i, g in enumerate(footprints):
            fw = fire_width_for(heights[i], rp)
            near = [c["width"] for c in carr if c["geometry"] is not None and not c["geometry"].is_empty and g.distance(c["geometry"]) <= 12.0]
            if not near or max(near) + 1e-6 < fw:
                viol.append({"building": i, "need_m": fw, "have_m": max(near) if near else None})
        out.append(_res("road_width_fire", "Ширина проезда у здания не меньше пожарной по высоте", n, viol,
                        "3,5 / 4,2 / 6,0 м (вторичный источник)"))

        # 3. тупики не длиннее 150 м
        dv = roads["diagnostics"].get("dead_end_violations", [])
        out.append(_res("dead_end_limit", "Тупиковые проезды не длиннее 150 м (СП 42 п. 8.49)", roads["diagnostics"].get("turnarounds", 0), dv,
                        f"наибольший тупик {roads['diagnostics'].get('max_dead_end_m')} м"))

        # 4. все проезды связаны с улицей (границей квартала)
        viol = []
        geoms = list(road_u.geoms) if road_u is not None and hasattr(road_u, "geoms") else ([road_u] if road_u is not None else [])
        for k, gm in enumerate(geoms):
            if gm.distance(block.boundary) > 2 * tol:
                viol.append({"component": k, "distance_to_street_m": round(gm.distance(block.boundary), 1)})
        out.append(_res("roads_connected", "Каждый участок сети проездов выходит на улицу (границу квартала)", len(geoms), viol))

        # 5. разворотные площадки свободны
        viol = []
        for k, t in enumerate(roads.get("turnarounds", [])):
            if bu is not None and not t.is_empty and t.distance(bu) < rp.wall_gap_low[0] - 0.5:
                viol.append({"turnaround": k, "reason": "closer than 5 m to wall", "m": round(t.distance(bu), 1)})
            if bu is not None and t.intersection(bu).area > 1.0:
                viol.append({"turnaround": k, "reason": "overlaps building"})
            for pt in patches:
                if t.intersection(pt["geometry"]).area > 1.0:
                    viol.append({"turnaround": k, "reason": "parking on turnaround (СП 42 п. 8.50)"})
                    break
        out.append(_res("turnaround_free", "Разворотные площадки свободны от зданий и стоянок", len(roads.get("turnarounds", [])), viol))

        # 6. проезды не пересекают здания и парковки
        viol = []
        if bu is not None and road_u is not None and road_u.intersection(bu).area > 1.0:
            viol.append({"reason": "roads overlap buildings", "area_m2": round(road_u.intersection(bu).area, 1)})
        if road_u is not None:
            for k, pt in enumerate(patches):
                a = pt["geometry"].intersection(road_u).area
                if a > 1.0:
                    viol.append({"patch": k, "reason": "parking overlaps road", "area_m2": round(a, 1)})
        out.append(_res("roads_no_overlap", "Проезды не пересекают здания и парковки", len(patches) + 1, viol))

        # 7. парковка рядом с проездом (информационно)
        far = []
        for k, pt in enumerate(patches):
            if pt["kind"] == "surface" and pt["geometry"].distance(road_u) > p.parking_road_reach + tol:
                far.append({"patch": k, "house": pt["house_index"], "distance_to_road_m": round(pt["geometry"].distance(road_u), 1)})
        out.append(_res("parking_near_road", "Пятна парковки не дальше 12 м от проезда", sum(1 for x in patches if x["kind"] == "surface"), far,
                        "допущение parking_road_reach", warn_only=True))

    # 8. разрывы стоянок от окон (СанПиН, вторичный источник)
    viol = []
    for k, pt in enumerate(patches):
        if pt["kind"] != "surface":
            continue
        need = _setback(pt["places"], p.window_setback)
        d = min((g.distance(pt["geometry"]) for g in footprints), default=1e9)
        if d + 0.01 < need:
            viol.append({"patch": k, "house": pt["house_index"], "places": pt["places"], "distance_m": round(d, 1), "need_m": need})
    out.append(_res("parking_window_setback", "Разрыв стоянки до окон по числу мест (СанПиН, вторичный источник)",
                    sum(1 for x in patches if x["kind"] == "surface"), viol,
                    "крупное пятно делится на части, разрыв считается по числу мест части (допущение)"))

    # 8а. пешеходный подход от стоянки до дома: гостевые ≤200 м (СП 42 п. 8.125), постоянные ≤800 м (п. 8.113)
    viol = []
    for k, pt in enumerate(patches):
        lim = p.walk_guest_m if pt["kind"] == "guest" else p.walk_permanent_m
        hi = pt["house_index"]
        dd = footprints[hi].distance(pt["geometry"]) if hi < n else 0.0
        if dd > lim:
            viol.append({"patch": k, "house": hi, "distance_m": round(dd, 1), "limit_m": lim})
    out.append(_res("parking_walk_distance", "Пешеходный подход от стоянки до дома: гостевые до 200 м, постоянные до 800 м (СП 42 п. 8.125, 8.113)",
                    len(patches), viol, "расстояние до контура дома, а не до входа (допущение: входы не заданы)"))

    # 8б. разрывы стоянок до школ и детских садов (СанПиН, вторичный источник): 25 м до 10 мест, 50 м больше
    edu = [i for i, nm in enumerate(services or []) if i < n and any(k in (nm or "").lower() for k in ("школа", "детский сад"))]
    viol = []
    for k, pt in enumerate(patches):
        if pt["kind"] != "surface":
            continue
        need = _setback(pt["places"], p.edu_setback)
        for i in edu:
            dd = footprints[i].distance(pt["geometry"])
            if dd + 0.01 < need:
                viol.append({"patch": k, "edu_building": i, "places": pt["places"], "distance_m": round(dd, 1), "need_m": need})
    out.append(_res("parking_edu_setback", "Разрыв открытой стоянки до школ и детских садов 25/50 м (СанПиН, вторичный источник)",
                    sum(1 for x in patches if x["kind"] == "surface") * max(len(edu), 1) if edu else 0, viol,
                    "применяется только к постоянным местам; для гостевых разрывы не устанавливаются (вторичный источник)"))

    # 9. баланс мест по домам
    viol = []
    for h in diag.get("houses", []):
        if h["permanent_surface"] + h["permanent_underground"] < h["permanent_places"] or h["guest_places"] < h["guest_required"]:
            viol.append({"house": h["house_index"], "permanent": h["permanent_places"], "placed_surface": h["permanent_surface"],
                         "underground": h["permanent_underground"], "guest": [h["guest_places"], h["guest_required"]]})
    out.append(_res("parking_balance", "Для каждого дома размещены все постоянные и гостевые места", len(diag.get("houses", [])), viol))

    # 10. деревья: буферы
    viol = []
    wall = diag.get("tree_wall_buffer_m", 5.0)
    pu = unary_union([x["geometry"] for x in patches]) if patches else None
    for k, t in enumerate(trees):
        if bu is not None and t.distance(bu) < wall - 0.01:
            viol.append({"tree": k, "reason": "closer to wall than buffer", "m": round(t.distance(bu), 1)})
        elif road_u is not None and t.distance(road_u) < 2.0 - 0.01:
            viol.append({"tree": k, "reason": "closer than 2 m to carriageway (СП 42 табл. 6.3)", "m": round(t.distance(road_u), 1)})
        elif pu is not None and t.distance(pu) < 0.7 - 0.01:
            viol.append({"tree": k, "reason": "closer than 0.7 m to parking", "m": round(t.distance(pu), 1)})
    out.append(_res("trees_buffers", "Деревья: 5,0 м (по кроне) от стен, 2,0 м от проезжей части, 0,7 м от стоянок", len(trees), viol,
                    "2,0 и 0,7 м — СП 42 табл. 6.3; отступ от стен по кроне — допущение"))

    # 10а. кустарники: 1,5 м от стен, 1,0 м от проезжей части, 0,5 м от стоянок (СП 42 табл. 6.3, текст проверен)
    viol = []
    shrubs = res.get("shrubs", [])
    for k, sh in enumerate(shrubs):
        g = sh["geometry"]
        if bu is not None and g.distance(bu) < p.shrub_wall_buffer - 0.01:
            viol.append({"shrub": k, "reason": "closer than 1.5 m to wall", "m": round(g.distance(bu), 2)})
        elif road_u is not None and g.distance(road_u) < p.shrub_road_buffer - 0.01:
            viol.append({"shrub": k, "reason": "closer than 1.0 m to carriageway", "m": round(g.distance(road_u), 2)})
        elif pu is not None and g.distance(pu) < p.shrub_parking_buffer - 0.01:
            viol.append({"shrub": k, "reason": "closer than 0.5 m to parking", "m": round(g.distance(pu), 2)})
    out.append(_res("shrubs_buffers", "Кустарники: 1,5 м от стен, 1,0 м от проезжей части, 0,5 м от стоянок (СП 42 табл. 6.3)", len(shrubs), viol,
                    "размеры кустов по видам и шаг — допущения; расстояния — норма"))

    # 11. норма озеленения
    out.append(_res("green_norm", "Площадь газона не меньше нормы (3,0 м2/чел. по СП 476, сверить)", 1,
                    [] if diag.get("green_deficit_m2", 0) <= 0 else [{"deficit_m2": diag["green_deficit_m2"]}]))

    # 12. информация: дома близко к оси улицы (улица не нарисована, ширина неизвестна)
    viol = []
    for i, g in enumerate(footprints):
        d = g.distance(block.boundary)
        if d < 8.0:
            viol.append({"building": i, "distance_to_street_axis_m": round(d, 1)})
    out.append(_res("near_street_axis", "Здания ближе 8 м к границе квартала (оси улицы)", n, viol,
                    "граница квартала = ось улицы; проезжая часть улицы не вычитается (допущение), реальный отступ от красной линии не проверяется",
                    warn_only=True))
    return _apply_norm_mode(out, getattr(p, "norm_mode", "extended"))

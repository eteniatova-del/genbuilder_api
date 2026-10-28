"""Единый список предупреждений о нехватке места и нарушениях нормативов в результате генерации.

Собирает сообщения из полей ответа (`capacity_diagnostics`, `service_diagnostics`, `landscape_diagnostics`),
чтобы клиенту не нужно было разбирать каждое из них отдельно. Новых расчётов не делает.
"""
from typing import Any, Optional

ERROR, WARNING = "error", "warning"


def _w(code: str, severity: str, message: str, source: str, block: Optional[int] = None, details: Any = None) -> dict:
    item = {"code": code, "severity": severity, "message": message, "source": source}
    if block is not None:
        item["block_index"] = block
    if details:
        item["details"] = details
    return item


def build_warnings(
    capacity: Optional[dict] = None,
    services: Optional[dict] = None,
    landscape_blocks: Optional[list] = None,
) -> list[dict]:
    out: list[dict] = []

    if capacity and capacity.get("status") == "insufficient_area":
        out.append(_w("residents_not_placed", ERROR, capacity.get("warning") or "Не все жители размещены.",
                      "capacity_diagnostics",
                      details={"requested": capacity.get("residents_requested"),
                               "placed": capacity.get("residents_placed")}))

    if services and services.get("status") in ("partial", "not_placed"):
        by_reason = services.get("unplaced_by_reason") or {}
        names = sorted({n for lst in by_reason.values() for n in (lst or [])})
        msg = services.get("warning") or "Не все сервисы удалось разместить."
        if names:
            msg += " Не размещены: " + ", ".join(names) + "."
        out.append(_w("services_not_placed", ERROR, msg, "service_diagnostics",
                      details={"unplaced_by_reason": by_reason, "capacity_unplaced": services.get("capacity_unplaced")}))

    for b in landscape_blocks or []:
        k = b.get("block_index")
        if (b.get("green_deficit_m2") or 0) > 0:
            out.append(_w("green_deficit", ERROR,
                          f"Квартал {k}: не хватает озеленения {round(b['green_deficit_m2'])} м² до нормы.",
                          "landscape_diagnostics", k))
        if (b.get("parking_deficit_surface") or 0) > 0:
            items = [u for u in (b.get("unplaced") or []) if str(u.get("object", "")).startswith("parking")]
            by_house = "; ".join(
                f"дом {u.get('house_index')}: {'гостевые' if u['object'] == 'parking_guest' else 'постоянные'} {u['placed']} из {u['required']}"
                for u in items)
            out.append(_w("parking_deficit", ERROR,
                          f"Квартал {k}: не размещено {b['parking_deficit_surface']} наземных машино-мест" + (f" ({by_house})." if by_house else "."),
                          "landscape_diagnostics", k, {"unplaced": items} if items else None))
        roads = b.get("roads") or {}
        if roads.get("unreachable"):
            out.append(_w("buildings_without_road", ERROR,
                          f"Квартал {k}: к {len(roads['unreachable'])} зданиям не удалось подвести проезд.",
                          "landscape_diagnostics", k, {"buildings": roads["unreachable"]}))
        if roads.get("parking_unlinked"):
            out.append(_w("parking_not_linked", ERROR,
                          f"Квартал {k}: {len(roads['parking_unlinked'])} пятен парковки не удалось соединить с проездом.",
                          "landscape_diagnostics", k, {"patches": roads["parking_unlinked"]}))
        if roads.get("turnaround_violations"):
            out.append(_w("turnaround_too_close", WARNING,
                          f"Квартал {k}: {len(roads['turnaround_violations'])} разворотных площадок ближе 5 м к стене.",
                          "landscape_diagnostics", k, {"turnarounds": roads["turnaround_violations"]}))
        if roads.get("dead_end_violations"):
            out.append(_w("dead_end_too_long", WARNING,
                          f"Квартал {k}: {len(roads['dead_end_violations'])} тупиков длиннее предела.",
                          "landscape_diagnostics", k, {"dead_ends": roads["dead_end_violations"]}))
        for c in b.get("checks") or []:
            if c.get("status") in ("fail", "warn"):
                msg = f"Квартал {k}: {c.get('title') or c.get('id')}: нарушений {c.get('violations_count', 0)}."
                det = {"check": c.get("id"), "status": c.get("status")}
                if c.get("id") == "fire_access_sides":
                    det["violations"] = c.get("violations")
                    why = "; ".join(
                        f"здание {v.get('building')}: охват сторон {v.get('coverage')}" +
                        (", мешает здание " + ", ".join(f"{x['building']} ({x['distance_m']} м)" for x in v["blocked_by"]) if v.get("blocked_by") else "")
                        for v in (c.get("violations") or []))
                    if why:
                        msg += " " + why + "."
                out.append(_w("check_" + str(c.get("id")), ERROR if c["status"] == "fail" else WARNING, msg,
                              "landscape_diagnostics.checks", k, det))
    return out

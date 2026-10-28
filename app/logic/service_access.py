"""Проверка территориальной доступности сервисов от жилых домов (по прямой).

Норма: СП 42.13330.2026, прил. Л, табл. Л.1, столбец «Территориальная доступность» (текст официального издания
сверен): детские сады (п. 1) и школы (п. 2) — в городских населённых пунктах не более 500 м пешеходной доступности
(800 м в стеснённых условиях застройки), поликлиники (п. 17) — 1000 м, но не более 60 мин пешеходной доступности
(сноска 17). Сноска 2: для климатических подрайонов IА, IБ, IГ, IД и IIА показатели уменьшаются на 30 % (кроме
дошкольных и общеобразовательных организаций).

ДОПУЩЕНИЕ: норма говорит о пешеходной доступности, а модель считает расстояние по прямой от стены жилого дома
до стены сервиса (входы и пешеходные пути не заданы). Результат — предварительная оценка, не проверка маршрута.
Проверка считается по всему участку (сервис может стоять в соседнем квартале).
"""
from typing import Optional

from shapely.geometry import shape
from shapely.ops import unary_union

# имя сервиса -> предельное расстояние, м
SERVICE_ACCESS_M = {"Детский сад": 500.0, "Школа": 500.0, "Поликлиника": 1000.0}
CONSTRAINED_M = {"Детский сад": 800.0, "Школа": 800.0}


def check_service_access(buildings_fc: Optional[dict], limits: Optional[dict] = None, constrained: bool = False) -> dict:
    """buildings_fc — GeoJSON (WGS84) со зданиями; у сервисов задано свойство `service`.

    Для каждого вида сервиса: жилые дома, до ближайшего здания этого вида которых дальше предела.
    Если сервиса данного вида на участке нет, он не проверяется (отсутствие сервисов отражается в service_diagnostics).
    """
    import geopandas as gpd

    feats = (buildings_fc or {}).get("features") or []
    lim = dict(SERVICE_ACCESS_M)
    if constrained:
        lim.update(CONSTRAINED_M)
    lim.update(limits or {})
    out = {"basis": "norm", "method": "straight_line_wall_to_wall", "limits_m": lim, "services": {}, "violations": []}
    if not feats:
        return out
    gdf = gpd.GeoDataFrame(
        [{"service": (f.get("properties") or {}).get("service")} for f in feats],
        geometry=[shape(f["geometry"]) for f in feats], crs=4326)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    homes = gdf[gdf["service"].isna()]
    for name, limit in lim.items():
        svc = gdf[gdf["service"] == name]
        if svc.empty or homes.empty:
            continue
        su = unary_union(list(svc.geometry))
        dist = homes.geometry.distance(su)
        bad = [(int(i), float(d)) for i, d in dist.items() if d > limit]
        out["services"][name] = {"count": int(len(svc)), "limit_m": limit, "homes_checked": int(len(homes)),
                                 "max_distance_m": round(float(dist.max()), 1), "violations": len(bad)}
        for i, d in bad:
            out["violations"].append({"service": name, "building_index": i, "distance_m": round(d, 1), "limit_m": limit})
    return out

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Hashable

import math
import random
import asyncio

import pandas as pd
import geopandas as gpd
from loguru import logger

from shapely.geometry import Polygon, box, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from shapely.errors import GEOSException
from shapely.validation import make_valid

from app.logic.generation_params import GenParams, ParamsProvider

from app.common.geo_utils import longest_edge_angle_mrr
from app.common.geo_utils import safe_make_valid
from app.common.geo_utils import ensure_crs


class ServiceGenerator:
    """
    Generates non-residential service buildings for blocks by
    converting built living area into per-zone service capacity targets,
    sampling service sites in block free space, and placing template-based
    buildings (from OSM-derived projects) into those sites.
    """

    def __init__(self, params_provider: ParamsProvider):
        self._params = params_provider

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    def compute_service_limits_for_blocks(
        self,
        blocks: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        service_normatives: pd.DataFrame,
    ) -> Dict[Hashable, Dict[str, float]]:

        if "zone" not in blocks.columns:
            raise ValueError("blocks must contain 'zone' column for service limits computation")

        la_per_block = buildings.groupby("src_index", dropna=False)["living_area"].sum()
        blocks = blocks.copy()
        blocks["actual_la_block"] = blocks["src_index"].map(la_per_block).fillna(0.0)
        blocks.loc[blocks["zone"] != "residential", "actual_la_block"] = 0.0
        la_per_zone = (
            blocks.groupby("zone", dropna=False)["actual_la_block"].sum()
        )

        all_limits: Dict[Hashable, Dict[str, float]] = {}

        for zone_name, total_la in la_per_zone.items():
            actual_la = float(total_la)
            if actual_la <= 0.0:
                continue

            people = actual_la / self.generation_parameters.la_per_person

            limits: Dict[str, float] = {}
            for _, srv in service_normatives.iterrows():
                service_name = srv["service_name"]
                cap_per_1000 = float(srv["service_capacity"])
                target_cap = round(cap_per_1000 * (people / 1000.0), 0)
                limits[service_name] = target_cap

            all_limits[zone_name] = limits

        return all_limits

    @staticmethod
    def summarize_service_generation(
        all_limits: Dict[Hashable, Dict[str, float]],
        service_buildings: gpd.GeoDataFrame,
        failure_reasons: Optional[Dict[tuple[Hashable, str], str]] = None,
    ) -> Dict[str, Any]:
        """Summarize requested targets and the capacity actually placed.

        A request is one positive ``(zone, service type)`` capacity target.  A
        request is considered placed only when generated buildings cover its
        full target capacity; partial placements remain visible through the
        capacity and service-building counters.
        """
        targets = [
            (zone, service_name, float(target_capacity))
            for zone, limits in all_limits.items()
            for service_name, target_capacity in limits.items()
            if float(target_capacity) > 0.0
        ]

        placed_capacity_by_target: Dict[tuple[Hashable, str], float] = {}
        if (
            not service_buildings.empty
            and {"zone", "service", "capacity"}.issubset(service_buildings.columns)
        ):
            grouped = service_buildings.groupby(
                ["zone", "service"], dropna=False
            )["capacity"].sum()
            placed_capacity_by_target = {
                (zone, str(service_name)): float(capacity)
                for (zone, service_name), capacity in grouped.items()
            }

        fulfilled = 0
        capacity_requested = 0.0
        capacity_unplaced = 0.0
        unplaced_by_reason = {
            "no_template": [],
            "no_space": [],
            "site_limit": [],
        }
        for zone, service_name, target_capacity in targets:
            placed_capacity = placed_capacity_by_target.get(
                (zone, str(service_name)), 0.0
            )
            capacity_requested += target_capacity
            capacity_unplaced += max(target_capacity - placed_capacity, 0.0)
            if placed_capacity >= target_capacity:
                fulfilled += 1
            else:
                reason = (failure_reasons or {}).get(
                    (zone, str(service_name)), "no_space"
                )
                if reason not in unplaced_by_reason:
                    reason = "no_space"
                unplaced_by_reason[reason].append(str(service_name))

        for names in unplaced_by_reason.values():
            names.sort()

        capacity_placed = 0.0
        if not service_buildings.empty and "capacity" in service_buildings.columns:
            capacity_placed = float(
                pd.to_numeric(service_buildings["capacity"], errors="coerce")
                .fillna(0.0)
                .sum()
            )

        return {
            "services_requested": len(targets),
            "services_placed": fulfilled,
            "services_unplaced": len(targets) - fulfilled,
            "service_buildings_placed": len(service_buildings),
            "capacity_requested": capacity_requested,
            "capacity_placed": capacity_placed,
            "capacity_unplaced": capacity_unplaced,
            "unplaced_no_template": len(unplaced_by_reason["no_template"]),
            "unplaced_no_space": len(unplaced_by_reason["no_space"]),
            "unplaced_site_limit": len(unplaced_by_reason["site_limit"]),
            "unplaced_by_reason": unplaced_by_reason,
        }

    def load_service_projects(self) -> gpd.GeoDataFrame:
        projects_gdf = gpd.read_file(self.generation_parameters.service_projects_file)
        projects_gdf = projects_gdf.to_crs("EPSG:4326")

        expected_cols = [
            "service",
            "type_id",
            "capacity",
            "floors_count",
            "plot_length_min",
            "plot_length_max",
            "plot_width_min",
            "plot_width_max",
            "address",
            "osm_type",
            "osm_id",
            "osm_url",
            "geometry",
        ]
        missing = [c for c in expected_cols if c not in projects_gdf.columns]
        if missing:
            raise ValueError(f"Columns are missing: {missing}.")

        projects_gdf = projects_gdf[expected_cols]
        return projects_gdf

    @staticmethod
    def _get_block_free_area(
        block_geom: BaseGeometry,
        occupied_gdf: gpd.GeoDataFrame,
    ) -> BaseGeometry:
        if block_geom is None or block_geom.is_empty:
            return block_geom

        block_geom_valid = safe_make_valid(block_geom)
        if block_geom_valid is None or block_geom_valid.is_empty:
            return block_geom
        block_geom = block_geom_valid

        if occupied_gdf.empty:
            return block_geom

        cleaned_geoms: List[BaseGeometry] = []
        for g in occupied_gdf.geometry:
            if g is None or g.is_empty:
                continue

            g_valid = safe_make_valid(g)
            if g_valid is None or g_valid.is_empty:
                continue

            cleaned_geoms.append(g_valid)

        if not cleaned_geoms:
            return block_geom

        try:
            occupied_union = unary_union(cleaned_geoms)
        except GEOSException:
            union_geom = cleaned_geoms[0]
            for g in cleaned_geoms[1:]:
                try:
                    union_geom = union_geom.union(g)
                except GEOSException:
                    continue
            occupied_union = union_geom

        try:
            free_area = block_geom.difference(occupied_union)
        except GEOSException:
            block_fixed = make_valid(block_geom)
            occupied_fixed = make_valid(occupied_union)
            free_area = block_fixed.difference(occupied_fixed)

        return free_area

    @staticmethod
    def _compute_main_axis_angle(poly: BaseGeometry) -> float:
        angle = longest_edge_angle_mrr(poly, degrees=True)
        if angle < -90.0:
            angle += 180.0
        elif angle >= 90.0:
            angle -= 180.0

        return angle

    def _sample_rect_in_polygon(
        self,
        poly: BaseGeometry,
        length_range: Tuple[float, float],
        width_range: Tuple[float, float],
        max_attempts: int = 200,
        preferred_angle: Optional[float] = None,
        existing_centroids: Optional[List[Point]] = None,
        min_dist_between_centers: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[Polygon]:
        rng = rng or random.Random()
        if poly.is_empty:
            return None

        minx, miny, maxx, maxy = poly.bounds
        if minx == maxx or miny == maxy:
            return None

        len_min, len_max = length_range
        wid_min, wid_max = width_range

        if preferred_angle is not None:
            base = preferred_angle
            angle_candidates = [
                base,
                base + 5.0,
                base - 5.0,
                base + 15.0,
                base - 15.0,
                base + 90.0,
                base - 90.0,
            ]
        else:
            angle_candidates = [0.0, 90.0, 45.0, -45.0, 30.0, -30.0]

        for _ in range(max_attempts):
            length = rng.uniform(len_min, len_max)
            width = rng.uniform(wid_min, wid_max)

            rect = box(-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)

            angle = rng.choice(angle_candidates)
            rect_rot = rotate(rect, angle, origin=(0, 0), use_radians=False)

            cx = rng.uniform(minx, maxx)
            cy = rng.uniform(miny, maxy)

            rect_shifted = translate(rect_rot, xoff=cx, yoff=cy)

            if not rect_shifted.within(poly):
                continue

            if (
                existing_centroids
                and min_dist_between_centers
                and min_dist_between_centers > 0.0
            ):
                center = rect_shifted.centroid
                too_close = any(
                    center.distance(c) < min_dist_between_centers
                    for c in existing_centroids
                )
                if too_close:
                    continue

            return rect_shifted

        return None

    @staticmethod
    def _normalize_building_geometry(geom: BaseGeometry) -> BaseGeometry:
        c = geom.centroid
        return translate(geom, xoff=-c.x, yoff=-c.y)

    def _place_building_in_plot(
        self,
        building_template: BaseGeometry,
        plot_geom: BaseGeometry,
        rng: Optional[random.Random] = None,
    ) -> Optional[BaseGeometry]:
        rng = rng or random.Random()
        allowed_area = plot_geom.buffer(-self.generation_parameters.INNER_BORDER)
        if allowed_area.is_empty:
            return None

        minx, miny, maxx, maxy = allowed_area.bounds
        if minx == maxx or miny == maxy:
            return None

        for _ in range(self.generation_parameters.max_service_attempts):
            cx = rng.uniform(minx, maxx)
            cy = rng.uniform(miny, maxy)

            if not allowed_area.contains(box(cx, cy, cx, cy)):
                continue

            b_shifted = translate(building_template, xoff=cx, yoff=cy)

            if b_shifted.within(allowed_area):
                return b_shifted

        return None

    def _select_project_for_remaining(
        self,
        service_projects: gpd.GeoDataFrame,
        remaining_capacity: float,
    ) -> List[pd.Series]:
        if service_projects.empty:
            return []

        df = service_projects.copy()
        df["capacity_diff"] = (df["capacity"] - remaining_capacity).abs()
        df_sorted = df.sort_values("capacity_diff")

        return list(df_sorted.itertuples(index=False))


    def _add_coverage_services(
        self,
        homes_gdf: gpd.GeoDataFrame,
        block_state: Dict[Any, Dict[str, Any]],
        all_limits: Dict[Hashable, Dict[str, float]],
        projects_local: gpd.GeoDataFrame,
        normalized_buildings: Dict[Any, BaseGeometry],
        rows: List[Dict[str, Any]],
        unserved: List[Dict[str, Any]],
        rng: random.Random,
        max_added_per_service: int = 12,
    ) -> int:
        """Добавляет сервисы там, где жилые дома дальше нормативной доступности от всех сервисов своего вида.

        Радиусы: service_access.SERVICE_ACCESS_M (СП 42, прил. Л, табл. Л.1; по прямой, от стены до стены).
        Вместимость нового здания не меньше потребности жителей домов, которые оно покрывает
        (норматив на 1000 жителей из all_limits); решение автора: превышение общей нормы допускается.
        Если здание не помещается, дома остаются без покрытия и попадают в `unserved`.
        """
        from app.logic.service_access import SERVICE_ACCESS_M

        if homes_gdf is None or len(homes_gdf) == 0 or "living_area" not in homes_gdf.columns:
            return 0
        lap = float(self.generation_parameters.la_per_person)
        homes = []
        for r in homes_gdf.itertuples():
            la = float(getattr(r, "living_area", 0.0) or 0.0)
            bid = getattr(r, "src_index", None)
            if la <= 0.0 or bid not in block_state:
                continue
            homes.append({"geom": r.geometry, "people": la / lap, "block": bid, "zone": block_state[bid]["zone"]})
        added = 0
        for zone in {h["zone"] for h in homes}:
            zh = [h for h in homes if h["zone"] == zone]
            people_zone = sum(h["people"] for h in zh)
            if people_zone <= 0:
                continue
            for service, limit in (all_limits.get(zone) or {}).items():
                radius = SERVICE_ACCESS_M.get(str(service))
                projs = projects_local[projects_local["service"] == service]
                if radius is None or projs.empty or float(limit) <= 0:
                    continue
                per_person = float(limit) / people_zone
                caps = sorted(projs["capacity"].astype(float).unique())
                min_plot = float((projs["plot_length_min"] * projs["plot_width_min"]).min())
                reach = radius - 1.0

                def svc_union():
                    g = [r["geometry"] for r in rows if r["service"] == service]
                    return unary_union(g) if g else None

                def uncovered_homes(skip):
                    su = svc_union()
                    return [i for i, h in enumerate(zh) if i not in skip and (su is None or h["geom"].distance(su) > radius)]

                def free_part(allowed):
                    return {b: st["free"].intersection(allowed) for b, st in block_state.items()
                            if st.get("zone") == zone and "free" in st and not st["free"].is_empty}

                skip: set = set()
                for _ in range(max_added_per_service):
                    unc = uncovered_homes(skip)
                    if not unc:
                        break
                    su = svc_union()
                    seed = max(unc, key=lambda i: zh[i]["geom"].distance(su) if su is not None else zh[i]["people"])
                    order = sorted((i for i in unc if i != seed), key=lambda i: zh[i]["geom"].distance(zh[seed]["geom"]))
                    cluster, allowed = [seed], zh[seed]["geom"].buffer(reach)
                    for i in order:
                        cand = allowed.intersection(zh[i]["geom"].buffer(reach))
                        demand = sum(zh[j]["people"] for j in cluster + [i]) * per_person
                        if demand > caps[-1] or cand.is_empty:
                            continue
                        if any(fp.area >= min_plot for fp in free_part(cand).values()):
                            cluster.append(i)
                            allowed = cand
                    placed = False
                    for attempt in (cluster, [seed]):
                        allowed = zh[attempt[0]]["geom"].buffer(reach)
                        for i in attempt[1:]:
                            allowed = allowed.intersection(zh[i]["geom"].buffer(reach))
                        demand = sum(zh[j]["people"] for j in attempt) * per_person
                        need = [c for c in caps if c >= demand] or [caps[-1]]
                        for proj in projs.sort_values("capacity").itertuples(index=False):
                            if float(proj.capacity) < need[0]:
                                continue
                            tmpl = normalized_buildings.get(proj.type_id)
                            if tmpl is None:
                                continue
                            for bid, part in free_part(allowed).items():
                                if part.is_empty:
                                    continue
                                ang = block_state[bid]["angle"]
                                plot = self._sample_rect_in_polygon(
                                    part, (float(proj.plot_length_min), float(proj.plot_length_max)),
                                    (float(proj.plot_width_min), float(proj.plot_width_max)),
                                    max_attempts=self.generation_parameters.max_service_attempts,
                                    preferred_angle=ang, rng=rng)
                                if plot is None:
                                    continue
                                geom = None
                                for a in (ang, ang + 90.0):
                                    geom = self._place_building_in_plot(rotate(tmpl, a, origin=(0, 0), use_radians=False), plot, rng=rng)
                                    if geom is not None:
                                        break
                                if geom is None:
                                    continue
                                try:
                                    fl = float(proj.floors_count)
                                except (TypeError, ValueError):
                                    fl = 0.0
                                rows.append({
                                    "src_index": bid, "zone": zone, "service": service, "project_id": proj.type_id,
                                    "capacity": float(proj.capacity), "floors_count": proj.floors_count,
                                    "living_area": 0.0, "functional_area": geom.area * fl if fl > 0 else 0.0,
                                    "osm_url": getattr(proj, "osm_url", None), "address": getattr(proj, "address", None),
                                    "geometry": geom,
                                })
                                block_state[bid]["free"] = block_state[bid]["free"].difference(plot)
                                added += 1
                                placed = True
                                break
                            if placed:
                                break
                        if placed:
                            break
                    if not placed or seed in uncovered_homes(skip):
                        skip.add(seed)
                        unserved.append({"zone": zone, "service": str(service), "home_geom": zh[seed]["geom"].wkt,
                                         "reason": "no_space_within_access_radius"})
        return added

    def place_service_buildings(
        self,
        blocks: gpd.GeoDataFrame,
        occupied_buildings_gdf: gpd.GeoDataFrame,
        all_limits: Dict[Hashable, Dict[str, float]],
        projects_gdf: gpd.GeoDataFrame,
        blocks_crs: int | str = 32636,
        rng: Optional[random.Random] = None,
    ) -> gpd.GeoDataFrame:
        rng = rng or random.Random(self.generation_parameters.seed)
        projects_local = ensure_crs(projects_gdf, blocks_crs)

        normalized_buildings: Dict[Any, BaseGeometry] = {}
        for row in projects_local.itertuples():
            type_id = getattr(row, "type_id")
            geom = getattr(row, "geometry")
            normalized_buildings[type_id] = self._normalize_building_geometry(geom)

        service_buildings_rows: List[Dict[str, Any]] = []

        zone_placed_capacity: Dict[Hashable, Dict[str, float]] = {}
        site_limit_targets: set[tuple[Hashable, str]] = set()
        block_state: Dict[Any, Dict[str, Any]] = {}

        for block_row in blocks.itertuples():
            block_id = getattr(block_row, "src_index")
            block_geom = getattr(block_row, "geometry")
            zone_id = getattr(block_row, "zone", None)

            if zone_id is None:
                continue

            block_angle = self._compute_main_axis_angle(block_geom)

            minx_b, miny_b, maxx_b, maxy_b = block_geom.bounds
            span_min = min(maxx_b - minx_b, maxy_b - miny_b)

            min_spacing = 0.0
            if (
                self.generation_parameters.max_sites_per_service_per_block > 1
                and span_min > 0
            ):
                min_spacing = 0.25 * span_min

            placed_plot_centers: List[Point] = []

            occupied_block = occupied_buildings_gdf[
                occupied_buildings_gdf["src_index"] == block_id
            ]

            # Residential plots normally cover nearly all buildable land.  They
            # are cadastral/algorithmic allocations, not occupied geometry; if
            # they are subtracted here, service placement sees no usable area.
            # Only generated building footprints are physical obstacles.
            free_area = self._get_block_free_area(block_geom, occupied_block)
            block_state[block_id] = {"angle": block_angle, "zone": zone_id}
            block_state[block_id]["free"] = free_area
            if free_area.is_empty:
                continue

            block_limits = all_limits.get(zone_id, {})
            if not block_limits:
                continue

            if zone_id not in zone_placed_capacity:
                zone_placed_capacity[zone_id] = {srv: 0.0 for srv in block_limits}

            zone_caps = zone_placed_capacity[zone_id]

            for service_name, target_capacity in block_limits.items():
                if target_capacity <= 0:
                    continue

                placed_so_far = float(zone_caps.get(service_name, 0.0))
                if placed_so_far >= target_capacity:
                    continue

                service_projects = projects_local[
                    projects_local["service"] == service_name
                ]
                if service_projects.empty:
                    continue

                placed_capacity = placed_so_far
                sites_in_block = 0

                while (
                    placed_capacity < target_capacity
                    and (
                        self.generation_parameters.max_sites_per_service_per_block <= 0
                        or sites_in_block
                        < self.generation_parameters.max_sites_per_service_per_block
                    )
                    and not free_area.is_empty
                ):
                    remaining_capacity = target_capacity - placed_capacity

                    project_candidates = self._select_project_for_remaining(
                        service_projects,
                        remaining_capacity,
                    )
                    if not project_candidates:
                        break

                    placed_in_iteration = False

                    for project_row in project_candidates:
                        type_id = getattr(project_row, "type_id")
                        capacity = float(getattr(project_row, "capacity"))
                        floors = getattr(project_row, "floors_count")
                        plot_length_min = float(getattr(project_row, "plot_length_min"))
                        plot_length_max = float(getattr(project_row, "plot_length_max"))
                        plot_width_min = float(getattr(project_row, "plot_width_min"))
                        plot_width_max = float(getattr(project_row, "plot_width_max"))
                        osm_url = getattr(project_row, "osm_url", None)
                        address = getattr(project_row, "address", None)

                        building_template_norm = normalized_buildings.get(type_id)
                        if building_template_norm is None:
                            continue

                        plot_geom = self._sample_rect_in_polygon(
                            free_area,
                            length_range=(plot_length_min, plot_length_max),
                            width_range=(plot_width_min, plot_width_max),
                            max_attempts=self.generation_parameters.max_service_attempts,
                            preferred_angle=block_angle,
                            existing_centroids=placed_plot_centers,
                            min_dist_between_centers=min_spacing,
                            rng=rng,
                        )

                        if plot_geom is None:
                            continue

                        building_oriented_main = rotate(
                            building_template_norm,
                            block_angle,
                            origin=(0, 0),
                            use_radians=False,
                        )

                        building_geom = self._place_building_in_plot(
                            building_template=building_oriented_main,
                            plot_geom=plot_geom,
                            rng=rng,
                        )

                        if building_geom is None:
                            building_oriented_orth = rotate(
                                building_template_norm,
                                block_angle + 90.0,
                                origin=(0, 0),
                                use_radians=False,
                            )
                            building_geom = self._place_building_in_plot(
                                building_template=building_oriented_orth,
                                plot_geom=plot_geom,
                                rng=rng,
                            )

                        if building_geom is None:
                            continue

                        footprint_area = building_geom.area
                        try:
                            floors_val = float(floors) if floors is not None else 0.0
                        except (TypeError, ValueError):
                            floors_val = 0.0

                        building_area = (
                            footprint_area * floors_val if floors_val > 0 else 0.0
                        )

                        row_out: Dict[str, Any] = {
                            "src_index": block_id,
                            "zone": zone_id,
                            "service": service_name,
                            "project_id": type_id,
                            "capacity": capacity,
                            "floors_count": floors,
                            "living_area": 0.0,
                            "functional_area": building_area,
                            "osm_url": osm_url,
                            "address": address,
                            "geometry": building_geom,
                        }
                        service_buildings_rows.append(row_out)

                        placed_capacity += capacity
                        sites_in_block += 1
                        placed_in_iteration = True

                        free_area = free_area.difference(plot_geom)
                        block_state[block_id]["free"] = free_area
                        placed_plot_centers.append(plot_geom.centroid)

                        break

                    if not placed_in_iteration:
                        break

                zone_caps[service_name] = placed_capacity
                if (
                    placed_capacity < target_capacity
                    and self.generation_parameters.max_sites_per_service_per_block > 0
                    and sites_in_block
                    >= self.generation_parameters.max_sites_per_service_per_block
                ):
                    site_limit_targets.add((zone_id, str(service_name)))

        access_unserved: List[Dict[str, Any]] = []
        coverage_added = 0
        try:
            coverage_added = self._add_coverage_services(
                occupied_buildings_gdf, block_state, all_limits, projects_local,
                normalized_buildings, service_buildings_rows, access_unserved, rng,
            )
        except Exception as exc:  # расширение покрытия не должно ронять генерацию
            logger.warning("service coverage step failed: {}", exc)

        available_service_names = set(projects_local["service"].dropna().astype(str))
        failure_reasons: Dict[tuple[Hashable, str], str] = {}
        for zone_id, block_limits in all_limits.items():
            placed_for_zone = zone_placed_capacity.get(zone_id, {})
            for service_name, target_capacity in block_limits.items():
                if float(target_capacity) <= 0.0:
                    continue
                service_key = (zone_id, str(service_name))
                if float(placed_for_zone.get(service_name, 0.0)) >= float(
                    target_capacity
                ):
                    continue
                if str(service_name) not in available_service_names:
                    failure_reasons[service_key] = "no_template"
                elif service_key in site_limit_targets:
                    failure_reasons[service_key] = "site_limit"
                else:
                    failure_reasons[service_key] = "no_space"

        if not service_buildings_rows:
            result = gpd.GeoDataFrame(
                columns=[
                    "service",
                    "capacity",
                    "floors_count",
                    "living_area",
                    "functional_area",
                    "geometry",
                ],
                geometry="geometry",
                crs=blocks_crs,
            )
        else:
            result = gpd.GeoDataFrame(
                service_buildings_rows,
                geometry="geometry",
                crs=blocks_crs,
            )
        result.attrs["failure_reasons"] = failure_reasons
        result.attrs["coverage_services_added"] = coverage_added
        result.attrs["access_unserved"] = access_unserved
        return result

    async def generate_services(
        self,
        blocks: gpd.GeoDataFrame,
        plots: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        service_normatives: pd.DataFrame,
        crs: int | str,
    ) -> gpd.GeoDataFrame:
        
        blocks = blocks.copy()

        if "zone" not in blocks.columns:
            raise ValueError("blocks must contain 'zone' column for service generation")
        blocks = blocks[blocks["zone"] == "residential"].copy()
        if blocks.empty:
            empty = gpd.GeoDataFrame(
                columns=[
                    "service",
                    "capacity",
                    "floors_count",
                    "living_area",
                    "functional_area",
                    "geometry",
                ],
                geometry="geometry",
                crs=crs,
            )
            empty.attrs["service_diagnostics"] = self.summarize_service_generation(
                {}, empty
            )
            return empty

        blocks = blocks.reset_index()
        blocks.rename(columns={"index": "src_index"}, inplace=True)

        projects_gdf = await asyncio.to_thread(self.load_service_projects)
        all_limits = await asyncio.to_thread(
            self.compute_service_limits_for_blocks,
            blocks,
            buildings,
            service_normatives,
        )

        services_buildings_gdf = await asyncio.to_thread(
            self.place_service_buildings,
            blocks,
            buildings,
            all_limits,
            projects_gdf,
            crs,
        )
        services_buildings_gdf.attrs["service_diagnostics"] = (
            self.summarize_service_generation(
                all_limits,
                services_buildings_gdf,
                services_buildings_gdf.attrs.get("failure_reasons"),
            )
        )
        services_buildings_gdf.attrs["service_diagnostics"]["coverage_services_added"] = int(
            services_buildings_gdf.attrs.get("coverage_services_added", 0)
        )
        services_buildings_gdf.attrs["service_diagnostics"]["access_unserved"] = list(
            services_buildings_gdf.attrs.get("access_unserved", [])
        )
        return services_buildings_gdf

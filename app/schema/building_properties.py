"""Display metadata for the properties of generated building features.

Generation endpoints return machine-readable property names and enum values.
This module maps them to Russian labels, units and value dictionaries so
clients render them without keeping their own copy that silently drifts from
the API. Served by ``GET /generate/properties_schema``.
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

PropertyKind = Literal["number", "integer", "boolean", "enum", "string", "array"]


class PropertyLabel(BaseModel):
    label: str = Field(..., description="Human-readable property name (Russian)")
    kind: PropertyKind = Field(..., description="How the value should be rendered")
    unit: Optional[str] = Field(default=None, description="Measurement unit, if any")
    excluded_only: bool = Field(
        default=False,
        description=(
            "Property is present only on excluded existing objects. "
            "Generated buildings omit the key entirely rather than sending a "
            "falsy value."
        ),
    )
    landscape_only: bool = Field(
        default=False,
        description="Property is present only when `landscape_enabled` is set in the generation parameters.",
    )


class BuildingPropertiesSchema(BaseModel):
    """Labels for property names and for the values of enum properties."""

    properties: Dict[str, PropertyLabel]
    values: Dict[str, Dict[str, str]]


_PROPERTIES: Dict[str, PropertyLabel] = {
    "floors_count": PropertyLabel(label="Количество этажей", kind="number", unit="эт."),
    "living_area": PropertyLabel(label="Жилая площадь", kind="number", unit="м²"),
    "building_area": PropertyLabel(label="Общая площадь здания", kind="number", unit="м²"),
    "residents_number": PropertyLabel(label="Число жителей", kind="number", unit="чел."),
    "building_type": PropertyLabel(label="Тип застройки", kind="enum"),
    "zone": PropertyLabel(label="Функциональная зона", kind="enum"),
    "service": PropertyLabel(label="Сервис", kind="string"),
    "capacity": PropertyLabel(label="Мощность сервиса", kind="number", unit="мест"),
    "broke_restriction_zone": PropertyLabel(
        label="Нарушение нормативных отступов", kind="boolean"
    ),
    "parking_class": PropertyLabel(label="Вид парковки дома", kind="enum", landscape_only=True),
    "has_underground_parking": PropertyLabel(label="Есть подземная парковка", kind="boolean", landscape_only=True),
    "underground_parking_places": PropertyLabel(
        label="Подземных машино-мест", kind="integer", unit="мест", landscape_only=True
    ),
    "surface_parking_places": PropertyLabel(
        label="Наземных постоянных машино-мест", kind="integer", unit="мест", landscape_only=True
    ),
    "guest_parking_places": PropertyLabel(
        label="Гостевых машино-мест", kind="integer", unit="мест", landscape_only=True
    ),
    "is_excluded": PropertyLabel(
        label="Существующий объект", kind="boolean", excluded_only=True
    ),
    "physical_object_id": PropertyLabel(
        label="ID физического объекта", kind="integer", excluded_only=True
    ),
}

_BUILDING_TYPE_VALUES: Dict[str, str] = {
    "private": "ИЖС",
    "low": "Малоэтажная",
    "medium": "Среднеэтажная",
    "high": "Многоэтажная",
    "extreme": "Высотная",
    "business_low": "Общественно-деловая малоэтажная",
    "business_mid": "Общественно-деловая среднеэтажная",
    "business_tower": "Деловая башня",
    "business_mall": "Торговый центр",
    "industrial_light": "Лёгкая промышленность",
    "industrial_warehouse": "Склад",
    "industrial_heavy": "Тяжёлая промышленность",
    "transport_station": "Транспортная станция",
    "transport_depot": "Депо",
    "transport_parking": "Парковка",
    "special_technical": "Технический объект",
    "special_waste": "Объект обращения с отходами",
}

_ZONE_VALUES: Dict[str, str] = {
    "residential": "Жилая",
    "business": "Общественно-деловая",
    "industrial": "Промышленная",
    "transport": "Транспортная",
    "special": "Специального назначения",
    "recreation": "Рекреационная",
    "agriculture": "Сельскохозяйственная",
    "unknown": "Не определена",
}

_PARKING_CLASS_VALUES: Dict[str, str] = {
    "underground": "Только подземная",
    "mixed": "Подземная и наземная",
    "surface": "Только наземная",
    "none": "Нет мест",
}

BUILDING_PROPERTIES_SCHEMA = BuildingPropertiesSchema(
    properties=_PROPERTIES,
    values={
        "building_type": _BUILDING_TYPE_VALUES,
        "zone": _ZONE_VALUES,
        "parking_class": _PARKING_CLASS_VALUES,
    },
)

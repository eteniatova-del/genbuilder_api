"""Ответ API содержит сигнал о том, что запрошенное число жителей не помещается."""
from app.schema.dto import BuildingFeatureCollection  # type: ignore


def test_capacity_diagnostics_is_a_documented_response_field():
    assert "capacity_diagnostics" in BuildingFeatureCollection.model_fields

from typing import Any, List

from datasets_preview_backend.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
    JsonColumn,
    check_feature_type,
)


def check_value(value: Any) -> None:
    if value is not None and type(value) != int:
        raise CellTypeError("class label values must be integers")


def check_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class ClassLabelColumn(Column):
    labels: List[str]

    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature is None:
            # we cannot infer from the values in that case (would be inferred as INT instead)
            raise ColumnTypeError("not a class label")
        try:
            check_feature_type(feature, "ClassLabel", [])
            self.labels = [str(name) for name in feature["names"]]
        except Exception:
            raise ColumnTypeError("feature type mismatch")
        check_values(values)
        self.name = name
        self.type = ColumnType.CLASS_LABEL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        return value

    def to_json(self) -> JsonColumn:
        return {"name": self.name, "type": self.type.name, "labels": self.labels}

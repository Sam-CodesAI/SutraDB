"""
Compound metadata filtering engine for SutraDB.
Evaluates JSON predicate ASTs ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains, $exists, $and, $or)
and generates high-speed boolean bitmasks.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class FilterEngine:
    """Evaluates predicate logic against document metadata dictionaries."""

    @staticmethod
    def match_value(field_value: Any, condition: Any) -> bool:
        """
        Evaluates a condition against a single field value.
        Condition can be a direct literal (implicit $eq) or an operator dictionary.
        """
        if not isinstance(condition, dict):
            # Implicit $eq comparison
            return field_value == condition

        # If dictionary has no operator keys ($...), treat as literal object comparison
        if not any(isinstance(k, str) and k.startswith("$") for k in condition):
            return field_value == condition

        # Process explicit operator dictionary
        for op, target in condition.items():
            if op == "$eq":
                if field_value != target:
                    return False
            elif op == "$ne":
                if field_value == target:
                    return False
            elif op == "$gt":
                if field_value is None or not (field_value > target):
                    return False
            elif op == "$gte":
                if field_value is None or not (field_value >= target):
                    return False
            elif op == "$lt":
                if field_value is None or not (field_value < target):
                    return False
            elif op == "$lte":
                if field_value is None or not (field_value <= target):
                    return False
            elif op == "$in":
                if not isinstance(target, (list, tuple, set)) or field_value not in target:
                    return False
            elif op == "$nin":
                if isinstance(target, (list, tuple, set)) and field_value in target:
                    return False
            elif op == "$contains":
                if field_value is None:
                    return False
                if isinstance(field_value, (list, tuple, set)):
                    if target not in field_value:
                        return False
                elif isinstance(field_value, str):
                    if str(target) not in field_value:
                        return False
                else:
                    return False
            elif op == "$exists":
                has_value = field_value is not None
                if has_value != bool(target):
                    return False
            else:
                raise ValueError(f"Unknown filter operator: {op}")

        return True

    @classmethod
    def evaluate(cls, metadata: Optional[Dict[str, Any]], filter_spec: Optional[Dict[str, Any]]) -> bool:
        """
        Evaluates a complete filter specification against a document metadata dictionary.
        Supports compound logical operators: $and, $or, $not.
        """
        if not filter_spec:
            return True
        if metadata is None:
            metadata = {}

        for key, value in filter_spec.items():
            if key == "$and":
                if not isinstance(value, list):
                    raise ValueError("$and operator requires a list of conditions")
                if not all(cls.evaluate(metadata, sub_filter) for sub_filter in value):
                    return False

            elif key == "$or":
                if not isinstance(value, list):
                    raise ValueError("$or operator requires a list of conditions")
                if not any(cls.evaluate(metadata, sub_filter) for sub_filter in value):
                    return False

            elif key == "$not":
                if not isinstance(value, dict):
                    raise ValueError("$not operator requires a dictionary condition")
                if cls.evaluate(metadata, value):
                    return False

            else:
                field_val = metadata.get(key)
                if not cls.match_value(field_val, value):
                    return False

        return True

    @classmethod
    def build_mask(cls, metadata_list: List[Dict[str, Any]], filter_spec: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Generates a 1D NumPy boolean mask for a list of document metadata dictionaries.
        True indicates the document passed the filter predicate.
        """
        n = len(metadata_list)
        if not filter_spec or n == 0:
            return np.ones(n, dtype=bool)

        mask = np.empty(n, dtype=bool)
        for i, meta in enumerate(metadata_list):
            mask[i] = cls.evaluate(meta, filter_spec)

        return mask

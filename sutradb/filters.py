"""
Compound metadata filtering engine for SutraDB.
Evaluates JSON predicate ASTs ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains, $all, $size, $regex, $exists, $and, $or)
and generates high-speed boolean bitmasks with nested dot-notation path resolution.
"""

from typing import Any, Dict, List, Optional
import re
import math
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance in meters between two lat/lon coordinates."""
    earth_radius_meters = 6_371_000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    a = math.sin(d_lat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_meters * c


def extract_lat_lon(val: Any) -> Optional[tuple]:
    """Extracts (lat, lon) from list/tuple or dict."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        try:
            return float(val[0]), float(val[1])
        except (ValueError, TypeError):
            return None
    elif isinstance(val, dict):
        lat = val.get("lat") if "lat" in val else val.get("latitude")
        lon = val.get("lon") if "lon" in val else (val.get("lng") if "lng" in val else val.get("longitude"))
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except (ValueError, TypeError):
                return None
    return None


def resolve_nested_path(data: Optional[Dict[str, Any]], path: str) -> Any:
    """Resolves dot-delimited path (e.g. 'user.profile.city' or 'tags.0') in nested dict/list."""
    if not isinstance(data, dict):
        return None
    if path in data:
        return data[path]

    curr: Any = data
    for part in path.split("."):
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        elif isinstance(curr, (list, tuple)) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(curr):
                curr = curr[idx]
            else:
                return None
        else:
            return None
    return curr


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
            elif op == "$all":
                if not isinstance(field_value, (list, tuple, set)) or not isinstance(target, (list, tuple, set)):
                    return False
                if not all(item in field_value for item in target):
                    return False
            elif op == "$size":
                if not isinstance(field_value, (list, tuple, set, str, dict)):
                    return False
                if len(field_value) != target:
                    return False
            elif op == "$regex":
                if field_value is None:
                    return False
                if not re.search(str(target), str(field_value)):
                    return False
            elif op == "$exists":
                has_value = field_value is not None
                if has_value != bool(target):
                    return False
            elif op == "$geo_within":
                point = extract_lat_lon(field_value)
                if point is None or not isinstance(target, dict):
                    return False
                center = extract_lat_lon(target.get("center"))
                radius_meters = target.get("radius_meters")
                if radius_meters is None and "radius_km" in target:
                    radius_meters = float(target["radius_km"]) * 1000.0
                if center is None or radius_meters is None:
                    return False
                dist = haversine_distance(point[0], point[1], center[0], center[1])
                if dist > float(radius_meters):
                    return False
            elif op == "$geo_box":
                point = extract_lat_lon(field_value)
                if point is None:
                    return False
                bounds = None
                if isinstance(target, (list, tuple)) and len(target) >= 4:
                    bounds = (float(target[0]), float(target[1]), float(target[2]), float(target[3]))
                elif isinstance(target, dict):
                    bl = extract_lat_lon(target.get("bottom_left"))
                    tr = extract_lat_lon(target.get("top_right"))
                    if bl and tr:
                        bounds = (min(bl[0], tr[0]), min(bl[1], tr[1]), max(bl[0], tr[0]), max(bl[1], tr[1]))
                    elif all(k in target for k in ("min_lat", "min_lon", "max_lat", "max_lon")):
                        bounds = (float(target["min_lat"]), float(target["min_lon"]), float(target["max_lat"]), float(target["max_lon"]))
                if bounds is None:
                    return False
                min_lat, min_lon, max_lat, max_lon = bounds
                lat, lon = point
                if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                    return False
            else:
                raise ValueError(f"Unknown filter operator: {op}")

        return True

    @classmethod
    def evaluate(cls, metadata: Optional[Dict[str, Any]], filter_spec: Optional[Dict[str, Any]]) -> bool:
        """
        Evaluates a complete filter specification against a document metadata dictionary.
        Supports compound logical operators: $and, $or, $not, and dot-notation paths.
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
                field_val = resolve_nested_path(metadata, key)
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

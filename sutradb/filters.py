"""
Compound metadata filtering engine for SutraDB.
Evaluates JSON predicate ASTs ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains, $all, $size, $regex, $exists, $and, $or)
and generates high-speed boolean bitmasks with nested dot-notation path resolution.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import re
import re._constants as _re_const
import math
import numpy as np

_MAX_REGEX_LENGTH = 512
_MAX_REGEX_DEPTH = 16
_MAX_FINITE_REPEAT = 1000
_REPEAT_OPS = {
    _re_const.MAX_REPEAT,
    _re_const.MIN_REPEAT,
    _re_const.REPEAT,
    _re_const.REPEAT_ONE,
    _re_const.MIN_REPEAT_ONE,
}
_ALL_REPEAT_OPS = _REPEAT_OPS | {
    op for op in (
        getattr(_re_const, "POSSESSIVE_REPEAT", None),
        getattr(_re_const, "POSSESSIVE_REPEAT_ONE", None),
    )
    if op is not None
}


def _node_can_overlap_wildcard(subpattern: Any) -> bool:
    """Returns True if a subpattern contains ANY or non-trivial alternation/broad classes."""
    for op, av in subpattern:
        if op == _re_const.ANY:
            return True
        if op == _re_const.SUBPATTERN:
            if _node_can_overlap_wildcard(av[3]):
                return True
        elif op in _ALL_REPEAT_OPS:
            if _node_can_overlap_wildcard(av[2]):
                return True
        elif op == _re_const.BRANCH:
            return True
        elif op in (_re_const.IN, _re_const.CATEGORY):
            return True
    return False


def _extract_literal_set(subpattern: Any) -> Optional[set]:
    """Returns exact set of literal chars if subpattern is simple literals/IN-literals, else None (broad)."""
    chars: set = set()
    for op, av in subpattern:
        if op == _re_const.LITERAL:
            chars.add(av)
        elif op == _re_const.IN:
            for item_op, item_av in av:
                if item_op == _re_const.LITERAL:
                    chars.add(item_av)
                else:
                    return None
        elif op == _re_const.SUBPATTERN:
            sub = _extract_literal_set(av[3])
            if sub is None:
                return None
            chars.update(sub)
        else:
            return None
    return chars


def _repeats_can_overlap(av1: Any, av2: Any) -> bool:
    """Checks if two adjacent unbounded repeats could share matching characters."""
    s1 = _extract_literal_set(av1[2])
    s2 = _extract_literal_set(av2[2])
    if s1 is not None and s2 is not None:
        return bool(s1 & s2)
    # Check disjoint category pairs (e.g., \d+ followed by \s+)
    def _cat_only(subpattern: Any) -> Optional[Any]:
        items = list(subpattern)
        if len(items) == 1 and items[0][0] == _re_const.CATEGORY:
            return items[0][1]
        if len(items) == 1 and items[0][0] == _re_const.IN and len(items[0][1]) == 1 and items[0][1][0][0] == _re_const.CATEGORY:
            return items[0][1][0][1]
        return None
    c1 = _cat_only(av1[2])
    c2 = _cat_only(av2[2])
    if c1 is not None and c2 is not None and c1 != c2:
        disjoint_pairs = {
            frozenset((_re_const.CATEGORY_DIGIT, _re_const.CATEGORY_SPACE)),
            frozenset((_re_const.CATEGORY_WORD, _re_const.CATEGORY_SPACE)),
            frozenset((_re_const.CATEGORY_DIGIT, _re_const.CATEGORY_NOT_DIGIT)),
            frozenset((_re_const.CATEGORY_WORD, _re_const.CATEGORY_NOT_WORD)),
            frozenset((_re_const.CATEGORY_SPACE, _re_const.CATEGORY_NOT_SPACE)),
        }
        if frozenset((c1, c2)) in disjoint_pairs:
            return False
    return True


def _validate_regex_ast(subpattern: Any, in_repeat: bool = False, depth: int = 0) -> None:
    """Validates parsed regex AST against catastrophic backtracking (ReDoS) patterns."""
    if depth > _MAX_REGEX_DEPTH:
        raise ValueError("Unsafe regex pattern: nesting depth exceeds maximum allowed")

    prev_unbounded_repeat: Optional[Any] = None
    for op, av in subpattern:
        is_curr_unbounded = False
        if op in _ALL_REPEAT_OPS:
            min_count, max_count, body = av
            is_unbounded = max_count == _re_const.MAXREPEAT
            is_multi = is_unbounded or max_count > 1

            if not is_unbounded and max_count > _MAX_FINITE_REPEAT:
                raise ValueError("Unsafe regex pattern: repetition count exceeds maximum allowed")

            if in_repeat and is_multi:
                raise ValueError("Unsafe regex pattern: nested quantifiers (potential ReDoS)")

            if is_unbounded and prev_unbounded_repeat is not None and op in _REPEAT_OPS:
                if _repeats_can_overlap(prev_unbounded_repeat, av):
                    raise ValueError("Unsafe regex pattern: adjacent overlapping unbounded quantifiers")

            _validate_regex_ast(body, in_repeat=(in_repeat or is_multi), depth=depth + 1)
            is_curr_unbounded = is_unbounded and op in _REPEAT_OPS
        elif op == _re_const.SUBPATTERN:
            _validate_regex_ast(av[3], in_repeat=in_repeat, depth=depth + 1)
        elif op == _re_const.BRANCH:
            if in_repeat:
                raise ValueError("Unsafe regex pattern: alternation inside quantifier (potential ReDoS)")
            for branch in av[1]:
                _validate_regex_ast(branch, in_repeat=in_repeat, depth=depth + 1)
        elif op in (
            _re_const.GROUPREF,
            _re_const.GROUPREF_IGNORE,
            _re_const.GROUPREF_LOC_IGNORE,
            _re_const.GROUPREF_UNI_IGNORE,
            _re_const.ASSERT,
            _re_const.ASSERT_NOT,
        ):
            if in_repeat:
                raise ValueError("Unsafe regex pattern: backtracking construct inside quantifier")
            if op in (_re_const.ASSERT, _re_const.ASSERT_NOT):
                _validate_regex_ast(av[1], in_repeat=in_repeat, depth=depth + 1)
        elif op == _re_const.GROUPREF_EXISTS:
            if in_repeat:
                raise ValueError("Unsafe regex pattern: conditional backreference inside quantifier")
            _validate_regex_ast(av[1], in_repeat=in_repeat, depth=depth + 1)
            if av[2] is not None:
                _validate_regex_ast(av[2], in_repeat=in_repeat, depth=depth + 1)

        prev_unbounded_repeat = av if is_curr_unbounded else None


@lru_cache(maxsize=256)
def _compile_safe_regex(pattern: str) -> Optional[re.Pattern]:
    """Compiles a regex after ReDoS AST validation. Invalid syntax returns None; unsafe AST raises ValueError."""
    if len(pattern) > _MAX_REGEX_LENGTH:
        raise ValueError(f"Unsafe regex pattern: length exceeds {_MAX_REGEX_LENGTH} characters")

    try:
        parsed = re._parser.parse(pattern)
    except re.error:
        return None

    _validate_regex_ast(parsed)

    try:
        return re.compile(pattern)
    except re.error:
        return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance in meters between two lat/lon coordinates."""
    earth_radius_meters = 6_371_000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    a = math.sin(d_lat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_meters * c


def extract_lat_lon(val: Any) -> Optional[Tuple[float, float]]:
    """Extracts (lat, lon) from list/tuple or dict, ensuring finite coordinates."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        if isinstance(val[0], bool) or isinstance(val[1], bool):
            return None
        try:
            lat_f, lon_f = float(val[0]), float(val[1])
            if math.isfinite(lat_f) and math.isfinite(lon_f):
                return lat_f, lon_f
        except (ValueError, TypeError):
            return None
    elif isinstance(val, dict):
        lat = val.get("lat") if "lat" in val else val.get("latitude")
        lon = val.get("lon") if "lon" in val else (val.get("lng") if "lng" in val else val.get("longitude"))
        if lat is not None and lon is not None and not isinstance(lat, bool) and not isinstance(lon, bool):
            try:
                lat_f, lon_f = float(lat), float(lon)
                if math.isfinite(lat_f) and math.isfinite(lon_f):
                    return lat_f, lon_f
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


def _values_equal_or_element(field_value: Any, target: Any) -> bool:
    """Matches Rust FilterEngine semantics: scalar equality, list equality, or element-in-list membership."""
    if target is None:
        return field_value is None
    if isinstance(field_value, list):
        if isinstance(target, list):
            return field_value == target
        return any(item == target for item in field_value)
    return field_value == target


class FilterEngine:
    """Evaluates predicate logic against document metadata dictionaries."""

    @staticmethod
    def match_value(field_value: Any, condition: Any) -> bool:
        """
        Evaluates a condition against a single field value.
        Condition can be a direct literal (implicit $eq) or an operator dictionary.
        """
        if not isinstance(condition, dict):
            # Implicit $eq comparison (supports array element membership matching Rust)
            return _values_equal_or_element(field_value, condition)

        # If dictionary has no operator keys ($...), treat as literal object comparison
        if not any(isinstance(k, str) and k.startswith("$") for k in condition):
            return _values_equal_or_element(field_value, condition)

        # Process explicit operator dictionary
        for op, target in condition.items():
            if op == "$eq":
                if not _values_equal_or_element(field_value, target):
                    return False
            elif op == "$ne":
                if _values_equal_or_element(field_value, target):
                    return False
            elif op == "$gt":
                if field_value is None:
                    return False
                try:
                    if not (field_value > target):
                        return False
                except TypeError:
                    return False
            elif op == "$gte":
                if field_value is None:
                    return False
                try:
                    if not (field_value >= target):
                        return False
                except TypeError:
                    return False
            elif op == "$lt":
                if field_value is None:
                    return False
                try:
                    if not (field_value < target):
                        return False
                except TypeError:
                    return False
            elif op == "$lte":
                if field_value is None:
                    return False
                try:
                    if not (field_value <= target):
                        return False
                except TypeError:
                    return False
            elif op == "$in":
                if not isinstance(target, (list, tuple, set)):
                    return False
                if field_value is None:
                    return False
                try:
                    if isinstance(field_value, list):
                        if (field_value not in target) and not any(item in target for item in field_value):
                            return False
                    elif field_value not in target:
                        return False
                except TypeError:
                    return False
            elif op == "$nin":
                if isinstance(target, (list, tuple, set)) and field_value is not None:
                    try:
                        if isinstance(field_value, list):
                            if (field_value in target) or any(item in target for item in field_value):
                                return False
                        elif field_value in target:
                            return False
                    except TypeError:
                        pass
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
                if isinstance(target, bool) or not isinstance(target, int):
                    return False
                if not isinstance(field_value, (list, tuple, set, str, dict)):
                    return False
                if len(field_value) != target:
                    return False
            elif op == "$regex":
                if field_value is None or target is None:
                    return False
                compiled = _compile_safe_regex(str(target))
                if compiled is None:
                    return False
                if not compiled.search(str(field_value)):
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
                    raw_km = target["radius_km"]
                    if isinstance(raw_km, bool):
                        return False
                    try:
                        radius_meters = float(raw_km) * 1000.0
                    except (ValueError, TypeError):
                        return False
                if center is None or radius_meters is None or isinstance(radius_meters, bool):
                    return False
                try:
                    rad_f = float(radius_meters)
                except (ValueError, TypeError):
                    return False
                if not math.isfinite(rad_f) or rad_f < 0.0:
                    return False
                dist = haversine_distance(point[0], point[1], center[0], center[1])
                if dist > rad_f:
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

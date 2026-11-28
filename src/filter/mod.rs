use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use regex::{Regex, RegexBuilder};
use serde_json::Value;

static REGEX_CACHE: OnceLock<RwLock<HashMap<String, Option<Regex>>>> = OnceLock::new();

fn get_cached_regex(pattern: &str) -> Option<Regex> {
    let lock = REGEX_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Ok(cache) = lock.read() {
        if let Some(entry) = cache.get(pattern) {
            return entry.clone();
        }
    }
    let compiled = RegexBuilder::new(pattern)
        .size_limit(1 << 20)
        .dfa_size_limit(1 << 20)
        .build()
        .ok();
    if let Ok(mut cache) = lock.write() {
        if cache.len() >= 1024 {
            cache.clear();
        }
        cache.insert(pattern.to_string(), compiled.clone());
    }
    compiled
}

pub mod index;
pub use index::SecondaryMetadataIndex;

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(ba), Value::Bool(bb)) => ba == bb,
        (Value::String(sa), Value::String(sb)) => sa == sb,
        (Value::Number(na), Value::Number(nb)) => {
            if na == nb {
                return true;
            }
            match (na.as_f64(), nb.as_f64()) {
                (Some(fa), Some(fb)) => fa == fb,
                _ => false,
            }
        }
        (Value::Array(aa), Value::Array(ab)) => {
            aa.len() == ab.len() && aa.iter().zip(ab.iter()).all(|(x, y)| values_equal(x, y))
        }
        (Value::Object(oa), Value::Object(ob)) => {
            oa.len() == ob.len()
                && oa
                    .iter()
                    .all(|(k, va)| ob.get(k).is_some_and(|vb| values_equal(va, vb)))
        }
        _ => false,
    }
}

/// Evaluates JSON condition AST against metadata values.
pub struct FilterEngine;

impl FilterEngine {
    pub fn match_value(field_value: Option<&Value>, condition: &Value) -> bool {
        match condition {
            Value::Object(map) => {
                // If map has no $-prefixed operator keys, treat as direct literal object comparison
                let has_op = map.keys().any(|k| k.starts_with('$'));
                if !has_op {
                    return field_value.is_some_and(|v| values_equal(v, condition));
                }

                for (op, target) in map {
                    match op.as_str() {
                        "$eq" => {
                            if target.is_null() {
                                if let Some(v) = field_value {
                                    if !v.is_null() {
                                        return false;
                                    }
                                }
                            } else if let Some(Value::Array(arr)) = field_value {
                                if target.is_array() {
                                    if !values_equal(field_value.unwrap(), target) {
                                        return false;
                                    }
                                } else if !arr.iter().any(|item| values_equal(item, target)) {
                                    return false;
                                }
                            } else if !field_value.is_some_and(|v| values_equal(v, target)) {
                                return false;
                            }
                        }
                        "$ne" => {
                            if target.is_null() {
                                match field_value {
                                    None => return false,
                                    Some(v) if v.is_null() => return false,
                                    _ => {}
                                }
                            } else if let Some(Value::Array(arr)) = field_value {
                                if target.is_array() {
                                    if values_equal(field_value.unwrap(), target) {
                                        return false;
                                    }
                                } else if arr.iter().any(|item| values_equal(item, target)) {
                                    return false;
                                }
                            } else if field_value.is_some_and(|v| values_equal(v, target)) {
                                return false;
                            }
                        }
                        "$gt" => {
                            if !compare_values(field_value, target, &|a, b| a > b) {
                                return false;
                            }
                        }
                        "$gte" => {
                            if !compare_values(field_value, target, &|a, b| a >= b) {
                                return false;
                            }
                        }
                        "$lt" => {
                            if !compare_values(field_value, target, &|a, b| a < b) {
                                return false;
                            }
                        }
                        "$lte" => {
                            if !compare_values(field_value, target, &|a, b| a <= b) {
                                return false;
                            }
                        }
                        "$in" => {
                            if let Value::Array(arr) = target {
                                if let Some(val) = field_value {
                                    let matched = match val {
                                        Value::Array(field_arr) => {
                                            arr.iter().any(|t| values_equal(val, t))
                                                || field_arr
                                                    .iter()
                                                    .any(|item| arr.iter().any(|t| values_equal(item, t)))
                                        }
                                        _ => arr.iter().any(|t| values_equal(val, t)),
                                    };
                                    if !matched {
                                        return false;
                                    }
                                } else {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                        "$nin" => {
                            if let Value::Array(arr) = target {
                                if let Some(val) = field_value {
                                    let matched = match val {
                                        Value::Array(field_arr) => {
                                            arr.iter().any(|t| values_equal(val, t))
                                                || field_arr
                                                    .iter()
                                                    .any(|item| arr.iter().any(|t| values_equal(item, t)))
                                        }
                                        _ => arr.iter().any(|t| values_equal(val, t)),
                                    };
                                    if matched {
                                        return false;
                                    }
                                }
                            }
                        }
                        "$contains" => match field_value {
                            Some(Value::Array(arr)) => {
                                if !arr.iter().any(|item| values_equal(item, target)) {
                                    return false;
                                }
                            }
                            Some(Value::String(s)) => {
                                let target_str = match target {
                                    Value::String(ts) => ts.as_str(),
                                    _ => return false,
                                };
                                if !s.contains(target_str) {
                                    return false;
                                }
                            }
                            _ => return false,
                        },
                        "$all" => match (field_value, target) {
                            (Some(Value::Array(field_arr)), Value::Array(target_arr)) => {
                                if !target_arr
                                    .iter()
                                    .all(|t| field_arr.iter().any(|f| values_equal(f, t)))
                                {
                                    return false;
                                }
                            }
                            _ => return false,
                        },
                        "$size" => match (field_value, target) {
                            (Some(Value::Array(field_arr)), Value::Number(num)) => {
                                if let Some(expected_len) = num.as_u64() {
                                    if field_arr.len() != expected_len as usize {
                                        return false;
                                    }
                                } else {
                                    return false;
                                }
                            }
                            (Some(Value::String(s)), Value::Number(num)) => {
                                if let Some(expected_len) = num.as_u64() {
                                    if s.len() != expected_len as usize {
                                        return false;
                                    }
                                } else {
                                    return false;
                                }
                            }
                            _ => return false,
                        },
                        "$exists" => {
                            let exists = field_value.is_some_and(|v| !v.is_null());
                            let expected = target.as_bool().unwrap_or(false);
                            if exists != expected {
                                return false;
                            }
                        }
                        "$regex" => {
                            let pattern = match target.as_str() {
                                Some(p) => p,
                                None => return false,
                            };
                            let re = match get_cached_regex(pattern) {
                                Some(r) => r,
                                None => return false,
                            };
                            let matched = match field_value {
                                Some(Value::String(s)) => re.is_match(s),
                                Some(Value::Array(arr)) => arr
                                    .iter()
                                    .any(|item| matches!(item, Value::String(s) if re.is_match(s))),
                                _ => false,
                            };
                            if !matched {
                                return false;
                            }
                        }
                        "$geo_within" => {
                            let point = match field_value.and_then(extract_lat_lon) {
                                Some(p) => p,
                                None => return false,
                            };

                            let (center, radius_meters) = match target {
                                Value::Object(map) => {
                                    let center = map.get("center").and_then(extract_lat_lon);
                                    let radius = map
                                        .get("radius_meters")
                                        .and_then(|v| v.as_f64())
                                        .or_else(|| {
                                            map.get("radius_km").and_then(|v| v.as_f64().map(|km| km * 1000.0))
                                        });
                                    match (center, radius) {
                                        (Some(c), Some(r)) => (c, r),
                                        _ => return false,
                                    }
                                }
                                _ => return false,
                            };

                            let dist = haversine_distance(point.0, point.1, center.0, center.1);
                            if dist > radius_meters {
                                return false;
                            }
                        }
                        "$geo_box" => {
                            let point = match field_value.and_then(extract_lat_lon) {
                                Some(p) => p,
                                None => return false,
                            };

                            let bounds = match target {
                                Value::Array(arr) if arr.len() >= 4 => {
                                    match (arr[0].as_f64(), arr[1].as_f64(), arr[2].as_f64(), arr[3].as_f64()) {
                                        (Some(min_lat), Some(min_lon), Some(max_lat), Some(max_lon)) => {
                                            Some((min_lat, min_lon, max_lat, max_lon))
                                        }
                                        _ => None,
                                    }
                                }
                                Value::Object(map) => {
                                    if let (Some(bl), Some(tr)) = (
                                        map.get("bottom_left").and_then(extract_lat_lon),
                                        map.get("top_right").and_then(extract_lat_lon),
                                    ) {
                                        let min_lat = bl.0.min(tr.0);
                                        let max_lat = bl.0.max(tr.0);
                                        let min_lon = bl.1.min(tr.1);
                                        let max_lon = bl.1.max(tr.1);
                                        Some((min_lat, min_lon, max_lat, max_lon))
                                    } else if let (Some(min_lat), Some(min_lon), Some(max_lat), Some(max_lon)) = (
                                        map.get("min_lat").and_then(|v| v.as_f64()),
                                        map.get("min_lon").and_then(|v| v.as_f64()),
                                        map.get("max_lat").and_then(|v| v.as_f64()),
                                        map.get("max_lon").and_then(|v| v.as_f64()),
                                    ) {
                                        Some((min_lat, min_lon, max_lat, max_lon))
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            };

                            match bounds {
                                Some((min_lat, min_lon, max_lat, max_lon)) => {
                                    let (lat, lon) = point;
                                    if lat < min_lat || lat > max_lat || lon < min_lon || lon > max_lon {
                                        return false;
                                    }
                                }
                                None => return false,
                            }
                        }
                        _ => return false,
                    }
                }
                true
            }
            Value::Null => match field_value {
                None => true,
                Some(v) => v.is_null(),
            },
            Value::Array(_) => field_value.is_some_and(|v| values_equal(v, condition)),
            _ => match field_value {
                Some(Value::Array(arr)) => arr.iter().any(|item| values_equal(item, condition)),
                Some(v) => values_equal(v, condition),
                None => false,
            },
        }
    }

    pub fn resolve_path<'a>(val: Option<&'a Value>, path: &str) -> Option<&'a Value> {
        let mut curr = val?;
        if let Value::Object(map) = curr {
            if let Some(v) = map.get(path) {
                return Some(v);
            }
        }
        for part in path.split('.') {
            match curr {
                Value::Object(map) => {
                    curr = map.get(part)?;
                }
                Value::Array(arr) => {
                    if let Ok(idx) = part.parse::<usize>() {
                        curr = arr.get(idx)?;
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        Some(curr)
    }

    pub fn evaluate(metadata: Option<&Value>, filter: Option<&Value>) -> bool {
        let filter = match filter {
            Some(f) if !f.is_null() => f,
            _ => return true,
        };

        let filter_obj = match filter.as_object() {
            Some(obj) if !obj.is_empty() => obj,
            _ => return true,
        };

        for (key, val) in filter_obj {
            match key.as_str() {
                "$and" => {
                    if let Some(arr) = val.as_array() {
                        if !arr.iter().all(|sub| Self::evaluate(metadata, Some(sub))) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                "$or" => {
                    if let Some(arr) = val.as_array() {
                        if !arr.iter().any(|sub| Self::evaluate(metadata, Some(sub))) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                "$not" => {
                    if Self::evaluate(metadata, Some(val)) {
                        return false;
                    }
                }
                _ => {
                    let field_val = Self::resolve_path(metadata, key);
                    if !Self::match_value(field_val, val) {
                        return false;
                    }
                }
            }
        }

        true
    }

    pub fn build_mask(metadata_list: &[Value], filter: Option<&Value>) -> Vec<bool> {
        let mut mask = Vec::with_capacity(metadata_list.len());
        for meta in metadata_list {
            mask.push(Self::evaluate(Some(meta), filter));
        }
        mask
    }
}

/// Detects if a filter JSON AST uses complex operators that require full per-document AST inspection
/// (such as $geo_within, $geo_box, $regex, $contains, $all, $size).
pub fn has_complex_operators(filter: &Value) -> bool {
    match filter {
        Value::Object(map) => {
            for (k, v) in map {
                if k == "$geo_within"
                    || k == "$geo_box"
                    || k == "$regex"
                    || k == "$contains"
                    || k == "$all"
                    || k == "$size"
                {
                    return true;
                }
                if has_complex_operators(v) {
                    return true;
                }
            }
            false
        }
        Value::Array(arr) => arr.iter().any(has_complex_operators),
        _ => false,
    }
}

fn compare_values<F>(a: Option<&Value>, b: &Value, op: &F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    match (a, b) {
        (Some(Value::Number(num_a)), Value::Number(num_b)) => {
            if let (Some(fa), Some(fb)) = (num_a.as_f64(), num_b.as_f64()) {
                op(fa, fb)
            } else {
                false
            }
        }
        (Some(Value::Array(arr)), Value::Number(_)) => {
            arr.iter().any(|item| compare_values(Some(item), b, op))
        }
        _ => false,
    }
}

/// Calculates the great-circle Haversine distance in meters between two (lat, lon) coordinates in degrees.
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_METERS: f64 = 6_371_000.0;

    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();

    let a = (d_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_METERS * c
}

/// Extracts (latitude, longitude) from a JSON value.
/// Supports arrays `[lat, lon]` and objects `{"lat": ..., "lon": ...}`.
pub fn extract_lat_lon(val: &Value) -> Option<(f64, f64)> {
    match val {
        Value::Array(arr) if arr.len() >= 2 => {
            let lat = arr[0].as_f64()?;
            let lon = arr[1].as_f64()?;
            Some((lat, lon))
        }
        Value::Object(map) => {
            let lat = map
                .get("lat")
                .or_else(|| map.get("latitude"))
                .and_then(|v| v.as_f64())?;
            let lon = map
                .get("lon")
                .or_else(|| map.get("lng"))
                .or_else(|| map.get("longitude"))
                .and_then(|v| v.as_f64())?;
            Some((lat, lon))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_haversine_distance() {
        // San Francisco (37.7749, -122.4194) to Oakland (37.8044, -122.2712) ~ 13.5 km
        let d = haversine_distance(37.7749, -122.4194, 37.8044, -122.2712);
        assert!((d - 13500.0).abs() < 1000.0);
    }

    #[test]
    fn test_geo_within_filter() {
        let meta = json!({
            "name": "SF Downtown",
            "coords": [37.7749, -122.4194]
        });

        // Filter within 5km of SF center
        let filter_pass = json!({
            "coords": {
                "$geo_within": {
                    "center": [37.7750, -122.4190],
                    "radius_meters": 5000.0
                }
            }
        });
        assert!(FilterEngine::evaluate(Some(&meta), Some(&filter_pass)));

        // Filter within 100m of Oakland (should fail)
        let filter_fail = json!({
            "coords": {
                "$geo_within": {
                    "center": [37.8044, -122.2712],
                    "radius_meters": 100.0
                }
            }
        });
        assert!(!FilterEngine::evaluate(Some(&meta), Some(&filter_fail)));
    }

    #[test]
    fn test_geo_box_filter() {
        let meta = json!({
            "name": "Market St",
            "loc": {"lat": 37.785, "lon": -122.405}
        });

        let box_filter = json!({
            "loc": {
                "$geo_box": {
                    "bottom_left": [37.70, -122.50],
                    "top_right": [37.80, -122.40]
                }
            }
        });
        assert!(FilterEngine::evaluate(Some(&meta), Some(&box_filter)));

        let box_filter_outside = json!({
            "loc": {
                "$geo_box": [38.0, -123.0, 39.0, -122.0]
            }
        });
        assert!(!FilterEngine::evaluate(Some(&meta), Some(&box_filter_outside)));
    }
}

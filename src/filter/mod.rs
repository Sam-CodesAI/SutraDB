use std::collections::HashMap;
use serde_json::Value;

/// Evaluates JSON condition AST against metadata values.
pub struct FilterEngine;

impl FilterEngine {
    pub fn match_value(field_value: Option<&Value>, condition: &Value) -> bool {
        match condition {
            Value::Object(map) => {
                // If map has no $-prefixed operator keys, treat as direct literal object comparison
                let has_op = map.keys().any(|k| k.starts_with('$'));
                if !has_op {
                    return match field_value {
                        Some(v) => v == condition,
                        None => false,
                    };
                }

                for (op, target) in map {
                    match op.as_str() {
                        "$eq" => {
                            if field_value != Some(target) {
                                return false;
                            }
                        }
                        "$ne" => {
                            if field_value == Some(target) {
                                return false;
                            }
                        }
                        "$gt" => {
                            if !compare_values(field_value, target, |a, b| a > b) {
                                return false;
                            }
                        }
                        "$gte" => {
                            if !compare_values(field_value, target, |a, b| a >= b) {
                                return false;
                            }
                        }
                        "$lt" => {
                            if !compare_values(field_value, target, |a, b| a < b) {
                                return false;
                            }
                        }
                        "$lte" => {
                            if !compare_values(field_value, target, |a, b| a <= b) {
                                return false;
                            }
                        }
                        "$in" => {
                            if let Value::Array(arr) = target {
                                if let Some(val) = field_value {
                                    if !arr.contains(val) {
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
                                    if arr.contains(val) {
                                        return false;
                                    }
                                }
                            }
                        }
                        "$contains" => match field_value {
                            Some(Value::Array(arr)) => {
                                if !arr.contains(target) {
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
                        "$exists" => {
                            let exists = field_value.is_some() && !field_value.unwrap().is_null();
                            let expected = target.as_bool().unwrap_or(false);
                            if exists != expected {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
                true
            }
            _ => field_value == Some(condition),
        }
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
                    let field_val = metadata.and_then(|m| m.get(key));
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

fn compare_values<F>(a: Option<&Value>, b: &Value, op: F) -> bool
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
        _ => false,
    }
}

/// Secondary Inverted Index for instantaneous O(1) categorical and tag filtering.
#[derive(Debug, Default, Clone)]
pub struct SecondaryIndex {
    // (field_name, string_value) -> doc_indices
    pub index: HashMap<String, HashMap<String, Vec<u32>>>,
}

impl SecondaryIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn index_document(&mut self, doc_id: u32, metadata: &Value) {
        if let Some(map) = metadata.as_object() {
            for (key, val) in map {
                let str_repr = match val {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => b.to_string(),
                    Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                self.index
                    .entry(key.clone())
                    .or_default()
                    .entry(str_repr)
                    .or_default()
                    .push(doc_id);
            }
        }
    }

    pub fn lookup(&self, field: &str, value: &str) -> Option<&Vec<u32>> {
        self.index.get(field).and_then(|m| m.get(value))
    }
}

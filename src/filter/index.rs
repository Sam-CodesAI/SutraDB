use std::collections::{BTreeMap, HashMap};
use roaring::RoaringBitmap;
use serde_json::Value;

/// Maps IEEE 754 f64 to u64 preserving total order (including negatives, zero, and positives).
pub fn float_to_sortable_u64(f: f64) -> u64 {
    let f = if f == 0.0 { 0.0 } else { f };
    let bits = f.to_bits();
    if (bits & 0x8000_0000_0000_0000) != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000_0000_0000
    }
}

/// Inverted and B-Tree Secondary Index for metadata fields.
/// Enables sub-microsecond set intersection and range scans using Roaring Bitmaps.
#[derive(Debug, Clone, Default)]
pub struct SecondaryMetadataIndex {
    /// Categorical inverted index: field_name -> field_value_str -> RoaringBitmap of doc_ids
    categorical: HashMap<String, HashMap<String, RoaringBitmap>>,
    /// Numerical range index: field_name -> sortable_u64 -> RoaringBitmap of doc_ids
    numerical: HashMap<String, BTreeMap<u64, RoaringBitmap>>,
    /// All active document IDs in this index
    all_docs: RoaringBitmap,
    /// Total indexed document count
    doc_count: usize,
}

impl SecondaryMetadataIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Indexes a document's metadata at `doc_id`.
    pub fn index_document(&mut self, doc_id: u32, metadata: &Value) {
        self.all_docs.insert(doc_id);
        if doc_id as usize >= self.doc_count {
            self.doc_count = (doc_id as usize) + 1;
        }

        if let Value::Object(map) = metadata {
            self.index_object_recursive("", map, doc_id);
        }
    }

    fn index_object_recursive(&mut self, prefix: &str, map: &serde_json::Map<String, Value>, doc_id: u32) {
        for (key, val) in map {
            let full_key = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{}.{}", prefix, key)
            };

            match val {
                Value::Null => {}
                Value::Bool(b) => {
                    self.insert_categorical(&full_key, if *b { "true" } else { "false" }, doc_id);
                }
                Value::Number(num) => {
                    if let Some(f) = num.as_f64() {
                        self.insert_numerical(&full_key, f, doc_id);
                    }
                    self.insert_categorical(&full_key, &num.to_string(), doc_id);
                }
                Value::String(s) => {
                    self.insert_categorical(&full_key, s, doc_id);
                }
                Value::Array(arr) => {
                    for item in arr {
                        match item {
                            Value::String(s) => self.insert_categorical(&full_key, s, doc_id),
                            Value::Number(n) => {
                                if let Some(f) = n.as_f64() {
                                    self.insert_numerical(&full_key, f, doc_id);
                                }
                                self.insert_categorical(&full_key, &n.to_string(), doc_id);
                            }
                            Value::Bool(b) => self.insert_categorical(&full_key, if *b { "true" } else { "false" }, doc_id),
                            _ => {}
                        }
                    }
                }
                Value::Object(inner_map) => {
                    self.index_object_recursive(&full_key, inner_map, doc_id);
                }
            }
        }
    }

    fn insert_categorical(&mut self, field: &str, value: &str, doc_id: u32) {
        let field_map = self.categorical.entry(field.to_string()).or_default();
        let bitmap = field_map.entry(value.to_string()).or_default();
        bitmap.insert(doc_id);
    }

    fn insert_numerical(&mut self, field: &str, value: f64, doc_id: u32) {
        let sort_key = float_to_sortable_u64(value);
        let field_btree = self.numerical.entry(field.to_string()).or_default();
        let bitmap = field_btree.entry(sort_key).or_default();
        bitmap.insert(doc_id);
    }

    /// Removes a document ID from all indexes.
    pub fn remove_document(&mut self, doc_id: u32) {
        self.all_docs.remove(doc_id);
        for field_map in self.categorical.values_mut() {
            for bitmap in field_map.values_mut() {
                bitmap.remove(doc_id);
            }
        }
        for field_btree in self.numerical.values_mut() {
            for bitmap in field_btree.values_mut() {
                bitmap.remove(doc_id);
            }
        }
    }

    /// Clears all secondary indexes.
    pub fn clear(&mut self) {
        self.categorical.clear();
        self.numerical.clear();
        self.all_docs.clear();
        self.doc_count = 0;
    }

    /// Returns a bitmap of all currently active documents.
    pub fn all_docs(&self) -> &RoaringBitmap {
        &self.all_docs
    }

    /// Evaluates a query filter AST against secondary indexes.
    /// Returns `Some(RoaringBitmap)` if the filter or part of it was evaluated via index,
    /// or `None` if the filter is null/empty.
    pub fn evaluate_filter(&self, filter: Option<&Value>) -> Option<RoaringBitmap> {
        let filter_val = filter?;
        if filter_val.is_null() {
            return None;
        }

        let filter_obj = filter_val.as_object()?;
        if filter_obj.is_empty() {
            return None;
        }

        Some(self.evaluate_object(filter_obj))
    }

    fn evaluate_object(&self, obj: &serde_json::Map<String, Value>) -> RoaringBitmap {
        let mut result_bitmap = self.all_docs.clone();
        let mut first = true;

        for (key, val) in obj {
            let field_bitmap = match key.as_str() {
                "$and" => {
                    let mut and_bm = self.all_docs.clone();
                    if let Value::Array(arr) = val {
                        for sub in arr {
                            if let Value::Object(sub_obj) = sub {
                                and_bm &= self.evaluate_object(sub_obj);
                            }
                        }
                    }
                    and_bm
                }
                "$or" => {
                    let mut or_bm = RoaringBitmap::new();
                    if let Value::Array(arr) = val {
                        for sub in arr {
                            if let Value::Object(sub_obj) = sub {
                                or_bm |= self.evaluate_object(sub_obj);
                            }
                        }
                    }
                    or_bm
                }
                "$not" => {
                    if let Value::Object(sub_obj) = val {
                        let sub_bm = self.evaluate_object(sub_obj);
                        &self.all_docs - &sub_bm
                    } else {
                        RoaringBitmap::new()
                    }
                }
                _ => self.evaluate_field_condition(key, val),
            };

            if first {
                result_bitmap = field_bitmap;
                first = false;
            } else {
                result_bitmap &= field_bitmap;
            }
        }

        result_bitmap
    }

    fn evaluate_field_condition(&self, field: &str, condition: &Value) -> RoaringBitmap {
        match condition {
            Value::String(s) => {
                self.get_categorical_exact(field, s)
            }
            Value::Bool(b) => {
                self.get_categorical_exact(field, if *b { "true" } else { "false" })
            }
            Value::Number(num) => {
                if let Some(f) = num.as_f64() {
                    self.get_numerical_exact(field, f)
                } else {
                    self.get_categorical_exact(field, &num.to_string())
                }
            }
            Value::Null => {
                self.get_null_or_missing(field)
            }
            Value::Object(map) => {
                let has_op = map.keys().any(|k| k.starts_with('$'));
                if !has_op {
                    // Fallback to all docs for custom structural objects
                    return self.all_docs.clone();
                }

                let mut cond_bm = self.all_docs.clone();
                let mut first = true;

                for (op, target) in map {
                    let op_bm = match op.as_str() {
                        "$eq" => match target {
                            Value::String(s) => self.get_categorical_exact(field, s),
                            Value::Bool(b) => self.get_categorical_exact(field, if *b { "true" } else { "false" }),
                            Value::Number(n) => {
                                if let Some(f) = n.as_f64() {
                                    self.get_numerical_exact(field, f)
                                } else {
                                    self.get_categorical_exact(field, &n.to_string())
                                }
                            }
                            Value::Null => self.get_null_or_missing(field),
                            _ => self.all_docs.clone(),
                        },
                        "$ne" => {
                            let match_bm = match target {
                                Value::String(s) => self.get_categorical_exact(field, s),
                                Value::Bool(b) => self.get_categorical_exact(field, if *b { "true" } else { "false" }),
                                Value::Number(n) => {
                                    if let Some(f) = n.as_f64() {
                                        self.get_numerical_exact(field, f)
                                    } else {
                                        self.get_categorical_exact(field, &n.to_string())
                                    }
                                }
                                Value::Null => self.get_null_or_missing(field),
                                _ => RoaringBitmap::new(),
                            };
                            &self.all_docs - &match_bm
                        }
                        "$in" => {
                            let mut in_bm = RoaringBitmap::new();
                            if let Value::Array(arr) = target {
                                for item in arr {
                                    match item {
                                        Value::String(s) => in_bm |= self.get_categorical_exact(field, s),
                                        Value::Number(n) => {
                                            if let Some(f) = n.as_f64() {
                                                in_bm |= self.get_numerical_exact(field, f);
                                            } else {
                                                in_bm |= self.get_categorical_exact(field, &n.to_string());
                                            }
                                        }
                                        Value::Bool(b) => in_bm |= self.get_categorical_exact(field, if *b { "true" } else { "false" }),
                                        _ => {}
                                    }
                                }
                            }
                            in_bm
                        }
                        "$nin" => {
                            let mut in_bm = RoaringBitmap::new();
                            if let Value::Array(arr) = target {
                                for item in arr {
                                    match item {
                                        Value::String(s) => in_bm |= self.get_categorical_exact(field, s),
                                        Value::Number(n) => {
                                            if let Some(f) = n.as_f64() {
                                                in_bm |= self.get_numerical_exact(field, f);
                                            } else {
                                                in_bm |= self.get_categorical_exact(field, &n.to_string());
                                            }
                                        }
                                        Value::Bool(b) => in_bm |= self.get_categorical_exact(field, if *b { "true" } else { "false" }),
                                        _ => {}
                                    }
                                }
                            }
                            &self.all_docs - &in_bm
                        }
                        "$gt" => {
                            if let Some(f) = target.as_f64() {
                                self.get_numerical_range(field, Some(f), None, false, false)
                            } else {
                                self.all_docs.clone()
                            }
                        }
                        "$gte" => {
                            if let Some(f) = target.as_f64() {
                                self.get_numerical_range(field, Some(f), None, true, false)
                            } else {
                                self.all_docs.clone()
                            }
                        }
                        "$lt" => {
                            if let Some(f) = target.as_f64() {
                                self.get_numerical_range(field, None, Some(f), false, false)
                            } else {
                                self.all_docs.clone()
                            }
                        }
                        "$lte" => {
                            if let Some(f) = target.as_f64() {
                                self.get_numerical_range(field, None, Some(f), false, true)
                            } else {
                                self.all_docs.clone()
                            }
                        }
                        "$exists" => {
                            let expected = target.as_bool().unwrap_or(true);
                            let mut exists_bm = RoaringBitmap::new();
                            if let Some(field_map) = self.categorical.get(field) {
                                for bm in field_map.values() {
                                    exists_bm |= bm;
                                }
                            }
                            if let Some(field_btree) = self.numerical.get(field) {
                                for bm in field_btree.values() {
                                    exists_bm |= bm;
                                }
                            }
                            if expected {
                                exists_bm
                            } else {
                                &self.all_docs - &exists_bm
                            }
                        }
                        _ => {
                            // Unsupported operators by index (like $regex, $geo_within) match all for now,
                            // allowing exact post-filtering to verify
                            self.all_docs.clone()
                        }
                    };

                    if first {
                        cond_bm = op_bm;
                        first = false;
                    } else {
                        cond_bm &= op_bm;
                    }
                }
                cond_bm
            }
            _ => self.all_docs.clone(),
        }
    }

    fn get_null_or_missing(&self, field: &str) -> RoaringBitmap {
        let mut non_null = RoaringBitmap::new();
        if let Some(field_map) = self.categorical.get(field) {
            for bm in field_map.values() {
                non_null |= bm;
            }
        }
        if let Some(field_btree) = self.numerical.get(field) {
            for bm in field_btree.values() {
                non_null |= bm;
            }
        }
        &self.all_docs - &non_null
    }

    fn get_categorical_exact(&self, field: &str, value: &str) -> RoaringBitmap {
        if let Some(field_map) = self.categorical.get(field) {
            if let Some(bm) = field_map.get(value) {
                return bm.clone();
            }
        }
        RoaringBitmap::new()
    }

    fn get_numerical_exact(&self, field: &str, value: f64) -> RoaringBitmap {
        let key = float_to_sortable_u64(value);
        if let Some(btree) = self.numerical.get(field) {
            if let Some(bm) = btree.get(&key) {
                return bm.clone();
            }
        }
        RoaringBitmap::new()
    }

    fn get_numerical_range(
        &self,
        field: &str,
        min: Option<f64>,
        max: Option<f64>,
        include_min: bool,
        include_max: bool,
    ) -> RoaringBitmap {
        let btree = match self.numerical.get(field) {
            Some(b) => b,
            None => return RoaringBitmap::new(),
        };

        let mut res = RoaringBitmap::new();

        let start_bound = match min {
            Some(f) => {
                let sort_val = float_to_sortable_u64(f);
                if include_min {
                    std::ops::Bound::Included(sort_val)
                } else {
                    std::ops::Bound::Excluded(sort_val)
                }
            }
            None => std::ops::Bound::Unbounded,
        };

        let end_bound = match max {
            Some(f) => {
                let sort_val = float_to_sortable_u64(f);
                if include_max {
                    std::ops::Bound::Included(sort_val)
                } else {
                    std::ops::Bound::Excluded(sort_val)
                }
            }
            None => std::ops::Bound::Unbounded,
        };

        for (_, bm) in btree.range((start_bound, end_bound)) {
            res |= bm;
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_float_to_sortable_u64_ordering() {
        let values = vec![-100.0, -10.5, -0.01, 0.0, 0.0001, 42.0, 1000.5];
        for i in 0..values.len() - 1 {
            let u_a = float_to_sortable_u64(values[i]);
            let u_b = float_to_sortable_u64(values[i + 1]);
            assert!(u_a < u_b, "Ordering failed for {} vs {}", values[i], values[i + 1]);
        }
    }

    #[test]
    fn test_secondary_index_basic_and_compound() {
        let mut idx = SecondaryMetadataIndex::new();

        idx.index_document(0, &json!({"tenant": "acme", "status": "active", "score": 95.0, "tags": ["rust", "database"]}));
        idx.index_document(1, &json!({"tenant": "globex", "status": "active", "score": 80.0, "tags": ["python"]}));
        idx.index_document(2, &json!({"tenant": "acme", "status": "pending", "score": 60.5, "tags": ["rust"]}));
        idx.index_document(3, &json!({"tenant": "acme", "status": "active", "score": 40.0, "tags": ["docker"]}));

        // Exact match
        let res = idx.evaluate_filter(Some(&json!({"tenant": "acme"}))).unwrap();
        assert_eq!(res.iter().collect::<Vec<_>>(), vec![0, 2, 3]);

        // Compound AND
        let res_and = idx.evaluate_filter(Some(&json!({"tenant": "acme", "status": "active"}))).unwrap();
        assert_eq!(res_and.iter().collect::<Vec<_>>(), vec![0, 3]);

        // Range $gte
        let res_range = idx.evaluate_filter(Some(&json!({"score": {"$gte": 80.0}}))).unwrap();
        assert_eq!(res_range.iter().collect::<Vec<_>>(), vec![0, 1]);

        // Array tag inclusion ($in or direct exact tag match)
        let res_tag = idx.evaluate_filter(Some(&json!({"tags": "rust"}))).unwrap();
        assert_eq!(res_tag.iter().collect::<Vec<_>>(), vec![0, 2]);

        // Deletion
        idx.remove_document(0);
        let res_after_del = idx.evaluate_filter(Some(&json!({"tenant": "acme", "status": "active"}))).unwrap();
        assert_eq!(res_after_del.iter().collect::<Vec<_>>(), vec![3]);
    }
}

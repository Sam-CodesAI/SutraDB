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

enum IndexEval {
    Exact(RoaringBitmap),
    Superset(RoaringBitmap),
}

impl IndexEval {
    fn into_parts(self) -> (RoaringBitmap, bool) {
        match self {
            Self::Exact(bm) => (bm, true),
            Self::Superset(bm) => (bm, false),
        }
    }
}

/// Inverted and B-Tree Secondary Index for metadata fields.
/// Enables sub-microsecond set intersection and range scans using Roaring Bitmaps.
#[derive(Debug, Clone, Default)]
pub struct SecondaryMetadataIndex {
    /// Categorical inverted index: field_name -> type-tagged value key ("s:<str>", "b:true"/"b:false", "n:<num>") -> RoaringBitmap of doc_ids
    categorical: HashMap<String, HashMap<String, RoaringBitmap>>,
    /// Numerical range index: field_name -> sortable_u64 -> RoaringBitmap of doc_ids
    numerical: HashMap<String, BTreeMap<u64, RoaringBitmap>>,
    /// Presence index for all non-null fields (including objects and empty arrays)
    field_exists: HashMap<String, RoaringBitmap>,
    /// Reverse maps for O(k) document removal
    doc_categorical_keys: HashMap<u32, Vec<(String, String)>>,
    doc_numerical_keys: HashMap<u32, Vec<(String, u64)>>,
    doc_exists_keys: HashMap<u32, Vec<String>>,
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

    fn categorical_string_key(s: &str) -> String {
        format!("s:{}", s)
    }

    fn categorical_bool_key(b: bool) -> &'static str {
        if b {
            "b:true"
        } else {
            "b:false"
        }
    }

    fn categorical_number_key(n: &serde_json::Number) -> String {
        format!("n:{}", n)
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
                    self.insert_exists(&full_key, doc_id);
                    self.insert_categorical(&full_key, Self::categorical_bool_key(*b), doc_id);
                }
                Value::Number(num) => {
                    self.insert_exists(&full_key, doc_id);
                    if let Some(f) = num.as_f64() {
                        self.insert_numerical(&full_key, f, doc_id);
                    } else {
                        self.insert_categorical(&full_key, &Self::categorical_number_key(num), doc_id);
                    }
                }
                Value::String(s) => {
                    self.insert_exists(&full_key, doc_id);
                    self.insert_categorical(&full_key, &Self::categorical_string_key(s), doc_id);
                }
                Value::Array(arr) => {
                    self.insert_exists(&full_key, doc_id);
                    for item in arr {
                        match item {
                            Value::String(s) => {
                                self.insert_categorical(&full_key, &Self::categorical_string_key(s), doc_id)
                            }
                            Value::Number(n) => {
                                if let Some(f) = n.as_f64() {
                                    self.insert_numerical(&full_key, f, doc_id);
                                } else {
                                    self.insert_categorical(&full_key, &Self::categorical_number_key(n), doc_id);
                                }
                            }
                            Value::Bool(b) => {
                                self.insert_categorical(&full_key, Self::categorical_bool_key(*b), doc_id)
                            }
                            _ => {}
                        }
                    }
                }
                Value::Object(inner_map) => {
                    self.insert_exists(&full_key, doc_id);
                    self.index_object_recursive(&full_key, inner_map, doc_id);
                }
            }
        }
    }

    fn insert_exists(&mut self, field: &str, doc_id: u32) {
        let bitmap = self.field_exists.entry(field.to_string()).or_default();
        if bitmap.insert(doc_id) {
            self.doc_exists_keys
                .entry(doc_id)
                .or_default()
                .push(field.to_string());
        }
    }

    fn insert_categorical(&mut self, field: &str, value: &str, doc_id: u32) {
        let field_map = self.categorical.entry(field.to_string()).or_default();
        let bitmap = field_map.entry(value.to_string()).or_default();
        if bitmap.insert(doc_id) {
            self.doc_categorical_keys
                .entry(doc_id)
                .or_default()
                .push((field.to_string(), value.to_string()));
        }
    }

    fn insert_numerical(&mut self, field: &str, value: f64, doc_id: u32) {
        let sort_key = float_to_sortable_u64(value);
        let field_btree = self.numerical.entry(field.to_string()).or_default();
        let bitmap = field_btree.entry(sort_key).or_default();
        if bitmap.insert(doc_id) {
            self.doc_numerical_keys
                .entry(doc_id)
                .or_default()
                .push((field.to_string(), sort_key));
        }
    }

    /// Removes a document ID from all indexes in O(k) time where k is the document's indexed keys.
    pub fn remove_document(&mut self, doc_id: u32) {
        self.all_docs.remove(doc_id);
        if let Some(keys) = self.doc_categorical_keys.remove(&doc_id) {
            for (field, val) in keys {
                let mut remove_field = false;
                if let Some(field_map) = self.categorical.get_mut(&field) {
                    let mut remove_val = false;
                    if let Some(bitmap) = field_map.get_mut(&val) {
                        bitmap.remove(doc_id);
                        remove_val = bitmap.is_empty();
                    }
                    if remove_val {
                        field_map.remove(&val);
                    }
                    remove_field = field_map.is_empty();
                }
                if remove_field {
                    self.categorical.remove(&field);
                }
            }
        }
        if let Some(keys) = self.doc_numerical_keys.remove(&doc_id) {
            for (field, sort_key) in keys {
                let mut remove_field = false;
                if let Some(field_btree) = self.numerical.get_mut(&field) {
                    let mut remove_key = false;
                    if let Some(bitmap) = field_btree.get_mut(&sort_key) {
                        bitmap.remove(doc_id);
                        remove_key = bitmap.is_empty();
                    }
                    if remove_key {
                        field_btree.remove(&sort_key);
                    }
                    remove_field = field_btree.is_empty();
                }
                if remove_field {
                    self.numerical.remove(&field);
                }
            }
        }
        if let Some(keys) = self.doc_exists_keys.remove(&doc_id) {
            for field in keys {
                let mut remove_field = false;
                if let Some(bitmap) = self.field_exists.get_mut(&field) {
                    bitmap.remove(doc_id);
                    remove_field = bitmap.is_empty();
                }
                if remove_field {
                    self.field_exists.remove(&field);
                }
            }
        }
    }

    /// Clears all secondary indexes.
    pub fn clear(&mut self) {
        self.categorical.clear();
        self.numerical.clear();
        self.field_exists.clear();
        self.doc_categorical_keys.clear();
        self.doc_numerical_keys.clear();
        self.doc_exists_keys.clear();
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
        self.evaluate_filter_checked(filter).map(|(bm, _)| bm)
    }

    /// Evaluates a query filter AST against secondary indexes and reports whether the returned
    /// bitmap is exact (`true`) or an over-approximating superset (`false`) requiring FilterEngine verification.
    pub fn evaluate_filter_checked(&self, filter: Option<&Value>) -> Option<(RoaringBitmap, bool)> {
        let filter_val = filter?;
        if filter_val.is_null() {
            return None;
        }

        let filter_obj = filter_val.as_object()?;
        if filter_obj.is_empty() {
            return None;
        }

        Some(self.evaluate_object_checked(filter_obj).into_parts())
    }

    fn evaluate_object_checked(&self, obj: &serde_json::Map<String, Value>) -> IndexEval {
        let mut result_bitmap = self.all_docs.clone();
        let mut all_exact = true;
        let mut first = true;

        for (key, val) in obj {
            let (field_bitmap, field_exact) = match key.as_str() {
                "$and" => {
                    let mut and_bm = self.all_docs.clone();
                    let mut and_exact = true;
                    if let Value::Array(arr) = val {
                        for sub in arr {
                            match sub {
                                Value::Object(sub_obj) if !sub_obj.is_empty() => {
                                    let (sub_bm, sub_exact) =
                                        self.evaluate_object_checked(sub_obj).into_parts();
                                    and_bm &= sub_bm;
                                    and_exact &= sub_exact;
                                }
                                Value::Object(_) | Value::Null => {}
                                _ => {
                                    and_exact = false;
                                }
                            }
                        }
                    } else {
                        and_bm = RoaringBitmap::new();
                    }
                    if !and_exact {
                        and_bm = self.all_docs.clone();
                    }
                    (and_bm, and_exact)
                }
                "$or" => {
                    let mut or_bm = RoaringBitmap::new();
                    let mut or_exact = true;
                    if let Value::Array(arr) = val {
                        for sub in arr {
                            match sub {
                                Value::Object(sub_obj) if !sub_obj.is_empty() => {
                                    let (sub_bm, sub_exact) =
                                        self.evaluate_object_checked(sub_obj).into_parts();
                                    or_bm |= sub_bm;
                                    or_exact &= sub_exact;
                                }
                                Value::Object(_) | Value::Null => {
                                    or_bm = self.all_docs.clone();
                                }
                                _ => {
                                    or_bm = self.all_docs.clone();
                                    or_exact = false;
                                }
                            }
                        }
                    }
                    if !or_exact {
                        or_bm = self.all_docs.clone();
                    }
                    (or_bm, or_exact)
                }
                "$not" => match val {
                    Value::Object(sub_obj) if !sub_obj.is_empty() => {
                        match self.evaluate_object_checked(sub_obj) {
                            IndexEval::Exact(sub_bm) => (&self.all_docs - &sub_bm, true),
                            IndexEval::Superset(_) => (self.all_docs.clone(), false),
                        }
                    }
                    _ => (RoaringBitmap::new(), true),
                },
                _ => self.evaluate_field_condition_checked(key, val).into_parts(),
            };

            all_exact &= field_exact;
            if first {
                result_bitmap = field_bitmap;
                first = false;
            } else {
                result_bitmap &= field_bitmap;
            }
        }

        if all_exact {
            IndexEval::Exact(result_bitmap)
        } else {
            IndexEval::Superset(result_bitmap)
        }
    }

    fn match_scalar_exact(&self, field: &str, val: &Value) -> Option<RoaringBitmap> {
        match val {
            Value::String(s) => Some(self.get_categorical_exact(field, &Self::categorical_string_key(s))),
            Value::Bool(b) => Some(self.get_categorical_exact(field, Self::categorical_bool_key(*b))),
            Value::Number(n) => Some(if let Some(f) = n.as_f64() {
                self.get_numerical_exact(field, f)
            } else {
                self.get_categorical_exact(field, &Self::categorical_number_key(n))
            }),
            _ => None,
        }
    }

    fn evaluate_field_condition_checked(&self, field: &str, condition: &Value) -> IndexEval {
        // Dotted array-index paths (e.g. "items.0") are resolved by FilterEngine, not flat index keys
        if field.split('.').any(|p| p.parse::<usize>().is_ok()) {
            return IndexEval::Superset(self.all_docs.clone());
        }

        match condition {
            Value::String(_) | Value::Bool(_) | Value::Number(_) => {
                if let Some(bm) = self.match_scalar_exact(field, condition) {
                    IndexEval::Exact(bm)
                } else {
                    IndexEval::Superset(self.all_docs.clone())
                }
            }
            Value::Null => IndexEval::Exact(self.get_null_or_missing(field)),
            Value::Object(map) => {
                let has_op = map.keys().any(|k| k.starts_with('$'));
                if !has_op {
                    return IndexEval::Superset(self.all_docs.clone());
                }

                let mut cond_bm = self.all_docs.clone();
                let mut cond_exact = true;
                let mut first = true;

                for (op, target) in map {
                    let (op_bm, op_exact) = match op.as_str() {
                        "$eq" => match target {
                            Value::Null => (self.get_null_or_missing(field), true),
                            _ => match self.match_scalar_exact(field, target) {
                                Some(bm) => (bm, true),
                                None => (self.all_docs.clone(), false),
                            },
                        },
                        "$ne" => match target {
                            Value::Null => (&self.all_docs - &self.get_null_or_missing(field), true),
                            _ => match self.match_scalar_exact(field, target) {
                                Some(bm) => (&self.all_docs - &bm, true),
                                None => (self.all_docs.clone(), false),
                            },
                        },
                        "$in" => {
                            let mut in_bm = RoaringBitmap::new();
                            let mut exact = true;
                            if let Value::Array(arr) = target {
                                for item in arr {
                                    if let Some(bm) = self.match_scalar_exact(field, item) {
                                        in_bm |= bm;
                                    } else {
                                        exact = false;
                                    }
                                }
                            } else {
                                exact = false;
                            }
                            if !exact {
                                in_bm = self.all_docs.clone();
                            }
                            (in_bm, exact)
                        }
                        "$nin" => {
                            let mut in_bm = RoaringBitmap::new();
                            let mut exact = true;
                            if let Value::Array(arr) = target {
                                for item in arr {
                                    if let Some(bm) = self.match_scalar_exact(field, item) {
                                        in_bm |= bm;
                                    } else {
                                        exact = false;
                                    }
                                }
                            }
                            if exact {
                                (&self.all_docs - &in_bm, true)
                            } else {
                                (self.all_docs.clone(), false)
                            }
                        }
                        "$gt" => {
                            if let Some(f) = target.as_f64() {
                                (
                                    self.get_numerical_range(field, Some(f), None, false, false),
                                    true,
                                )
                            } else {
                                (self.all_docs.clone(), false)
                            }
                        }
                        "$gte" => {
                            if let Some(f) = target.as_f64() {
                                (
                                    self.get_numerical_range(field, Some(f), None, true, false),
                                    true,
                                )
                            } else {
                                (self.all_docs.clone(), false)
                            }
                        }
                        "$lt" => {
                            if let Some(f) = target.as_f64() {
                                (
                                    self.get_numerical_range(field, None, Some(f), false, false),
                                    true,
                                )
                            } else {
                                (self.all_docs.clone(), false)
                            }
                        }
                        "$lte" => {
                            if let Some(f) = target.as_f64() {
                                (
                                    self.get_numerical_range(field, None, Some(f), false, true),
                                    true,
                                )
                            } else {
                                (self.all_docs.clone(), false)
                            }
                        }
                        "$exists" => {
                            let expected = target.as_bool().unwrap_or(false);
                            let exists_bm =
                                self.field_exists.get(field).cloned().unwrap_or_default();
                            if expected {
                                (exists_bm, true)
                            } else {
                                (&self.all_docs - &exists_bm, true)
                            }
                        }
                        _ => (self.all_docs.clone(), false),
                    };

                    cond_exact &= op_exact;
                    if first {
                        cond_bm = op_bm;
                        first = false;
                    } else {
                        cond_bm &= op_bm;
                    }
                }
                if cond_exact {
                    IndexEval::Exact(cond_bm)
                } else {
                    IndexEval::Superset(cond_bm)
                }
            }
            _ => IndexEval::Superset(self.all_docs.clone()),
        }
    }

    fn get_null_or_missing(&self, field: &str) -> RoaringBitmap {
        let non_null = self.field_exists.get(field).cloned().unwrap_or_default();
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
        let values = [-100.0, -10.5, -0.01, 0.0, 0.0001, 42.0, 1000.5];
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

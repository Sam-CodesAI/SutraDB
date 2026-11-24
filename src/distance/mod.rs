pub mod bq;
pub mod pq;
pub mod simd;
pub mod sq8;

use serde::{Deserialize, Serialize};
use std::str::FromStr;
use crate::error::SutraError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    #[serde(rename = "cosine")]
    Cosine = 0,
    #[serde(rename = "euclidean")]
    Euclidean = 1,
    #[serde(rename = "dot_product")]
    DotProduct = 2,
}

impl Metric {
    pub fn as_str(&self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::DotProduct => "dot_product",
        }
    }

    pub fn to_u8(&self) -> u8 {
        *self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self, SutraError> {
        match val {
            0 => Ok(Metric::Cosine),
            1 => Ok(Metric::Euclidean),
            2 => Ok(Metric::DotProduct),
            _ => Err(SutraError::InvalidHeader(format!("Unknown metric code: {}", val))),
        }
    }
}

impl FromStr for Metric {
    type Err = SutraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "cosine" => Ok(Metric::Cosine),
            "euclidean" => Ok(Metric::Euclidean),
            "dot_product" | "dot" => Ok(Metric::DotProduct),
            _ => Err(SutraError::InvalidQuery(format!(
                "Unsupported metric: '{}'. Supported metrics are 'cosine', 'euclidean', 'dot_product'.",
                s
            ))),
        }
    }
}

use std::collections::BTreeMap;

/// A single feature or input value.
///
/// `Missing` is distinct from `Number(0.0)`: an absent input must never be
/// silently coerced into a zero, which would read as "no liquidity"/"flat book".
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureValue {
    Number(f64),
    Missing,
}

impl FeatureValue {
    pub fn number(&self) -> Option<f64> {
        match self {
            Self::Number(v) => Some(*v),
            Self::Missing => None,
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

impl From<f64> for FeatureValue {
    fn from(v: f64) -> Self {
        Self::Number(v)
    }
}

/// One observation: a flat map from column name to value.
///
/// This is the seam between the two halves of `core::feature`: `ingest` builds
/// rows from a websocket frame, `compute` consumes the same row shape and
/// returns a row of features. Neither knows about the other.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureRow {
    values: BTreeMap<String, FeatureValue>,
}

impl FeatureRow {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, value: f64) {
        self.values.insert(name.into(), FeatureValue::Number(value));
    }

    /// Record an explicitly absent value, keeping it distinguishable from zero.
    pub fn insert_missing(&mut self, name: impl Into<String>) {
        self.values.insert(name.into(), FeatureValue::Missing);
    }

    pub fn insert_value(&mut self, name: impl Into<String>, value: FeatureValue) {
        self.values.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).and_then(FeatureValue::number)
    }

    pub fn get_value(&self, name: &str) -> Option<FeatureValue> {
        self.values.get(name).copied()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, FeatureValue)> {
        self.values.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

impl FromIterator<(String, FeatureValue)> for FeatureRow {
    fn from_iter<I: IntoIterator<Item = (String, FeatureValue)>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_is_not_zero() {
        let mut row = FeatureRow::new();
        row.insert("bid", 0.0);
        row.insert_missing("ask");
        assert_eq!(row.get("bid"), Some(0.0));
        assert_eq!(row.get("ask"), None);
        assert!(row.get_value("ask").unwrap().is_missing());
    }

    #[test]
    fn round_trips_through_name_value_pairs() {
        let mut row = FeatureRow::new();
        row.insert("a", 1.0);
        row.insert_missing("b");
        let rebuilt: FeatureRow = row.iter().map(|(k, v)| (k.to_string(), v)).collect();
        assert_eq!(rebuilt, row);
    }
}

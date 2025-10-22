/// tolerance for numerical comparisons to avoid floating point errors
pub const NUMERICAL_TOLERANCE: f64 = 1e-12;

/// performs division with nan protection for near-zero denominators
pub fn safe_divide(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() < NUMERICAL_TOLERANCE {
        if numerator.abs() < NUMERICAL_TOLERANCE {
            f64::NAN
        } else {
            f64::INFINITY * numerator.signum() * denominator.signum()
        }
    } else {
        numerator / denominator
    }
}

/// checks if a value is within numerical tolerance of zero
pub fn is_numerically_zero(value: f64) -> bool {
    value.abs() < NUMERICAL_TOLERANCE
}

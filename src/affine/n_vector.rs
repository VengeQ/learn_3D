use crate::affine::Affine;

///
/// Represent N-size vector
///
/// Todo[Daniil] Need to difference n-dimension and m-dimension vector
///
#[derive(Default, Debug, PartialOrd, PartialEq, Clone)]
pub struct NVector {
    size: usize,
    set: Vec<f64>,
}

impl NVector {
    ///
    /// Create new n-dimension Vector from Vec<f64>
    /// # Example
    /// ```
    /// use learn_3D::vector::n_vector::NVector;
    /// let vector_4d = NVector::new(vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    pub fn new(input: Vec<f64>) -> Self {
        Self {
            size: input.len(),
            set: input,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_vector_new_smoke_test() {
        let vector = NVector::new(vec![1.0, 3.0, 2.0, 4.0, 8.0]);
    }
}
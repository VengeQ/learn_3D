use std::ops::{Add, Sub, Mul};
use crate::affine::Affine;

///
/// Implementation in 3-space.
///
#[derive(Default, Debug, PartialOrd, PartialEq, Copy, Clone)]
pub struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3 {
    ///
    /// Create new Vector
    /// # Example
    /// ```
    /// use learn_3D::vector::vector_3::Vector3;
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    /// ```
    ///
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
        }
    }

    ///ToDo[Daniil] Need more clear and effective impl
    ///
    /// Return [Vector projection](https://en.wikipedia.org/wiki/Vector_projection "Wiki explanation") on another vector.
    ///
    pub fn projection(&self, proj_vector: Self) -> Self {
        let numer = proj_vector.scalar_product(self.dot_product(proj_vector));
        let proj_vector_norma = proj_vector.normalize();
        let denom = proj_vector_norma.dot_product(proj_vector_norma);

        proj_vector_norma.scalar_product(1.0 / denom)
    }

    /// Perpendicular to vector projection
    pub fn perp(&self, proj_vector: Self) -> Self {
        *self - self.projection(proj_vector)
    }

    fn err(left: Self, right: Self) -> f64 {
        (left.x - right.x).abs() + (left.y - right.y).abs() + (left.z - right.z).abs()
    }

    ///Return coordinates of vector as tuple3
     /// # Example
    /// ```
    /// use learn_3D::vector::vector_3::Vector3;
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(vec.xyz(),(1.0, 2.0, 3.0));
    /// ```
    pub fn xyz(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    pub fn distance(v1: Vector3, v2: Vector3) -> f64 {
        let x = v1.x - v2.x;
        let y = v1.y - v2.y;
        let z = v1.z - v2.z;
        (x * x + y * y + z * z).sqrt()
    }

    pub fn distance_squared(v1: Vector3, v2: Vector3) -> f64 {
        let x = v1.x - v2.x;
        let y = v1.y - v2.y;
        let z = v1.z - v2.z;
        x * x + y * y + z * z
    }

    ///
    /// Return product vector on value
    /// # Example
    /// ```
    /// use learn_3D::vector::vector_3::Vector3;
    /// use learn_3D::vector::Affine;
    /// let x = 1.0;
    /// let y = 2.0;
    /// let z = 3.0;
    /// let vec = Vector3::new(x, y, z);
    /// let vec2 = vec.scalar_product(2.0);
    /// assert_eq!(Vector3::new(x * 2.0, y * 2.0, z * 2.0), vec2);
    /// ```

    pub fn scalar_product(&self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    pub fn normalize(&self) -> Self {
        if self == &Self::zero() {
            Self::zero()
        } else {
            let magnitude = 1_f64 / self.length();
            self.scalar_product(magnitude)
        }
    }

    pub fn dot_product(&self, right: Self) -> f64 {
        self.x * right.x + self.y * right.y + self.z * right.z
    }

    pub fn cross_product(&self, right: Self) -> Self {
        let x = self.y * right.z - right.y * self.z;
        let y = self.z * right.x - right.z * self.x;
        let z = self.x * right.y - right.x * self.y;
        Vector3::new(x, y, z)
    }
}

impl Affine for Vector3 {
    /// Return zero vector in 3-dimension space (x=0,y=0,z=0)
    /// # Example
    /// ```
    /// use learn_3D::vector::vector_3::Vector3;
    /// use learn_3D::vector::Affine;
    /// let vec = Vector3::zero();
    /// ```
    ///
    fn zero() -> Self {
        Self::default()
    }

    fn polar(&self) -> (f64, f64, f64) {
        unimplemented!()
    }

    /// For vector (x,y,z) return Vector(-x,-y,-z)
    fn reverse(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    ///Return magnitude of 3-space Vector
    /// magnitude = sqrt(x^2+y^2+z^2)
    fn length(&self) -> f64 {
        self.x.mul_add(self.x, self.y.mul_add(self.y, self.z * self.z)).sqrt()
    }
}

impl Add for Vector3 {
    type Output = Vector3;

    fn add(self, rhs: Self) -> Self::Output {
        Vector3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector3 {
    type Output = Vector3;

    fn sub(self, rhs: Self) -> Self::Output {
        let tempo = rhs.reverse();
        self + tempo
    }
}

impl Mul for Vector3 {
    type Output = Vector3;

    fn mul(self, rhs: Self) -> Self::Output {
        self.cross_product(rhs)
    }
}

impl From<(f64, f64, f64)> for Vector3 {
    fn from(v: (f64, f64, f64)) -> Self {
        Vector3::new(v.0, v.1, v.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    const DIFF_ERR: f64 = 10e-7;

    #[test]
    fn vector_new_test() {
        let vector = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(vector.x, 1.0);
        assert_eq!(vector, Vector3::new(1.0, 2.0, 3.0))
    }

    #[test]
    fn vector_add_test() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(2.0, 4.0, 11.0);
        let result = v1 + v2;
        assert_eq!(result, Vector3::new(1.0 + 2.0, 2.0 + 4.0, 3.0 + 11.0));
    }

    #[test]
    fn vector_reverse_test() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(-1.0, -2.0, -3.0);
        let reversed = v1.reverse();

        assert_eq!(reversed, v2)
    }

    #[test]
    fn vector_sub_test() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let result = v1 - v1;
        assert_eq!(result, Vector3::zero());
    }

    #[test]
    fn vector_zero_test() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let zero = Vector3::zero();
        assert_eq!(v1 + zero, zero + v1);
    }

    #[test]
    fn vector_scalar_prod_test() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = v1.scalar_product(4.0);
        assert_eq!(v2, Vector3::new(1.0 * 4.0, 2.0 * 4.0, 3.0 * 4.0))
    }

    #[test]
    fn vector_length_test() {
        let vector = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(vector.length(), (1.0_f64 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt());
    }

    #[test]
    fn vector_magnitude_rules_test() {
        let mut rng = rand::thread_rng();
        //||av||=|a|*||v||
        for _ in 0..100 {
            let x = rng.gen_range(0, 100) as f64;
            let y = rng.gen_range(0, 100) as f64;
            let z = rng.gen_range(0, 100) as f64;
            let vector = Vector3::new(x, y, z);
            let a = rng.gen_range(0, 100) as f64;

            let av = vector.length() * a;
            let a_v = vector.scalar_product(a).length();
            let difference = av - a_v;
            assert!(difference <= 10e-8);
        }
        //||v+w|| <= ||v||+||w||
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 100) as f64;
            let y1 = rng.gen_range(0, 100) as f64;
            let z1 = rng.gen_range(0, 100) as f64;
            let x2 = rng.gen_range(0, 100) as f64;
            let y2 = rng.gen_range(0, 100) as f64;
            let z2 = rng.gen_range(0, 100) as f64;

            let v = Vector3::new(x1, y1, z1);
            let w = Vector3::new(x2, y2, z2);

            let left = (v + w).length();
            let right = v.length() + w.length();
            assert!(left <= right)
        }
    }

    #[test]
    fn vector_normalize_test() {
        let vector = Vector3::new(1.0, 2.0, 3.0);
        let den = vector.length();
        let normalized = Vector3::new(vector.x / den, vector.y / den, vector.z / den);
        //    dbg!("{:?}", normalized);
        assert_eq!(normalized, vector.normalize());
        assert!(normalized.length() - 1.0 <= 10e-7 || normalized.length() == 0.0);
    }

    #[test]
    fn vector_dot_product_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 100) as f64;
            let y1 = rng.gen_range(0, 100) as f64;
            let z1 = rng.gen_range(0, 100) as f64;
            let x2 = rng.gen_range(0, 100) as f64;
            let y2 = rng.gen_range(0, 100) as f64;
            let z2 = rng.gen_range(0, 100) as f64;

            let v = Vector3::new(x1, y1, z1);
            let w = Vector3::new(x2, y2, z2);

            let dot_prod = v.dot_product(w);
            let prod = x1 * x2 + y1 * y2 + z1 * z2;

            assert_eq!(dot_prod, prod);
        }
    }

    #[test]
    fn vector_dot_product_rules_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 100) as f64 / 10.0;
            let y1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z1 = rng.gen_range(0, 1000) as f64 / 10.0;

            let x2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z2 = rng.gen_range(0, 1000) as f64 / 10.0;

            let x3 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y3 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z3 = rng.gen_range(0, 1000) as f64 / 10.0;

            let v = Vector3::new(x1, y1, z1);
            let w = Vector3::new(x2, y2, z2);
            let u = Vector3::new(x3, y3, z3);

            let a = rng.gen_range(0, 1000) as f64 / 10.0;

            //symmetry w*v = v*w
            assert!(v.dot_product(w) - w.dot_product(v) <= DIFF_ERR);

            //additivity (u+v)*w = u*w +v*w
            assert!((u + v).dot_product(w) - (u.dot_product(w) + v.dot_product(w)) <= DIFF_ERR);

            //homogeneity a(v*w) =(av)*w = v*(aw)
            assert!(v.dot_product(w) * a - v.scalar_product(a).dot_product(w) <= DIFF_ERR);

            //positivity
            assert!(v.dot_product(v) >= 0.0);

            //definiteness
            if v != Vector3::zero() {
                assert_ne!(v.dot_product(v), 0.0);
            }
            assert_eq!(Vector3::zero().dot_product(Vector3::zero()), 0.0)
        }
    }

    #[test]
    fn vector_projection_smoke_test() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let w = Vector3::new(3.0, 4.0, 5.0);
        let v_proj_w = v.projection(w);
        //println!("{:?}", v_proj_w);
    }

    #[test]
    fn vector_cross_product_smoke_test() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let w = Vector3::new(3.0, 4.0, 5.0);
        let cross_v_w = v * w;
        //println!("{:?}", cross_v_w);
    }

    #[test]
    fn vector_cross_product_rules_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 100) as f64 / 10.0;
            let y1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z1 = rng.gen_range(0, 1000) as f64 / 10.0;

            let x2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z2 = rng.gen_range(0, 1000) as f64 / 10.0;

            let x3 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y3 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z3 = rng.gen_range(0, 1000) as f64 / 10.0;

            let v = Vector3::new(x1, y1, z1);
            let w = Vector3::new(x2, y2, z2);
            let u = Vector3::new(x3, y3, z3);

            let a = rng.gen_range(0, 1000) as f64 / 10.0;

            // w × v = -w × v
            assert!(Vector3::err(v.cross_product(w), (w.cross_product(v)).reverse()) <= DIFF_ERR);

            // u × (v + w) = (u × v) + (u × w)
            assert!(Vector3::err(u * (v + w), u * v + u * w) <= DIFF_ERR);

            //(u + v) × w = (u × w) + (v × w)
            assert!(Vector3::err((u + v) * w, u * w + v * w) <= DIFF_ERR);

            //a(v × w) = (av) × w = v × (aw).
            assert!(Vector3::err((v * w).scalar_product(a), v.scalar_product(a) * w) <= DIFF_ERR);

            //v × 0 = 0 × v = 0
            assert!(Vector3::err(v * Vector3::zero(), Vector3::zero() * v) <= DIFF_ERR);
            assert!(Vector3::err(v * Vector3::zero(), Vector3::zero()) <= DIFF_ERR);

            //v × v = 0.
            assert!(Vector3::err(v * v, Vector3::zero()) <= DIFF_ERR);
        }
    }

    #[test]
    fn vector_triple_product_rules() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 100) as f64 / 10.0;
            let y1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z1 = rng.gen_range(0, 1000) as f64 / 10.0;

            let x2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z2 = rng.gen_range(0, 1000) as f64 / 10.0;

            let x3 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y3 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z3 = rng.gen_range(0, 1000) as f64 / 10.0;

            let v = Vector3::new(x1, y1, z1);
            let w = Vector3::new(x2, y2, z2);
            let u = Vector3::new(x3, y3, z3);

            let a = rng.gen_range(0, 1000) as f64 / 10.0;

            //u × (v × w) = (u · w)v − (u · v)w.
            let left = u * (v * w);
            let right = v.scalar_product(u.dot_product(w)) - w.scalar_product(u.dot_product(v));
            assert!(Vector3::err(left, right) <= DIFF_ERR);

            //(u × v) × w = (u · w)v − (v · w)u
            let left = (u * v) * w;
            let right = v.scalar_product(u.dot_product(w)) - u.scalar_product(v.dot_product(w));
            assert!(Vector3::err(left, right) <= DIFF_ERR);

            //u · (v × w) = w· (u × v) = v · (w × u).
            let first = u.dot_product(v * w);
            let second = w.dot_product(u * v);
            let third = v.dot_product(w * u);
            assert!((first - second).abs() <= DIFF_ERR);
            assert!((second - third).abs() <= DIFF_ERR);
            assert!((third - first).abs() <= DIFF_ERR);
        }
    }
}

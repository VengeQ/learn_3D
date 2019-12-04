use crate::object_3d::vector3d::Vector3d;
use crate::object_3d::affine3d::Affine3d;
use std::ops::{Add, Sub};
use std::fmt::{Display, Formatter, Error};

///
/// Matrix represent as plain array in column major order
///
/// [ 0  3  6 ]
///
/// [ 1  4  7 ]
///
/// [ 2  5  8 ]
///
#[derive(Debug, PartialOrd, PartialEq, Copy, Clone)]
pub struct Matrix3d {
    inner: [f64; 9]
}

impl Display for Matrix3d {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:4}   {:4}    {:4}\n{:4}   {:4}    {:4}\n{:4}   {:4}    {:4}\n",
               self.inner[0], self.inner[3], self.inner[6],
               self.inner[1], self.inner[4], self.inner[7],
               self.inner[2], self.inner[5], self.inner[8])
    }
}


impl Matrix3d {
    ///Create 3x3 matrix filled zero.
    pub fn new_zero() -> Self {
        let inner = [0.0; 9];
        Matrix3d { inner }.to_owned()
    }

    /// Create 3x3 matrix from array.
    /// There is column major order in use
    /// Follow matrix will be create in example below.
    ///
    /// [ 0  3  6 ]
    ///
    /// [ 1  4  7 ]
    ///
    /// [ 2  5  8 ]
    ///
    /// # Example
    /// ```
    /// use astra::object_3d::matrix3d::Matrix3d;
    /// let array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let matrix = Matrix3d::from_array(array);
    /// ```
    pub fn from_array(input: [f64; 9]) -> Self {
        Self { inner: input }
    }

    ///Create 3x3 matrix filled zero from borrowed 9-length array
    pub fn from_borrowed(input: &[f64; 9]) -> Self {
        let inner: [f64; 9] = [input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8]];
        Self { inner }
    }

    ///Create matrix from columns
    /// Example below create matrix:
    ///
    /// [ 3.0  7.0  -2.5 ]
    ///
    /// [ 1.5  2.5   3.0 ]
    ///
    /// [ 4.0  4.5  -1.0 ]
    /// # Example
    /// ```
    /// use astra::object_3d::matrix3d::Matrix3d;
    /// let col0 = [3.0, 1.5, 4.0];
    /// let col1 = [7.0, 2.5, 4.5];
    /// let col2 = [-2.5, 3.0, -1.0];
    /// let matrix = Matrix3d::from_cols(col0, col1, col2);
    /// ```
    pub fn from_cols(column0: [f64; 3], column1: [f64; 3], column2: [f64; 3]) -> Self {
        Matrix3d::from_array([
            column0[0], column0[1], column0[2],
            column1[0], column1[1], column1[2],
            column2[0], column2[1], column2[2]
        ])
    }

    ///Return value of index
    pub fn idx(&self, row: usize, col: usize) -> f64 {
        self.inner[row + 3 * col]
    }

    ///Return tuple-3 of rows
    /// # Example
    /// ```
    /// use astra::object_3d::matrix3d::Matrix3d;
    /// let array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let matrix = Matrix3d::from_array(array);
    /// assert_eq!(matrix.columns().1, [3.0, 4.0, 5.0]);
    /// ```
    pub fn columns(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let row0 = [self.inner[0], self.inner[1], self.inner[2]];
        let row1 = [self.inner[3], self.inner[4], self.inner[5]];
        let row2 = [self.inner[6], self.inner[7], self.inner[8]];
        (row0, row1, row2)
    }

    ///Return tuple-3 of columns
    /// # Example
    /// ```
    /// use astra::object_3d::matrix3d::Matrix3d;
    /// let array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let matrix = Matrix3d::from_array(array);
    /// assert_eq!(matrix.rows().1, [1.0, 4.0, 7.0]);
    /// ```
    pub fn rows(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let row0 = [self.inner[0], self.inner[3], self.inner[6]];
        let row1 = [self.inner[1], self.inner[4], self.inner[7]];
        let row2 = [self.inner[2], self.inner[5], self.inner[8]];
        (row0, row1, row2)
    }

    ///Return product matrix on vector
    pub fn vector_product(&self, other: &Vector3d) -> Vector3d {
        let x = self.inner[0] * other.x() + self.inner[3] * other.y() + self.inner[6] * other.z();
        let y = self.inner[1] * other.x() + self.inner[4] * other.y() + self.inner[7] * other.z();
        let z = self.inner[2] * other.x() + self.inner[5] * other.y() + self.inner[8] * other.z();
        Vector3d::new(x, y, z)
    }

    ///Return product of matrices, equal to Mul implementation
    pub fn matrix_product(&self, other: &Matrix3d) -> Matrix3d {
        let mut result = Self::new_zero();
        let other = other.inner;
        result.inner[0] = self.inner[0] * other[0] + self.inner[3] * other[1] + self.inner[6] * other[2];
        result.inner[1] = self.inner[1] * other[0] + self.inner[4] * other[1] + self.inner[7] * other[2];
        result.inner[2] = self.inner[2] * other[0] + self.inner[5] * other[1] + self.inner[8] * other[2];

        result.inner[3] = self.inner[0] * other[3] + self.inner[3] * other[4] + self.inner[6] * other[5];
        result.inner[4] = self.inner[1] * other[3] + self.inner[4] * other[4] + self.inner[7] * other[5];
        result.inner[5] = self.inner[2] * other[3] + self.inner[5] * other[4] + self.inner[8] * other[5];

        result.inner[6] = self.inner[0] * other[6] + self.inner[3] * other[7] + self.inner[6] * other[8];
        result.inner[7] = self.inner[1] * other[6] + self.inner[4] * other[7] + self.inner[7] * other[8];
        result.inner[8] = self.inner[2] * other[6] + self.inner[5] * other[7] + self.inner[8] * other[8];

        result
    }

    ///Reverse sign of all elements of Matrix3d
    pub fn negate(&self) -> Self {
        Self::from_array([
            -self.inner[0], -self.inner[1], -self.inner[2],
            -self.inner[3], -self.inner[4], -self.inner[5],
            -self.inner[6], -self.inner[7], -self.inner[8]
        ])
    }

    ///Return transposed matrix
    /// # Example
    /// ```
    /// use astra::object_3d::matrix3d::Matrix3d;
    /// let matrix = Matrix3d::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let transposed = matrix.transpose();
    /// assert_eq!(matrix.rows(), transposed.columns());
    /// assert_eq!(matrix.columns(),transposed.rows());
    /// ```
    pub fn transpose(&self) -> Self {
        Self::from_array([
            self.inner[0], self.inner[3], self.inner[6],
            self.inner[1], self.inner[4], self.inner[7],
            self.inner[2], self.inner[5], self.inner[8]
        ])
    }

    ///Return Product Matrix to scalar.
    pub fn scalar_product(&self, scalar: f64) -> Self {
        let inner = self.inner;
        let na = [
            inner[0] * scalar, inner[1] * scalar, inner[2] * scalar,
            inner[3] * scalar, inner[4] * scalar, inner[5] * scalar,
            inner[6] * scalar, inner[7] * scalar, inner[8] * scalar
        ];
        Self::from_array(na)
    }

    /// Return diagonal matrix
    ///
    /// [ 1  0  0 ]
    ///
    /// [ 0  1  0 ]
    ///
    /// [ 0  0  1 ]
    pub fn new_diagonal() -> Self {
        let inner = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        Matrix3d { inner }.to_owned()
    }

    ///ToDo/[Daniil/]
    pub fn linear_eq(&self, v: Vector3d) -> Result<(f64, f64, f64), String> {
        let mut matrix = self.to_owned();
        let mut vector = [v.x(), v.y(), v.z()];
        for p in 0..2 {
            if matrix.idx(0, p) < matrix.idx(2, p) {
                matrix = matrix.swap_rows(0, 2);
            }
            if matrix.idx(0, p) < matrix.idx(1, p) {
                matrix = matrix.swap_rows(0, 1);
            }
            if matrix.idx(0, p) == 0.0 {
                return Err("To zero".to_owned());
            }
            println!("swap:\n{}", (matrix));
            let scale = matrix.inner[p + p * 3];
            println!("scale:{}", scale);
            matrix.inner[3 + p] = matrix.inner[3 + p] / scale;
            matrix.inner[6 + p] = matrix.inner[6 + p] / scale;
            vector[p] = vector[p] / matrix.inner[0 + p];
            matrix.inner[p + p * 3] = 1.0;

            println!("scale_first:\n{}", (matrix));
            for r in p + 1..=2 {
                let scale_p = -matrix.inner[r + p * 3] / matrix.inner[p + p * 3];
                println!("scale:\n {}", (scale_p));
                let tempo = [matrix.inner[p] * scale_p, matrix.inner[p + 3] * scale_p, matrix.inner[p + 6] * scale_p];
                println!("tempo:\n {:?}", (tempo));
                matrix.inner[r] = matrix.inner[r] + tempo[0];
                matrix.inner[r + 3] = matrix.inner[r + 3] + tempo[1];
                matrix.inner[r + 6] = matrix.inner[r + 6] + tempo[2];
                vector[r] = vector[r] - matrix.inner[0 + r];
                println!("scale after:\n{}", (matrix));
            }
        }
        println!("Result:\n{}", (matrix));

        Ok((0.0, 0.0, 0.0))
    }

    pub fn swap_rows(&self, i: usize, j: usize) -> Self {
        let mut matrix = self.to_owned();
        matrix.inner.swap(i, j);
        matrix.inner.swap(i + 3, j + 3);
        matrix.inner.swap(i + 6, j + 6);
        matrix
    }

    ///Return new matrix with swapped columns
    pub fn swap_cols(&self, i: usize, j: usize) -> Self {
        let mut matrix = self.to_owned();
        matrix.inner.swap(i * 3, j * 3);
        matrix.inner.swap(i * 3 + 1, j * 3 + 1);
        matrix.inner.swap(i * 3 + 1, j * 3 + 1);
        matrix
    }

    /// Return determinant of matrix
    /// # Example
    /// ```
    /// use astra::object_3d::matrix3d::Matrix3d;
    /// let m1 = Matrix3d::from_array([2.0,4.0,7.0,3.0,4.0,9.0,11.0,5.0,6.0]);
    /// assert_eq!(m1.det(),79.0);
    /// ```
    pub fn det(&self) -> f64 {
        let inner = &self.inner;
        inner[0] * Self::det2([inner[4], inner[5], inner[7], inner[8]])
            - inner[3] * Self::det2([inner[1], inner[2], inner[7], inner[8]])
            + inner[6] * Self::det2([inner[1], inner[2], inner[4], inner[5]])
    }

    fn det2(input: [f64; 4]) -> f64 {
        input[0] * input[3] - input[2] * input[1]
    }
    /// Return [Adjunct matrix](https://en.wikipedia.org/wiki/Adjugate_matrix)
    /// Determinant of matrix must be nonzero
    /// Example below return matrix
    ///
    /// [-11   6  -1]
    ///
    /// [-2  -12   8]
    ///
    /// [10    6  -4]
    /// # Example
    /// ```
    ///  use astra::object_3d::matrix3d::Matrix3d;
    ///  let matrix = Matrix3d::from_cols([0.0,1.0,2.0],[4.0,3.0,5.0],[6.0,7.0,8.0]);
    ///  let adj = matrix.adj_matrix();
    /// ```
    pub fn adj_matrix(&self) -> Self {
        let t = self.to_owned();
        let z = t.inner;
        let m0 = Self::det2([z[4], z[5], z[7], z[8]]);
        let m1 = -Self::det2([z[3], z[5], z[6], z[8]]);
        let m2 = Self::det2([z[3], z[4], z[6], z[7]]);

        let m3 = -Self::det2([z[1], z[2], z[7], z[8]]);
        let m4 = Self::det2([z[0], z[2], z[6], z[8]]);
        let m5 = -Self::det2([z[0], z[1], z[6], z[7]]);

        let m6 = Self::det2([z[1], z[2], z[4], z[5]]);
        let m7 = -Self::det2([z[0], z[2], z[3], z[5]]);
        let m8 = Self::det2([z[0], z[1], z[3], z[4]]);
        Self::from_array([m0, m1, m2, m3, m4, m5, m6, m7, m8])
    }

    ///Return inverse matrix as option.
    /// If determinant equal to zero then return None.
    pub fn inverse(&self) -> Option<Self> {
        let det = self.det();
        println!("{}", det);
        if det == 0.0 {
            None
        } else {
            println!("{}", self.adj_matrix());
            Some(self.adj_matrix().transpose().scalar_product(1.0 / det))
        }
    }
}

impl Add for Matrix3d {
    type Output = Matrix3d;

    fn add(self, rhs: Self) -> Self::Output {
        let l = self.inner;
        let r = rhs.inner;
        Matrix3d::from_array([
            l[0] + r[0], l[1] + r[1], l[2] + r[2],
            l[3] + r[3], l[4] + r[4], l[5] + r[5],
            l[6] + r[6], l[7] + r[7], l[8] + r[8],
        ])
    }
}

impl Sub for Matrix3d {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs.negate()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn matrix3d_new_zero_test() {
        let m = Matrix3d::new_zero();
        m.inner.iter().for_each(|x| assert_eq!(*x, 0.0))
    }

    #[test]
    fn matrix3d_new_diagonal_test() {
        let m = Matrix3d::new_zero();
        m.inner.iter().for_each(|x| assert!(*x >= 0.0 && *x <= 1.0))
    }

    #[test]
    fn matrix3d_columns_test() {
        let m = Matrix3d::new_diagonal();
        let columns = m.columns();
        assert_eq!(columns.0, [1.0, 0.0, 0.0]);
        assert_eq!(columns.1, [0.0, 1.0, 0.0]);
        assert_eq!(columns.2, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn matrix3d_rows_test() {
        let m = Matrix3d::new_diagonal();
        let rows = m.rows();
        assert_eq!(rows.0, [1.0, 0.0, 0.0]);
        assert_eq!(rows.1, [0.0, 1.0, 0.0]);
        assert_eq!(rows.2, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn from_borrowed_test() {
        let i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Matrix3d::from_borrowed(&i);
        assert_eq!(m.rows().0, [1.0, 4.0, 7.0]);
    }

    #[test]
    fn from_array_test() {
        let i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Matrix3d::from_array(i);
        println!("{}", m);
        assert_eq!(m.rows().0, [1.0, 4.0, 7.0]);
    }

    #[test]
    fn idx_test() {
        let i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Matrix3d::from_array(i);
        assert_eq!(m.idx(2, 1), 6.0)
    }

    #[test]
    fn vector_product_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x00 = rng.gen_range(0, 100) as f64;
            let x01 = rng.gen_range(0, 100) as f64;
            let x02 = rng.gen_range(0, 100) as f64;
            let x10 = rng.gen_range(0, 100) as f64;
            let x11 = rng.gen_range(0, 100) as f64;
            let x12 = rng.gen_range(0, 100) as f64;
            let x20 = rng.gen_range(0, 100) as f64;
            let x21 = rng.gen_range(0, 100) as f64;
            let x22 = rng.gen_range(0, 100) as f64;

            let v0 = rng.gen_range(0, 100) as f64;
            let v1 = rng.gen_range(0, 100) as f64;
            let v2 = rng.gen_range(0, 100) as f64;

            let i = [x00, x01, x02, x10, x11, x12, x20, x21, x22];
            let matrix = Matrix3d::from_array(i);
            let vector = Vector3d::new(v0, v1, v2);

            let mv_product = matrix.vector_product(&vector);
            let manual_product = Vector3d::new(
                v0 * x00 + v1 * x10 + v2 * x20,
                v0 * x01 + v1 * x11 + v2 * x21,
                v0 * x02 + v1 * x12 + v2 * x22);
            assert_eq!(manual_product, mv_product);
        }
    }

    #[test]
    fn matrix_product_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x00 = rng.gen_range(0, 100) as f64;
            let x01 = rng.gen_range(0, 100) as f64;
            let x02 = rng.gen_range(0, 100) as f64;
            let x10 = rng.gen_range(0, 100) as f64;
            let x11 = rng.gen_range(0, 100) as f64;
            let x12 = rng.gen_range(0, 100) as f64;
            let x20 = rng.gen_range(0, 100) as f64;
            let x21 = rng.gen_range(0, 100) as f64;
            let x22 = rng.gen_range(0, 100) as f64;

            let y00 = rng.gen_range(0, 100) as f64;
            let y01 = rng.gen_range(0, 100) as f64;
            let y02 = rng.gen_range(0, 100) as f64;
            let y10 = rng.gen_range(0, 100) as f64;
            let y11 = rng.gen_range(0, 100) as f64;
            let y12 = rng.gen_range(0, 100) as f64;
            let y20 = rng.gen_range(0, 100) as f64;
            let y21 = rng.gen_range(0, 100) as f64;
            let y22 = rng.gen_range(0, 100) as f64;

            let mut i = [x00, x01, x02, x10, x11, x12, x20, x21, x22];
            let m1 = Matrix3d::from_array(i);
            i = [y00, y01, y02, y10, y11, y12, y20, y21, y22];
            let m2 = Matrix3d::from_array(i);
            let mv_product = m1.matrix_product(&m2);

            let mut manual_product = Matrix3d::new_zero();
            manual_product.inner[0] = x00 * y00 + x10 * y01 + x20 * y02;
            manual_product.inner[1] = x01 * y00 + x11 * y01 + x21 * y02;
            manual_product.inner[2] = x02 * y00 + x12 * y01 + x22 * y02;

            manual_product.inner[3] = x00 * y10 + x10 * y11 + x20 * y12;
            manual_product.inner[4] = x01 * y10 + x11 * y11 + x21 * y12;
            manual_product.inner[5] = x02 * y10 + x12 * y11 + x22 * y12;

            manual_product.inner[6] = x00 * y20 + x10 * y21 + x20 * y22;
            manual_product.inner[7] = x01 * y20 + x11 * y21 + x21 * y22;
            manual_product.inner[8] = x02 * y20 + x12 * y21 + x22 * y22;

            assert_eq!(manual_product, mv_product)
        }


        let i = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m1 = Matrix3d::from_array(i);
        println!("{:?}", m1.idx(2, 0));
    }

    #[test]
    fn matrix_scalar_product_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x00 = rng.gen_range(0, 100) as f64;
            let x01 = rng.gen_range(0, 100) as f64;
            let x02 = rng.gen_range(0, 100) as f64;
            let x10 = rng.gen_range(0, 100) as f64;
            let x11 = rng.gen_range(0, 100) as f64;
            let x12 = rng.gen_range(0, 100) as f64;
            let x20 = rng.gen_range(0, 100) as f64;
            let x21 = rng.gen_range(0, 100) as f64;
            let x22 = rng.gen_range(0, 100) as f64;

            let scalar = rng.gen_range(0, 100) as f64;
            let array_1 = [x00, x01, x02, x10, x11, x12, x20, x21, x22];
            let array_2 = [x00 * scalar, x01 * scalar, x02 * scalar, x10 * scalar, x11 * scalar, x12 * scalar, x20 * scalar, x21 * scalar, x22 * scalar];
            let m1 = Matrix3d::from_array(array_1);
            let m2 = Matrix3d::from_array(array_2);
            let scalar_product = m1.scalar_product(scalar);
            assert_eq!(scalar_product, m2)
        }
    }

    #[test]
    fn matrix_add_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x00 = rng.gen_range(0, 100) as f64;
            let x01 = rng.gen_range(0, 100) as f64;
            let x02 = rng.gen_range(0, 100) as f64;
            let x10 = rng.gen_range(0, 100) as f64;
            let x11 = rng.gen_range(0, 100) as f64;
            let x12 = rng.gen_range(0, 100) as f64;
            let x20 = rng.gen_range(0, 100) as f64;
            let x21 = rng.gen_range(0, 100) as f64;
            let x22 = rng.gen_range(0, 100) as f64;
            let array_1 = [x00, x01, x02, x10, x11, x12, x20, x21, x22];
            let array_2 = [x00 * 2.0, x01 * 2.0, x02 * 2.0, x10 * 2.0, x11 * 2.0, x12 * 2.0, x20 * 2.0, x21 * 2.0, x22 * 2.0];
            let m1 = Matrix3d::from_array(array_1);
            let m2 = Matrix3d::from_array(array_2);

            let sum = m1 + m1;
            assert_eq!(sum, m2)
        }
    }

    #[test]
    fn matrix_sub_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x00 = rng.gen_range(0, 100) as f64;
            let x01 = rng.gen_range(0, 100) as f64;
            let x02 = rng.gen_range(0, 100) as f64;
            let x10 = rng.gen_range(0, 100) as f64;
            let x11 = rng.gen_range(0, 100) as f64;
            let x12 = rng.gen_range(0, 100) as f64;
            let x20 = rng.gen_range(0, 100) as f64;
            let x21 = rng.gen_range(0, 100) as f64;
            let x22 = rng.gen_range(0, 100) as f64;
            let array_1 = [x00, x01, x02, x10, x11, x12, x20, x21, x22];
            let array_2 = [x00 * 2.0, x01 * 2.0, x02 * 2.0, x10 * 2.0, x11 * 2.0, x12 * 2.0, x20 * 2.0, x21 * 2.0, x22 * 2.0];
            let m1 = Matrix3d::from_array(array_1);
            let m2 = Matrix3d::from_array(array_2);

            let sub = m2 - m1;
            assert_eq!(sub, m1)
        }
    }

    #[test]
    fn linear_eq_test() {
        let m1 = Matrix3d::from_array([1.0, 2.0, 3.0, -3.0, -1.0, 6.0, 1.0, 2.0, 9.0]);
        let v = Vector3d::new(5.0, 5.0, 3.0);
        m1.linear_eq(v).unwrap();
    }

    #[test]
    fn swap_rows_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1 {
            let x00 = rng.gen_range(0, 100) as f64;
            let x01 = rng.gen_range(0, 100) as f64;
            let x02 = rng.gen_range(0, 100) as f64;
            let x10 = rng.gen_range(0, 100) as f64;
            let x11 = rng.gen_range(0, 100) as f64;
            let x12 = rng.gen_range(0, 100) as f64;
            let x20 = rng.gen_range(0, 100) as f64;
            let x21 = rng.gen_range(0, 100) as f64;
            let x22 = rng.gen_range(0, 100) as f64;
            let array_1 = [x00, x01, x02, x10, x11, x12, x20, x21, x22];
            let m1 = Matrix3d::from_array(array_1);
            let m2 = m1.swap_rows(0, 1);
            assert_ne!(m1, m2);
            println!("{}", m1);
            let m3 = Matrix3d::from_array([x01, x00, x02, x11, x10, x12, x21, x20, x22]);
            println!("{}", m3);
            assert_eq!(m3, m2);
        }
    }

    #[test]
    fn transpose_test() {
        let matrix1 = Matrix3d::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let matrix2 = Matrix3d::from_array([0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]);
        assert_eq!(matrix1.transpose(), matrix2);
        assert_eq!(matrix1.rows(), matrix2.columns());
    }

    #[test]
    fn determinant_test() {
        let m1 = Matrix3d::from_array([2.0, 4.0, 7.0, 3.0, 4.0, 9.0, 11.0, 5.0, 6.0]);
        assert_eq!(m1.det(), 79.0);
    }

    #[test]
    fn from_cols_test() {
        let col0 = [3.0, 1.5, 4.0];
        let col1 = [7.0, 2.5, 4.5];
        let col2 = [-2.5, 3.0, -1.0];

        let matrix1 = Matrix3d::from_cols(col0, col1, col2);
        let matrix2 = Matrix3d::from_array([3.0, 1.5, 4.0, 7.0, 2.5, 4.5, -2.5, 3.0, -1.0]);
        assert_eq!(matrix1, matrix2);
    }

    #[test]
    fn adj_matrix_test() {
        let matrix = Matrix3d::from_cols([0.0, 1.0, 2.0], [4.0, 3.0, 5.0], [6.0, 7.0, 8.0]);
        let adj = matrix.adj_matrix();
        let expected = Matrix3d::from_cols([-11.0, -2.0, 10.0], [6.0, -12.0, 6.0], [-1.0, 8.0, -4.0]);
        println!("{}", expected);
        assert_eq!(adj, expected)
    }

    #[test]
    fn inverse_test() {
        let matrix = Matrix3d::from_cols([5.0, 4.0, 3.0], [1.0, 8.0, 11.0], [2.0, 3.0, 4.0]);
        let inverse = matrix.inverse().unwrap();

        let expected = Matrix3d::from_cols([-1.0, -7.0, 20.0], [18.0, 14.0, -52.0], [-13.0, -7.0, 36.0])
            .scalar_product(1.0/matrix.det());

        assert_eq!(inverse, expected);
    }

    #[test]
    fn inverse_zero_det_test() {
        let matrix = Matrix3d::from_cols([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]);
        assert!(matrix.inverse().is_none())
    }

    #[test]
    fn determinant_rules_test() {}
}
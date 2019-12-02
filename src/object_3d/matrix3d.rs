use crate::object_3d::vector3d::Vector3d;
use crate::object_3d::affine3d::Affine3d;

///
/// Matrix represent as plain array in column major order
/// [ 0  3  6 ]
/// [ 1  4  7 ]
/// [ 2  5  8 ]
///
#[derive(Debug, PartialOrd, PartialEq, Copy, Clone)]
pub struct Matrix3d {
    inner: [f64; 9]
}

impl Matrix3d {
    ///Create 3x3 matrix filled zero.
    pub fn new_zero() -> Self {
        let inner = [0.0; 9];
        Matrix3d { inner }.to_owned()
    }

    ///Create 3x3 matrix filled zero from borrowed 9-length array
    pub fn from_borrowed(input: &[f64; 9]) -> Self {
        let inner: [f64; 9] = [input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8]];
        Self { inner }
    }

    ///Create 3x3 matrix filled zero from array
    pub fn from_array(input: [f64; 9]) -> Self {
        Self { inner: input }
    }

    ///Return value of index
    pub fn idx(&self, row: usize, col: usize) -> f64 {
        self.inner[row + 3 * col]
    }

    ///Return tuple-3 of rows
    pub fn columns(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let row0 = [self.inner[0], self.inner[1], self.inner[2]];
        let row1 = [self.inner[3], self.inner[4], self.inner[5]];
        let row2 = [self.inner[6], self.inner[7], self.inner[8]];
        (row0, row1, row2)
    }

    ///Return tuple-3 of columns
    pub fn rows(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let row0 = [self.inner[0], self.inner[3], self.inner[6]];
        let row1 = [self.inner[1], self.inner[4], self.inner[7]];
        let row2 = [self.inner[2], self.inner[5], self.inner[8]];
        (row0, row1, row2)
    }

    pub fn vector_product(&self, right:Vector3d) ->Vector3d{
        let x = self.inner[0] * right.x()+self.inner[3]*right.y()+self.inner[6]*right.z();
        let y = self.inner[1] * right.x()+self.inner[4]*right.y()+self.inner[7]*right.z();
        let z = self.inner[2] * right.x()+self.inner[5]*right.y()+self.inner[8]*right.z();
        Vector3d::new(x,y,z)
    }

    pub fn new_diagonal() -> Self {
        let inner = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        Matrix3d { inner }.to_owned()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix3d_new_zero_test() {
        let m = Matrix3d::new_zero();
        m.inner.iter().for_each(|x| assert_eq!(*x, 0.0))
    }

    #[test]
    fn matrix3d_new_diagonal_test() {
        let m = Matrix3d::new_zero();
        m.inner.iter().for_each(|x| assert_eq!(*x, 0.0))
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
        assert_eq!(m.rows().0, [1.0, 4.0, 7.0]);
    }

    #[test]
    fn idx_test() {
        let i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Matrix3d::from_array(i);
        assert_eq!(m.idx(2, 1), 6.0)
    }

    #[test]
    fn vector_product_test(){
        let i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let matrix = Matrix3d::from_array(i);
        let vector = Vector3d::new(2.0,3.0,6.0);
        let mv_product = matrix.vector_product(vector);
        println!("{:?}", mv_product);
    }
}
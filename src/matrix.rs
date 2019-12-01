use std::fmt::{Display, Formatter, Error};

pub struct Matrix {
    n: usize,
    m: usize,
    cells: Vec<Vec<f64>>,
}



impl Matrix {
    pub fn new_empty(n: usize, m: usize) -> Self {
        let cells: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; m]).collect();
        Self {
            n,
            m,
            cells,
        }
    }



    pub fn size(&self) -> [usize; 2] {
        [self.n, self.m]
    }

    pub fn from_cells(cells: Vec<Vec<f64>>) -> Self {
        let (n, m) = if cells.is_empty() || cells[0].is_empty() {
            (0, 0)
        } else {
            (cells.len(), cells[0].len())
        };
        Self {
            n,
            m,
            cells,
        }
    }

    pub fn try_add(m1: Matrix, m2: Matrix) -> Result<Matrix, String> {
        if m1.m != m2.m || m1.n != m2.n {
            Result::Err("Can't add".to_owned())
        } else {
            let result = m1.cells.iter()
                .enumerate()
                .map(|(row, vec)| vec.iter().enumerate()
                    .map(|(column, value)|
                        *value + m2.cells[row][column]).collect()).collect();
            Result::Ok(Matrix::from_cells(result))
        }
    }

    pub fn product(m1: Matrix, m2: Matrix) -> Result<Matrix, String> {
        if m1.m == m2.n {
            let mut result = Matrix::new_empty(m1.n, m2.m);
            let n = m1.n;
            let _m = m1.m;
            let k = m2.m;
            for i in 0..n {
                for j in 0..k {
                    let cell = m1.cells[i].iter().enumerate()
                        .fold(0.0, |acc, (number, value)| {
                            acc + (*value) * m2.cells[number][j]
                        });
                    result.cells[i][j] = cell;
                }
            }
            Ok(result)
        } else {
            Err(String::from("Can't product"))
        }
    }
}

impl Display for Matrix{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        let mut result = "".to_owned();
        for i in &self.cells {
            for j in i {
                result += &j.to_string();
                result += "  ";
            }
            result += "\n";
        }
        write!(f,"{}",result)

    }
}

#[cfg(test)]
mod test    {
    use super::*;

    #[test]
    fn new_empty_test() {
        let matrix = Matrix::new_empty(4, 5);
        for i in 0..4 {
            for j in 0..5 {
                assert_eq!(matrix.cells[i][j], 0.0);
            }
        }
    }

    #[test]
    fn size_test() {
        let n = 10;
        let m = 20;
        let matrix = Matrix::new_empty(n, m);
        assert_eq!(matrix.size(), [10, 20]);
    }

    #[test]
    fn from_cells_test() {
        let first = vec![1.0, 2.0, 3.0];
        let second = vec![4.0, 5.0, 6.0];
        let matrix = Matrix::from_cells(vec![first, second]);
        println!("{}", matrix.to_string());
        assert_eq!(matrix.size(), [2, 3]);
    }

    #[test]
    fn from_cells_zero_vec_test() {
        let first: Vec<f64> = vec![];
        let second: Vec<f64> = vec![];
        let matrix = Matrix::from_cells(vec![first, second]);
        assert_eq!(matrix.size(), [0, 0]);
        let matrix = Matrix::from_cells(vec![]);
        assert_eq!(matrix.size(), [0, 0]);
    }

    #[test]
    fn add_test() {
        let first = vec![1.0, 2.0, 3.0];
        let second = vec![4.0, 5.0, 6.0];
        let matrix1 = Matrix::from_cells(vec![first.clone(), second.clone()]);
        let matrix2 = Matrix::from_cells(vec![first.clone(), second.clone()]);
        println!("{}", Matrix::try_add(matrix1, matrix2).unwrap().to_string())
    }

    #[test]
    fn product_test() {
        let matrix1 = Matrix::from_cells(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]]);
        let matrix2 = Matrix::from_cells(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        println!("{}", Matrix::product(matrix1, matrix2).unwrap().to_string());
    }
}
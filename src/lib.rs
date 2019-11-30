mod matrix;
pub mod vector;
pub mod points;
pub mod line;
pub mod triangle;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let mut x = 1.0e30_f32;
        x *= 1.0e38_f32;
        x /= 1.0e38_f32;
        println!("{}", x);
    }
}




mod matrix;
mod vector;

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




#![feature(core_intrinsics)]

mod matrix;
pub mod affine;
pub mod line;
pub mod triangle;
pub mod object_3d;

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




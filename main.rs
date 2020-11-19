extern crate ndarray;
use ndarray::prelude::*;
//[dependencies]
//ndarray = { version = "0.13.0", features = ["blas"] }

fn LU_solve(L: Array2<f64>,U: Array2<f64>,b: Array1<f64>) -> Array1<f64> {
    
    let mut y = b.clone();
    for j in 0..L.len_of(Axis(0)) {
        for i in j+1..L.len_of(Axis(0)) {
            y[i] -= y[j] * L[[i,j]];
        }
    }

    let mut x = y.clone();

    for j in (0..U.len_of(Axis(0))).rev() {
        x[j] /= U[[j,j]];
        for i in 0..j {
            x[i] -= U[[i,j]] * x[j];
        }
    }

    return x;
}

fn LU_decomposition(A: Array2<f64>) -> (Array2<f64>,Array2<f64>){

    let mut L: Array2<f64> = Array2::eye(A.len_of(Axis(0)));
    let mut U: Array2<f64> = Array2::zeros((A.len_of(Axis(0)),A.len_of(Axis(0))));

    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    for i in 0..A.len_of(Axis(0)) {
        for j in 0..A.len_of(Axis(0)) {
            if i <= j {
                sum1 = 0.0;
                for m in 0..i {
                    sum1 += L[[i,m]] * U[[m,j]];
                }
                U[[i,j]] = A[[i,j]] - sum1;
            }
            else if i > j {
                sum2 = 0.0;
                for n in 0..j {
                    sum2 += L[[i,n]] * U[[n,j]];
                }
                L[[i,j]] = (A[[i,j]] - sum2) / U[[j,j]];
            }
        }
    }

    return (L,U);
}

fn inverse_matrix(matrix: Array2<f64> ) -> Array2<f64> {

    let X = LU_decomposition(matrix.clone());
    let L = X.0;
    let U = X.1;

    let I_vec: Vec<Array1<f64>> = Vec::new();
    let mut b: Array1<f64> = Array1::zeros(matrix.len_of(Axis(0)));
    let mut r = b.clone();

    let mut inv_A = matrix.clone();

    for i in 0..matrix.len_of(Axis(0)) {
        b[i] = 1.0;
        r = LU_solve(L.clone(),U.clone(),b.clone());
        for j in 0..matrix.len_of(Axis(0)) {
            inv_A[[j,i]] = r[j];
        }
        b[i] = 0.0;
    }

    return inv_A;

}

fn main() {
    let A: Array2<f64> = array![[2.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]];

    let inv_A = inverse_matrix(A.clone());
    println!("{}",A);
    println!("{}",inv_A);
    //println!("{}",A[[0,1]]);
    //println!("{}",inv_A[[1,2]]);
}

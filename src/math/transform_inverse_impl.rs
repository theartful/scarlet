use crate::math::scalar::GFloat;

#[inline(always)]
pub fn inverse_impl<T: GFloat>(data: &[[T; 4]; 4]) -> [[T; 4]; 4] {
    inverse_impl_adjugate(&data)
}

#[inline(always)]
fn inverse_impl_adjugate<T: GFloat>(data: &[[T; 4]; 4]) -> [[T; 4]; 4] {
    // compute determinants of all sub matrices
    // 00   01  02  03
    // 10   11  12  13
    // 20   21  22  23
    // 30   31  32  33

    let d00 = data[1][1] * (data[2][2] * data[3][3] - data[2][3] * data[3][2])
        - data[1][2] * (data[2][1] * data[3][3] - data[2][3] * data[3][1])
        + data[1][3] * (data[2][1] * data[3][2] - data[2][2] * data[3][1]);

    let d01 = data[1][0] * (data[2][2] * data[3][3] - data[2][3] * data[3][2])
        - data[1][2] * (data[2][0] * data[3][3] - data[2][3] * data[3][0])
        + data[1][3] * (data[2][0] * data[3][2] - data[2][2] * data[3][0]);

    let d02 = data[1][0] * (data[2][1] * data[3][3] - data[2][3] * data[3][1])
        - data[1][1] * (data[2][0] * data[3][3] - data[2][3] * data[3][0])
        + data[1][3] * (data[2][0] * data[3][1] - data[2][1] * data[3][0]);

    let d03 = data[1][0] * (data[2][1] * data[3][2] - data[2][2] * data[3][1])
        - data[1][1] * (data[2][0] * data[3][2] - data[2][2] * data[3][0])
        + data[1][2] * (data[2][0] * data[3][1] - data[2][1] * data[3][0]);

    let d10 = data[0][1] * (data[2][2] * data[3][3] - data[2][3] * data[3][2])
        - data[0][2] * (data[2][1] * data[3][3] - data[2][3] * data[3][1])
        + data[0][3] * (data[2][1] * data[3][2] - data[2][2] * data[3][1]);

    let d11 = data[0][0] * (data[2][2] * data[3][3] - data[2][3] * data[3][2])
        - data[0][2] * (data[2][0] * data[3][3] - data[2][3] * data[3][0])
        + data[0][3] * (data[2][0] * data[3][2] - data[2][2] * data[3][0]);

    let d12 = data[0][0] * (data[2][1] * data[3][3] - data[2][3] * data[3][1])
        - data[0][1] * (data[2][0] * data[3][3] - data[2][3] * data[3][0])
        + data[0][3] * (data[2][0] * data[3][1] - data[2][1] * data[3][0]);

    let d13 = data[0][0] * (data[2][1] * data[3][2] - data[2][2] * data[3][1])
        - data[0][1] * (data[2][0] * data[3][2] - data[2][2] * data[3][0])
        + data[0][2] * (data[2][0] * data[3][1] - data[2][1] * data[3][0]);

    let d20 = data[0][1] * (data[1][2] * data[3][3] - data[1][3] * data[3][2])
        - data[0][2] * (data[1][1] * data[3][3] - data[1][3] * data[3][1])
        + data[0][3] * (data[1][1] * data[3][2] - data[1][2] * data[3][1]);

    let d21 = data[0][0] * (data[1][2] * data[3][3] - data[1][3] * data[3][2])
        - data[0][2] * (data[1][0] * data[3][3] - data[1][3] * data[3][0])
        + data[0][3] * (data[1][0] * data[3][2] - data[1][2] * data[3][0]);

    let d22 = data[0][0] * (data[1][1] * data[3][3] - data[1][3] * data[3][1])
        - data[0][1] * (data[1][0] * data[3][3] - data[1][3] * data[3][0])
        + data[0][3] * (data[1][0] * data[3][1] - data[1][1] * data[3][0]);

    let d23 = data[0][0] * (data[1][1] * data[3][2] - data[1][2] * data[3][1])
        - data[0][1] * (data[1][0] * data[3][2] - data[1][2] * data[3][0])
        + data[0][2] * (data[1][0] * data[3][1] - data[1][1] * data[3][0]);

    let d30 = data[0][1] * (data[1][2] * data[2][3] - data[1][3] * data[2][2])
        - data[0][2] * (data[1][1] * data[2][3] - data[1][3] * data[2][1])
        + data[0][3] * (data[1][1] * data[2][2] - data[1][2] * data[2][1]);

    let d31 = data[0][0] * (data[1][2] * data[2][3] - data[1][3] * data[2][2])
        - data[0][2] * (data[1][0] * data[2][3] - data[1][3] * data[2][0])
        + data[0][3] * (data[1][0] * data[2][2] - data[1][2] * data[2][0]);

    let d32 = data[0][0] * (data[1][1] * data[2][3] - data[1][3] * data[2][1])
        - data[0][1] * (data[1][0] * data[2][3] - data[1][3] * data[2][0])
        + data[0][3] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);

    let d33 = data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])
        - data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
        + data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);

    // copmute determinant of the 4x4 matrix
    let det = data[0][0] * d00 - data[0][1] * d01 + data[0][2] * d02 - data[0][3] * d03;
    let det_inv = T::one() / det;

    // inverse = det_inv * adjugate
    [
        [d00 * det_inv, -d10 * det_inv, d20 * det_inv, -d30 * det_inv],
        [-d01 * det_inv, d11 * det_inv, -d21 * det_inv, d31 * det_inv],
        [d02 * det_inv, -d12 * det_inv, d22 * det_inv, -d32 * det_inv],
        [-d03 * det_inv, d13 * det_inv, -d23 * det_inv, d33 * det_inv],
    ]
}

// TODO: implement gaussian elimination and compare performance
// fn inverse_impl_gaussian_elimination<T: GFloat>(_data: &[[T; 4]; 4]) -> [[T; 4]; 4] {
//     std::unimplemented!();
// }

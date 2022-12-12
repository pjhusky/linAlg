

// for matrix: inner array stores the rows vectors, outer array says how many row vectors we have
// vector = matrix * vector
template<typename val_T, std::size_t numColumns_T, std::size_t numRows_T>
static linAlg::vec_t<val_T, numRows_T> operator*( const linAlg::mat_t<val_T, numRows_T, numColumns_T >& matrix,
    const linAlg::vec_t<val_T, numColumns_T>& vector ) {
    linAlg::vec_t<val_T, numRows_T> result;
    // result[0] = linAlg::dot( matrix[0], vector );
    // result[1] = linAlg::dot( matrix[1], vector );
    // result[2] = linAlg::dot( matrix[2], vector );
    for (std::size_t row = 0; row < numRows_T; row++) {
        result[row] = linAlg::dot( matrix[row], vector );
    }
    return result;
}

// matrix = matrix * matrix
// inner array stores the rows vectors, outer array says how many row vectors we have
// "inner" index - the one closer to the var name - accesses the row, the outer "last" indexing parentheses index the column within that row
template<typename val_T, std::size_t rowsL_T, std::size_t colsL_rowsR_T, std::size_t colsR_T>
static linAlg::mat_t<val_T, rowsL_T, colsR_T > operator* ( const linAlg::mat_t< val_T, rowsL_T, colsL_rowsR_T >& matrixLhs,
    const linAlg::mat_t< val_T, colsL_rowsR_T, colsR_T >& matrixRhs ) {
    linAlg::mat_t< val_T, rowsL_T, colsR_T > result;
    for (std::size_t col = 0; col < colsR_T; col++) { // left columns
        for (std::size_t row = 0; row < rowsL_T; row++) { // right rows
            val_T accum{ 0 };
            for (std::size_t k = 0; k < colsL_rowsR_T; k++) { // dot of left row (over columns) and right columns ( over rows )
                accum += matrixLhs[row][k] * matrixRhs[k][col];
            }
            result[row][col] = accum;
        }
    }
    return result;
}

// e.g., in 3D, non-perspective transformations can be represented by 3x4 matrices => treat them as 3x3 rotations and fix up translational part
template<typename val_T, std::size_t rowsL_T>
static linAlg::mat_t<val_T, rowsL_T, rowsL_T + 1 > operator* ( const linAlg::mat_t< val_T, rowsL_T, rowsL_T + 1 >& matrixLhs,
    const linAlg::mat_t< val_T, rowsL_T, rowsL_T + 1 >& matrixRhs ) {
    constexpr std::size_t colsR_T = rowsL_T + 1;
    constexpr std::size_t squareDim = rowsL_T;
    linAlg::mat_t< val_T, rowsL_T, colsR_T > result;
    for (std::size_t col = 0; col < squareDim; col++) { // left columns => only rowsL_T number of columns to iterate over
        for (std::size_t row = 0; row < squareDim; row++) { // right rows
            val_T accum{ 0 };
            for (std::size_t k = 0; k < squareDim; k++) { // dot of left row (over columns) and right columns ( over rows )
                accum += matrixLhs[row][k] * matrixRhs[k][col];
            }
            result[row][col] = accum;
        }
    }
    // fix-up for last column (translations)
    constexpr std::size_t lastColumnIdx = colsR_T - 1;

    for (std::size_t row = 0; row < rowsL_T; row++) {
        val_T accum{ 0 };
        for (std::size_t k = 0; k < squareDim; k++) {
            accum += matrixLhs[row][k] * matrixRhs[k][lastColumnIdx];
        }
        result[row][lastColumnIdx] = accum + matrixLhs[row][lastColumnIdx];
    }


    return result;
}

template<typename val_T, std::size_t numCoords_T>
static bool operator==( const linAlg::vec_t<val_T, numCoords_T>& vectorLhs,
    const linAlg::vec_t<val_T, numCoords_T>& vectorRhs ) {
    for (std::size_t coord = 0; coord < numCoords_T; coord++) {
        if (vectorLhs[coord] != vectorRhs[coord]) { return false; }
    }
    return true;
}

template<typename val_T, std::size_t rows_T, std::size_t cols_T>
static bool operator== ( const linAlg::mat_t< val_T, rows_T, cols_T >& matrixLhs,
    const linAlg::mat_t< val_T, rows_T, cols_T >& matrixRhs ) {

    for (std::size_t col = 0; col < cols_T; col++) {
        for (std::size_t row = 0; row < rows_T; row++) {
            if (matrixLhs[row][col] != matrixRhs[row][col]) { return false; }
        }
    }

    return true;
}

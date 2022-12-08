#include "linAlg.h"

#ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
#endif
#include <math.h>
// #include <cassert>
#include <string.h> // for memcpy & memset

void linAlg::cast( vec2_t& dst, const vec3_t& src ) {
    memcpy( dst.data(), src.data(), dst.size() * sizeof( float ) );
}
void linAlg::cast( vec2_t& dst, const vec4_t& src ) {
    memcpy( dst.data(), src.data(), dst.size() * sizeof( float ) );
}
void linAlg::cast( vec3_t& dst, const vec4_t& src ) {
    memcpy( dst.data(), src.data(), dst.size() * sizeof( float ) );
}

void linAlg::stripRowAndColumn( mat2_t& dst, const mat3_t&src, const int32_t row, const int32_t column ) {
    int32_t dstRowIdx = 0;
    for ( size_t rowIdx = 0; rowIdx < 3; rowIdx++ ) {
        if ( rowIdx == row - 1 ) { continue; }
        int32_t dstColIdx = 0;
        for ( size_t colIdx = 0; colIdx < 3; colIdx++ ) {
            if ( colIdx == column - 1 ) { continue; }
            dst[ dstRowIdx ][ dstColIdx ] = src[ rowIdx ][ colIdx ];
            dstColIdx++;
        }
        dstRowIdx++;
    }
}

void linAlg::stripRowAndColumn( mat3_t& dst, const mat4_t&src, const int32_t row, const int32_t column ) {
    int32_t dstRowIdx = 0;
    for ( size_t rowIdx = 0; rowIdx < 4; rowIdx++ ) {
        if ( rowIdx == row - 1 ) { continue; }
        int32_t dstColIdx = 0;
        for ( size_t colIdx = 0; colIdx < 4; colIdx++ ) {
            if ( colIdx == column - 1 ) { continue; }
            dst[ dstRowIdx ][ dstColIdx ] = src[ rowIdx ][ colIdx ];
            dstColIdx++;
        }
        dstRowIdx++;
    }
}

void linAlg::cross( vec3_t& dst, const vec3_t& lhs, const vec3_t& rhs ) {
    dst[ 0 ] =  lhs[ 1 ] * rhs[ 2 ] - lhs[ 2 ] * rhs[ 1 ];
    dst[ 1 ] = -lhs[ 0 ] * rhs[ 2 ] + lhs[ 2 ] * rhs[ 0 ];
    dst[ 2 ] =  lhs[ 0 ] * rhs[ 1 ] - lhs[ 1 ] * rhs[ 0 ];
}

float linAlg::determinant( const mat2_t& mat ) {
    const linAlg::vec2_t& row1 = mat[ 0 ];
    const linAlg::vec2_t& row2 = mat[ 1 ];
    return row1[ 0 ] * row2[ 1 ] - row2[ 0 ] * row1[ 1 ];
}

float linAlg::determinant( const mat3_t& mat ) {
    const vec3_t& row1 = mat[ 0 ];
    mat2_t subMat_11, subMat_12, subMat_13;
    stripRowAndColumn( subMat_11, mat, 1, 1 );
    stripRowAndColumn( subMat_12, mat, 1, 2 );
    stripRowAndColumn( subMat_13, mat, 1, 3 );

    return row1[ 0 ] * determinant( subMat_11 ) - row1[ 1 ] * determinant( subMat_12 ) + row1[ 2 ] * determinant( subMat_13 );
}

float linAlg::determinant( const mat4_t& mat ) {
    const vec4_t& row1 = mat[ 0 ];
    mat3_t subMat_11, subMat_12, subMat_13, subMat_14;
    stripRowAndColumn( subMat_11, mat, 1, 1 );
    stripRowAndColumn( subMat_12, mat, 1, 2 );
    stripRowAndColumn( subMat_13, mat, 1, 3 );
    stripRowAndColumn( subMat_14, mat, 1, 4 );

    return 
        row1[ 0 ] * determinant( subMat_11 ) 
      - row1[ 1 ] * determinant( subMat_12 ) 
      + row1[ 2 ] * determinant( subMat_13 )
      - row1[ 3 ] * determinant( subMat_14 );
}

//void linAlg::transpose( mat2_t& dst, const mat2_t& src ) {
//    for (size_t y = 0; y < 2; y++) {
//        for (size_t x = 0; x < 2; x++) {
//            dst[y][x] = src[x][y];
//        }
//    }
//}

void linAlg::inverse( mat2_t& dst, const mat2_t& src ) {
    const float oneOverDeterminant = 1.0f / determinant( src );
    dst[ 0 ][ 0 ] =  oneOverDeterminant * src[ 1 ][ 1 ];
    dst[ 0 ][ 1 ] = -oneOverDeterminant * src[ 0 ][ 1 ]; // remember to transpose
    dst[ 1 ][ 0 ] = -oneOverDeterminant * src[ 1 ][ 0 ]; // remember to transpose
    dst[ 1 ][ 1 ] =  oneOverDeterminant * src[ 0 ][ 0 ];
}
void linAlg::inverse( mat3_t& dst, const mat3_t& src ) {
    mat2_t sub_11, sub_12, sub_13;
    stripRowAndColumn( sub_11, src, 1, 1 );
    stripRowAndColumn( sub_12, src, 1, 2 );
    stripRowAndColumn( sub_13, src, 1, 3 );
    
    mat2_t sub_21, sub_22, sub_23;
    stripRowAndColumn( sub_21, src, 2, 1 );
    stripRowAndColumn( sub_22, src, 2, 2 );
    stripRowAndColumn( sub_23, src, 2, 3 );
    
    mat2_t sub_31, sub_32, sub_33;
    stripRowAndColumn( sub_31, src, 3, 1 );
    stripRowAndColumn( sub_32, src, 3, 2 );
    stripRowAndColumn( sub_33, src, 3, 3 );

    const float det_11 = determinant( sub_11 );
    const float det_12 = determinant( sub_12 );
    const float det_13 = determinant( sub_13 );

    const float det_21 = determinant( sub_21 );
    const float det_22 = determinant( sub_22 );
    const float det_23 = determinant( sub_23 );

    const float det_31 = determinant( sub_31 );
    const float det_32 = determinant( sub_32 );
    const float det_33 = determinant( sub_33 );

    // assert( ( src[ 0 ][ 0 ] * det_11 - src[ 0 ][ 1 ] * det_12 + src[ 0 ][ 2 ] * det_13 ) == determinant( src ) );

    const float oneOverDeterminant = 1.0f / ( src[ 0 ][ 0 ] * det_11 - src[ 0 ][ 1 ] * det_12 + src[ 0 ][ 2 ] * det_13 );

    dst[ 0 ][ 0 ] =   oneOverDeterminant * det_11;
    dst[ 0 ][ 1 ] =  -oneOverDeterminant * det_21;
    dst[ 0 ][ 2 ] =   oneOverDeterminant * det_31;

    dst[ 1 ][ 0 ] =  -oneOverDeterminant * det_12;
    dst[ 1 ][ 1 ] =   oneOverDeterminant * det_22;
    dst[ 1 ][ 2 ] =  -oneOverDeterminant * det_32;

    dst[ 2 ][ 0 ] =   oneOverDeterminant * det_13;
    dst[ 2 ][ 1 ] =  -oneOverDeterminant * det_23;
    dst[ 2 ][ 2 ] =   oneOverDeterminant * det_33;
}

void linAlg::inverse( mat4_t& dst, const mat4_t& src ) {
    mat3_t sub_11, sub_12, sub_13, sub_14;
    stripRowAndColumn( sub_11, src, 1, 1 );
    stripRowAndColumn( sub_12, src, 1, 2 );
    stripRowAndColumn( sub_13, src, 1, 3 );
    stripRowAndColumn( sub_14, src, 1, 4 );
    
    mat3_t sub_21, sub_22, sub_23, sub_24;
    stripRowAndColumn( sub_21, src, 2, 1 );
    stripRowAndColumn( sub_22, src, 2, 2 );
    stripRowAndColumn( sub_23, src, 2, 3 );
    stripRowAndColumn( sub_24, src, 2, 4 );
    
    mat3_t sub_31, sub_32, sub_33, sub_34;
    stripRowAndColumn( sub_31, src, 3, 1 );
    stripRowAndColumn( sub_32, src, 3, 2 );
    stripRowAndColumn( sub_33, src, 3, 3 );
    stripRowAndColumn( sub_34, src, 3, 4 );

    mat3_t sub_41, sub_42, sub_43, sub_44;
    stripRowAndColumn( sub_41, src, 4, 1 );
    stripRowAndColumn( sub_42, src, 4, 2 );
    stripRowAndColumn( sub_43, src, 4, 3 );
    stripRowAndColumn( sub_44, src, 4, 4 );

    const float det_11 = determinant( sub_11 );
    const float det_12 = determinant( sub_12 );
    const float det_13 = determinant( sub_13 );
    const float det_14 = determinant( sub_14 );

    const float det_21 = determinant( sub_21 );
    const float det_22 = determinant( sub_22 );
    const float det_23 = determinant( sub_23 );
    const float det_24 = determinant( sub_24 );

    const float det_31 = determinant( sub_31 );
    const float det_32 = determinant( sub_32 );
    const float det_33 = determinant( sub_33 );
    const float det_34 = determinant( sub_34 );

    const float det_41 = determinant( sub_41 );
    const float det_42 = determinant( sub_42 );
    const float det_43 = determinant( sub_43 );
    const float det_44 = determinant( sub_44 );

    // assert( ( src[ 0 ][ 0 ] * det_11 - src[ 0 ][ 1 ] * det_12 + src[ 0 ][ 2 ] * det_13 - src[ 0 ][ 3 ] * det_14 ) == determinant( src ) );

    const float det = src[ 0 ][ 0 ] * det_11 - src[ 0 ][ 1 ] * det_12 + src[ 0 ][ 2 ] * det_13 - src[ 0 ][ 3 ] * det_14;
    const float oneOverDeterminant = 1.0f / ( det );

    dst[ 0 ][ 0 ] =  oneOverDeterminant * det_11;
    dst[ 0 ][ 1 ] = -oneOverDeterminant * det_21;
    dst[ 0 ][ 2 ] =  oneOverDeterminant * det_31;
    dst[ 0 ][ 3 ] = -oneOverDeterminant * det_41;

    dst[ 1 ][ 0 ] = -oneOverDeterminant * det_12;
    dst[ 1 ][ 1 ] =  oneOverDeterminant * det_22;
    dst[ 1 ][ 2 ] = -oneOverDeterminant * det_32;
    dst[ 1 ][ 3 ] =  oneOverDeterminant * det_42;

    dst[ 2 ][ 0 ] =  oneOverDeterminant * det_13;
    dst[ 2 ][ 1 ] = -oneOverDeterminant * det_23;
    dst[ 2 ][ 2 ] =  oneOverDeterminant * det_33;
    dst[ 2 ][ 3 ] = -oneOverDeterminant * det_43;

    dst[ 3 ][ 0 ] = -oneOverDeterminant * det_14;
    dst[ 3 ][ 1 ] =  oneOverDeterminant * det_24;
    dst[ 3 ][ 2 ] = -oneOverDeterminant * det_34;
    dst[ 3 ][ 3 ] =  oneOverDeterminant * det_44;
}

void linAlg::loadIdentityMatrix( mat3_t& matrix ) {
    matrix[ 0 ] = linAlg::vec3_t{ 1.0f, 0.0f, 0.0f };
    matrix[ 1 ] = linAlg::vec3_t{ 0.0f, 1.0f, 0.0f };
    matrix[ 2 ] = linAlg::vec3_t{ 0.0f, 0.0f, 1.0f };
}

void linAlg::loadIdentityMatrix( mat3x4_t& matrix ) {
    matrix[0] = linAlg::vec4_t{ 1.0f, 0.0f, 0.0f, 0.0f };
    matrix[1] = linAlg::vec4_t{ 0.0f, 1.0f, 0.0f, 0.0f };
    matrix[2] = linAlg::vec4_t{ 0.0f, 0.0f, 1.0f, 0.0f };
}

void linAlg::loadIdentityMatrix( mat4_t& matrix ) {
    matrix[ 0 ] = linAlg::vec4_t{ 1.0f, 0.0f, 0.0f, 0.0f };
    matrix[ 1 ] = linAlg::vec4_t{ 0.0f, 1.0f, 0.0f, 0.0f };
    matrix[ 2 ] = linAlg::vec4_t{ 0.0f, 0.0f, 1.0f, 0.0f };
    matrix[ 3 ] = linAlg::vec4_t{ 0.0f, 0.0f, 0.0f, 1.0f };
}

void linAlg::loadTranslationMatrix( mat3_t& matrix, const vec2_t& translationVec ) {
    matrix[ 0 ] = linAlg::vec3_t{ 1.0f, 0.0f, translationVec[ 0 ] };
    matrix[ 1 ] = linAlg::vec3_t{ 0.0f, 1.0f, translationVec[ 1 ] };
    matrix[ 2 ] = linAlg::vec3_t{ 0.0f, 0.0f, 1.0f };
}

void linAlg::loadTranslationMatrix( mat3x4_t& matrix, const vec3_t& translationVec ) {
    loadIdentityMatrix( matrix );
    matrix[0][3] = translationVec[0];
    matrix[1][3] = translationVec[1];
    matrix[2][3] = translationVec[2];
}

void linAlg::loadRotationXMatrix( mat3x4_t& matrix, const float radiantsX ) {
    loadIdentityMatrix( matrix );
    float *const pMatrix = getMatrixPtr( matrix );
    const float cX = cos( radiantsX );
    const float sX = sin( radiantsX );
    pMatrix[  5 ] =  cX;
    pMatrix[  6 ] = -sX;
    pMatrix[  9 ] =  sX;
    pMatrix[ 10 ] =  cX;
}

void linAlg::loadRotationYMatrix( mat3x4_t& matrix, const float radiantsY ) {
    loadIdentityMatrix( matrix );
    float *const pMatrix = getMatrixPtr( matrix );
    const float cY = cos( radiantsY );
    const float sY = sin( radiantsY );
    pMatrix[  0 ] =  cY;
    pMatrix[  2 ] =  sY;
    pMatrix[  8 ] = -sY;
    pMatrix[ 10 ] =  cY;
}

void linAlg::loadRotationZMatrix( mat2_t& matrix, const float radiantsZ ) {
    float *const pMatrix = getMatrixPtr( matrix );
    const float cZ = cos( radiantsZ );
    const float sZ = sin( radiantsZ );
    pMatrix[  0 ] =  cZ;
    pMatrix[  1 ] =  -sZ;
    pMatrix[  2 ] =  sZ;
    pMatrix[  3 ] =  cZ;
}

void linAlg::loadRotationZMatrix( mat3x4_t& matrix, const float radiantsZ ) {
    loadIdentityMatrix( matrix );
    float *const pMatrix = getMatrixPtr( matrix );
    const float cZ = cos( radiantsZ );
    const float sZ = sin( radiantsZ );
    pMatrix[  0 ] =  cZ;
    pMatrix[  1 ] = -sZ;
    pMatrix[  4 ] =  sZ;
    pMatrix[  5 ] =  cZ;
}

void linAlg::loadRotationAroundAxis( linAlg::mat3x4_t& rotMat, const linAlg::vec3_t& axis, const float angleRad ) {
    const float ux = axis[0];
    const float ux2 = ux * ux;
    const float uy = axis[1];
    const float uy2 = uy * uy;
    const float uz = axis[2];
    const float uz2 = uz * uz;
    const float cphi = cosf( angleRad ); // should be equal to cosMousePtDirs
    const float sphi = sinf( angleRad );
    rotMat[0] = linAlg::vec4_t{ ux2 + (1.0f - ux2) * cphi,            ux * uy * (1.0f - cphi) - uz * sphi,  ux * uz * (1.0f - cphi) + uy * sphi,  0.0f }; // row 0
    rotMat[1] = linAlg::vec4_t{ uy * ux * (1.0f - cphi) + uz * sphi,  uy2 + (1.0f - uy2) * cphi,            uy * uz * (1.0f - cphi) - ux * sphi,  0.0f }; // row 1
    rotMat[2] = linAlg::vec4_t{ uz * ux * (1.0f - cphi) - uy * sphi,  uz * uy * (1.0f - cphi) + ux * sphi,  uz2 + (1.0f - uz2) * cphi,            0.0f }; // row 2
}

void linAlg::loadScaleMatrix( mat2_t& matrix, const vec2_t& scaleVec ) {
    matrix[ 0 ] = linAlg::vec2_t{ scaleVec[ 0 ], 0.0f };
    matrix[ 1 ] = linAlg::vec2_t{ 0.0f, scaleVec[ 1 ] };
}

void linAlg::loadScaleMatrix( mat3_t& matrix, const vec3_t& scaleVec ) {
    matrix[ 0 ] = linAlg::vec3_t{ scaleVec[ 0 ], 0.0f, 0.0f };
    matrix[ 1 ] = linAlg::vec3_t{ 0.0f, scaleVec[ 1 ], 0.0f };
    matrix[ 2 ] = linAlg::vec3_t{ 0.0f, 0.0f, scaleVec[ 2 ] };
}

void linAlg::loadScaleMatrix( mat3x4_t& matrix, const vec3_t& scaleVec ) {
    float *const pMatrix = getMatrixPtr( matrix );
    std::fill( pMatrix, pMatrix + 12, 0.0f );
    pMatrix[  0 ] = scaleVec[ 0 ];
    pMatrix[  5 ] = scaleVec[ 1 ];
    pMatrix[ 10 ] = scaleVec[ 2 ];
}

void linAlg::multMatrix( mat2_t& result, const mat2_t& left, const mat2_t& right ) {
    for ( size_t row = 0; row < 2; row++ ) {
        for ( size_t col = 0; col < 2; col++ ) {
            result[ row ][ col ] = linAlg::dot( left[ row ], linAlg::vec2_t{ right[ 0 ][ col ], right[ 1 ][ col ] } );
        }
    }
}

void linAlg::multMatrix( mat3_t& result, const mat3_t& left, const mat3_t& right ) {
    for ( size_t row = 0; row < 3; row++ ) {
        for ( size_t col = 0; col < 3; col++ ) {
            result[ row ][ col ] = linAlg::dot( left[ row ], linAlg::vec3_t{ right[ 0 ][ col ], right[ 1 ][ col ], right[ 2 ][ col ] } );
        }
    }
}

void linAlg::multMatrix( mat3x4_t& result, const mat3x4_t& left, const mat3x4_t& right ) {
    float *const pM = getMatrixPtr( result );
    const float *const pL = getMatrixPtr( left );
    const float *const pR = getMatrixPtr( right );

    pM[  0 ] = pL[ 0 ] * pR[ 0 ] + pL[ 1 ] * pR[ 4 ] + pL[  2 ] * pR[  8 ];
    pM[  1 ] = pL[ 0 ] * pR[ 1 ] + pL[ 1 ] * pR[ 5 ] + pL[  2 ] * pR[  9 ];
    pM[  2 ] = pL[ 0 ] * pR[ 2 ] + pL[ 1 ] * pR[ 6 ] + pL[  2 ] * pR[ 10 ];
    pM[  3 ] = pL[ 0 ] * pR[ 3 ] + pL[ 1 ] * pR[ 7 ] + pL[  2 ] * pR[ 11 ] + pL[  3 ];

    pM[  4 ] = pL[ 4 ] * pR[ 0 ] + pL[ 5 ] * pR[ 4 ] + pL[  6 ] * pR[  8 ];
    pM[  5 ] = pL[ 4 ] * pR[ 1 ] + pL[ 5 ] * pR[ 5 ] + pL[  6 ] * pR[  9 ];
    pM[  6 ] = pL[ 4 ] * pR[ 2 ] + pL[ 5 ] * pR[ 6 ] + pL[  6 ] * pR[ 10 ];
    pM[  7 ] = pL[ 4 ] * pR[ 3 ] + pL[ 5 ] * pR[ 7 ] + pL[  6 ] * pR[ 11 ] + pL[  7 ];

    pM[  8 ] = pL[ 8 ] * pR[ 0 ] + pL[ 9 ] * pR[ 4 ] + pL[ 10 ] * pR[  8 ];
    pM[  9 ] = pL[ 8 ] * pR[ 1 ] + pL[ 9 ] * pR[ 5 ] + pL[ 10 ] * pR[  9 ];
    pM[ 10 ] = pL[ 8 ] * pR[ 2 ] + pL[ 9 ] * pR[ 6 ] + pL[ 10 ] * pR[ 10 ];
    pM[ 11 ] = pL[ 8 ] * pR[ 3 ] + pL[ 9 ] * pR[ 7 ] + pL[ 10 ] * pR[ 11 ] + pL[ 11 ];
}

void linAlg::applyTransformation(
    const mat3_t& transformationMatrix,
    vec3_t* const pVertices,
    const size_t numVertices ) {

    const float* const pM = getMatrixPtr( transformationMatrix );

    for (size_t i = 0; i < numVertices; i++) {
        const auto transformVertex = pVertices[i]; // making a copy is important!
        pVertices[i][0] = pM[0] * transformVertex[0] + pM[1] * transformVertex[1] + pM[2] * transformVertex[2];
        pVertices[i][1] = pM[3] * transformVertex[0] + pM[4] * transformVertex[1] + pM[5] * transformVertex[2];
        pVertices[i][2] = pM[6] * transformVertex[0] + pM[7] * transformVertex[1] + pM[8] * transformVertex[2];
    }
}

void linAlg::applyTransformation( 
    const mat3x4_t& transformationMatrix, 
    vec3_t *const pVertices, 
    const size_t numVertices ) {

    const float *const pM = getMatrixPtr( transformationMatrix );

    for( size_t i = 0; i < numVertices; i++ ) {
        const auto transformVertex = pVertices[ i ]; // making a copy is important!
        pVertices[ i ][ 0 ] = pM[ 0 ] * transformVertex[ 0 ] + pM[ 1 ] * transformVertex[ 1 ] + pM[  2 ] * transformVertex[ 2 ] + pM[  3 ];
        pVertices[ i ][ 1 ] = pM[ 4 ] * transformVertex[ 0 ] + pM[ 5 ] * transformVertex[ 1 ] + pM[  6 ] * transformVertex[ 2 ] + pM[  7 ];
        pVertices[ i ][ 2 ] = pM[ 8 ] * transformVertex[ 0 ] + pM[ 9 ] * transformVertex[ 1 ] + pM[ 10 ] * transformVertex[ 2 ] + pM[ 11 ];
    }
}

void linAlg::applyTransformation( 
    const mat4_t& transformationMatrix, 
    vec4_t *const pVertices, 
    const size_t numVertices ) {

    const float *const pM = getMatrixPtr( transformationMatrix );

    for( size_t i = 0; i < numVertices; i++ ) {
        const auto transformVertex = pVertices[ i ]; // making a copy is important!
        pVertices[ i ][ 0 ] = pM[  0 ] * transformVertex[ 0 ] + pM[  1 ] * transformVertex[ 1 ] + pM[  2 ] * transformVertex[ 2 ] + pM[  3 ] * transformVertex[ 3 ];
        pVertices[ i ][ 1 ] = pM[  4 ] * transformVertex[ 0 ] + pM[  5 ] * transformVertex[ 1 ] + pM[  6 ] * transformVertex[ 2 ] + pM[  7 ] * transformVertex[ 3 ];
        pVertices[ i ][ 2 ] = pM[  8 ] * transformVertex[ 0 ] + pM[  9 ] * transformVertex[ 1 ] + pM[ 10 ] * transformVertex[ 2 ] + pM[ 11 ] * transformVertex[ 3 ];
        pVertices[ i ][ 3 ] = pM[ 12 ] * transformVertex[ 0 ] + pM[ 13 ] * transformVertex[ 1 ] + pM[ 14 ] * transformVertex[ 2 ] + pM[ 15 ] * transformVertex[ 3 ];
    }
}

void linAlg::loadPerspectiveFovYMatrix( mat4_t& matrix, const float fovY_deg, const float aspectRatio, const float zNear, const float zFar ) {

    const float xAspect = (aspectRatio > 1.0f) ? aspectRatio : 1.0f;
    const float yAspect = (aspectRatio > 1.0f) ? 1.0f : 1.0f / aspectRatio;

    const float yNoAspect = zNear * tanf( fovY_deg * 0.5f * static_cast<float>(M_PI) / 180.0f );
    const float yMax = yNoAspect * yAspect;
    const float xMax = yNoAspect * xAspect;
    loadPerspectiveMatrix( matrix, -xMax, xMax, -yMax, yMax, zNear, zFar );
}

void linAlg::loadPerspectiveMatrix( mat4_t& matrix, const float l, const float r, const float b, const float t, const float n, const float f ) {
    matrix[ 0 ] = vec4_t{ ( 2.0f * n ) / ( r - l ), 0.0f, ( r + l ) / ( r - l ), 0.0f };        // row 0
    matrix[ 1 ] = vec4_t{ 0.0f, ( 2.0f * n ) / ( t - b ), ( t + b ) / ( t - b ), 0.0f };        // row 1
    matrix[ 2 ] = vec4_t{ 0.0f, 0.0f, -( f + n ) / ( f - n ), ( -2.0f * f * n ) / ( f - n ) };  // row 2
    matrix[ 3 ] = vec4_t{ 0.0f, 0.0f, -1.0f, 0.0f };                                            // row 3
}

void linAlg::cast( mat4_t& mat4, const mat3x4_t& mat3x4 ) {
    mat4[ 0 ] = mat3x4[ 0 ];
    mat4[ 1 ] = mat3x4[ 1 ];
    mat4[ 2 ] = mat3x4[ 2 ];
    mat4[ 3 ] = vec4_t{ 0.0f, 0.0f, 0.0f, 1.0f };
}

void linAlg::quaternionFromAxisAngle( quat_t& quat, const vec3_t& axis, float angle ) {
    const float sina2 = (float)sinf( 0.5f * angle );
    const float norm = (float)sqrtf( axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] );
    quat[0] = sina2 * axis[0] / norm;
    quat[1] = sina2 * axis[1] / norm;
    quat[2] = sina2 * axis[2] / norm;
    quat[3] = (float)cosf( 0.5f * angle );
}

void linAlg::quaternionToMatrix( mat3x4_t& mat, const quat_t& quat ) {
    const float yy2 = 2.0f * quat[1] * quat[1];
    const float xy2 = 2.0f * quat[0] * quat[1];
    const float xz2 = 2.0f * quat[0] * quat[2];
    const float yz2 = 2.0f * quat[1] * quat[2];
    const float zz2 = 2.0f * quat[2] * quat[2];
    const float wz2 = 2.0f * quat[3] * quat[2];
    const float wy2 = 2.0f * quat[3] * quat[1];
    const float wx2 = 2.0f * quat[3] * quat[0];
    const float xx2 = 2.0f * quat[0] * quat[0];
    
    //mat[0 * 4 + 0] = -yy2 - zz2 + 1.0f;
    //mat[0 * 4 + 1] = xy2 + wz2;
    //mat[0 * 4 + 2] = xz2 - wy2;
    //mat[0 * 4 + 3] = 0;
    //mat[1 * 4 + 0] = xy2 - wz2;
    //mat[1 * 4 + 1] = -xx2 - zz2 + 1.0f;
    //mat[1 * 4 + 2] = yz2 + wx2;
    //mat[1 * 4 + 3] = 0;
    //mat[2 * 4 + 0] = xz2 + wy2;
    //mat[2 * 4 + 1] = yz2 - wx2;
    //mat[2 * 4 + 2] = -xx2 - yy2 + 1.0f;
    //mat[2 * 4 + 3] = 0;
    //mat[3 * 4 + 0] = mat[3 * 4 + 1] = mat[3 * 4 + 2] = 0;
    //mat[3 * 4 + 3] = 1;

    vec4_t& row0 = mat[0];
    vec4_t& row1 = mat[1];
    vec4_t& row2 = mat[2];

    row0 = vec4_t{ -yy2 - zz2 + 1.0f, xy2 - wz2, xz2 + wy2, 0.0f };
    row1 = vec4_t{ +xy2 + wz2, -xx2 - zz2 + 1.0f, yz2 - wx2, 0.0f };
    row2 = vec4_t{ xz2 - wy2, yz2 + wx2, -xx2 - yy2 + 1.0f, 0.0f };

    //row0 = vec4_t{ -yy2 - zz2 + 1.0f, xy2 + wz2, xz2 - wy2, 0.0f };
    //row1 = vec4_t{ xy2 - wz2, -xx2 - zz2 + 1.0f, yz2 + wx2, 0.0f };
    //row2 = vec4_t{ xz2 + wy2, yz2 - wx2, -xx2 - yy2 + 1.0f, 0.0f };

    //row3 = vec4_t{ 0.0f, 0.0f, 0.0f, 1.0f };
}

void linAlg::multQuaternion( quat_t& result, const quat_t& q1, const quat_t& q2 ) {
    result[0] = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1];
    result[1] = q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2];
    result[2] = q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0];
    result[3] = q1[3] * q2[3] - (q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2]);
}

void linAlg::multMatrix( mat4_t& result, const mat4_t& left, const mat4_t& right ) {
    float *const pM = getMatrixPtr( result );
    const float *const pL = getMatrixPtr( left );
    const float *const pR = getMatrixPtr( right );

    pM[  0 ] = pL[  0 ] * pR[ 0 ] + pL[  1 ] * pR[ 4 ] + pL[  2 ] * pR[  8 ] + pL[  3 ] * pR[ 12 ];
    pM[  1 ] = pL[  0 ] * pR[ 1 ] + pL[  1 ] * pR[ 5 ] + pL[  2 ] * pR[  9 ] + pL[  3 ] * pR[ 13 ];
    pM[  2 ] = pL[  0 ] * pR[ 2 ] + pL[  1 ] * pR[ 6 ] + pL[  2 ] * pR[ 10 ] + pL[  3 ] * pR[ 14 ];
    pM[  3 ] = pL[  0 ] * pR[ 3 ] + pL[  1 ] * pR[ 7 ] + pL[  2 ] * pR[ 11 ] + pL[  3 ] * pR[ 15 ];

    pM[  4 ] = pL[  4 ] * pR[ 0 ] + pL[  5 ] * pR[ 4 ] + pL[  6 ] * pR[  8 ] + pL[  7 ] * pR[ 12 ];
    pM[  5 ] = pL[  4 ] * pR[ 1 ] + pL[  5 ] * pR[ 5 ] + pL[  6 ] * pR[  9 ] + pL[  7 ] * pR[ 13 ];
    pM[  6 ] = pL[  4 ] * pR[ 2 ] + pL[  5 ] * pR[ 6 ] + pL[  6 ] * pR[ 10 ] + pL[  7 ] * pR[ 14 ];
    pM[  7 ] = pL[  4 ] * pR[ 3 ] + pL[  5 ] * pR[ 7 ] + pL[  6 ] * pR[ 11 ] + pL[  7 ] * pR[ 15 ];

    pM[  8 ] = pL[  8 ] * pR[ 0 ] + pL[  9 ] * pR[ 4 ] + pL[ 10 ] * pR[  8 ] + pL[ 11 ] * pR[ 12 ];
    pM[  9 ] = pL[  8 ] * pR[ 1 ] + pL[  9 ] * pR[ 5 ] + pL[ 10 ] * pR[  9 ] + pL[ 11 ] * pR[ 13 ];
    pM[ 10 ] = pL[  8 ] * pR[ 2 ] + pL[  9 ] * pR[ 6 ] + pL[ 10 ] * pR[ 10 ] + pL[ 11 ] * pR[ 14 ];
    pM[ 11 ] = pL[  8 ] * pR[ 3 ] + pL[  9 ] * pR[ 7 ] + pL[ 10 ] * pR[ 11 ] + pL[ 11 ] * pR[ 15 ];

    pM[ 12 ] = pL[ 12 ] * pR[ 0 ] + pL[ 13 ] * pR[ 4 ] + pL[ 14 ] * pR[  8 ] + pL[ 15 ] * pR[ 12 ];
    pM[ 13 ] = pL[ 12 ] * pR[ 1 ] + pL[ 13 ] * pR[ 5 ] + pL[ 14 ] * pR[  9 ] + pL[ 15 ] * pR[ 13 ];
    pM[ 14 ] = pL[ 12 ] * pR[ 2 ] + pL[ 13 ] * pR[ 6 ] + pL[ 14 ] * pR[ 10 ] + pL[ 15 ] * pR[ 14 ];
    pM[ 15 ] = pL[ 12 ] * pR[ 3 ] + pL[ 13 ] * pR[ 7 ] + pL[ 14 ] * pR[ 11 ] + pL[ 15 ] * pR[ 15 ];
}


#ifndef _LINALG_H_671dfc89_7e49_4a11_812b_1b04a6e746cc
#define _LINALG_H_671dfc89_7e49_4a11_812b_1b04a6e746cc

#include <cmath>
#include <array>
#include <cstdint>
#include <cstring> // for memcpy

#define DEFINE_OPERATORS   1
//#define DEBUG_PRINT        1

struct linAlg {

    template<typename val_T, std::size_t numElements>
    using vec_t = std::array< val_T, numElements >;

    template<std::size_t numElements>
    using floatVec_t = vec_t< float, numElements >;

    using vec2_t = floatVec_t< 2 >;
    using vec3_t = floatVec_t< 3 >;
    using vec4_t = floatVec_t< 4 >;

    using quat_t = vec4_t;


    using i32vec2_t = vec_t< int32_t, 2 >;
    using i32vec3_t = vec_t< int32_t, 3 >;

    using u16vec2_t = vec_t< uint16_t, 2 >;
    using u16vec3_t = vec_t< uint16_t, 3 >;

    using u32vec2_t = vec_t< uint32_t, 2 >;
    using u32vec3_t = vec_t< uint32_t, 3 >;


    template<typename val_T, std::size_t rows, std::size_t cols>
    using mat_t = vec_t< vec_t< val_T, cols >, rows >;

    using mat2_t   = mat_t< float, 2, 2 >;
    using mat3_t   = mat_t< float, 3, 3 >;
    using mat3x4_t = mat_t< float, 3, 4 >; // 3 rows, 4 columns
    using mat4_t   = mat_t< float, 4, 4 >;

//#if ( DEBUG_PRINT != 0 )
//    #include "linAlgPrint.h"
//#endif

    template< typename val_T, std::size_t numElementsDst, std::size_t numElementsSrc >
    static void castVector( vec_t<val_T, numElementsDst>& dst, const vec_t<val_T, numElementsSrc>& src ) {
        memcpy( dst.data(), src.data(), minimum( src.size(), dst.size() ) * sizeof( val_T ) );
        if (dst.size() > src.size()) {
           std::fill( &dst[src.size()], dst.data() + dst.size(), val_T{ 0 } );
        }
    }

    static void stripRowAndColumn( mat2_t& dst, const mat3_t&src, const int32_t row, const int32_t column );
    static void stripRowAndColumn( mat3_t& dst, const mat4_t&src, const int32_t row, const int32_t column );

    template< typename val_T >
    static val_T minimum( const val_T& x, const val_T& y ) {
        return ( x <= y ) ? x : y;
    }

    template< typename val_T >
    static val_T maximum( const val_T& x, const val_T& y ) {
        return ( x > y ) ? x : y;
    }

    template< typename val_T >
    static val_T clamp( const val_T& x, const val_T& minVal, const val_T& maxVal ) {
        return  minimum( maximum( x, minVal ), maxVal );
    }

    template< typename val_T >
    static void add( val_T& dst, const val_T& lhs, const val_T& rhs ) {
        for ( size_t i = 0; i < lhs.size(); i++ ) { dst[ i ] = lhs[ i ] + rhs[ i ]; }
    }

    template< typename val_T >
    static void sub( val_T& dst, const val_T& lhs, const val_T& rhs ) {
        for ( size_t i = 0; i < lhs.size(); i++ ) { dst[ i ] = lhs[ i ] - rhs[ i ]; }
    }

    template< typename val_T >
    static void scale( val_T& dst, const float& scale ) {
        for ( size_t i = 0; i < dst.size(); i++ ) { dst[ i ] *= scale; }
    }

    template< typename val_T >
    static float len( const val_T& valVec ) {
        float len = 0.0f;
        for ( size_t i = 0; i < valVec.size(); i++ ) { 
            len += valVec[ i ] * valVec[ i ]; 
        }
        return sqrtf( len );
    }

    //template< typename val_T >
    //static float dist( const val_T& v1, const val_T& v2 ) {
    //    float len = 0.0f;
    //    val_T diff;
    //    sub( diff, v2, v1 );
    //    for ( size_t i = 0; i < diff.size(); i++ ) { 
    //        len += diff[ i ] * diff[ i ]; 
    //    }
    //    return sqrtf( len );
    //}

    template< typename val_T >
    static void normalize( val_T& dst ) {
        float len = 0.0f;
        for ( size_t i = 0; i < dst.size(); i++ ) { 
            len += dst[ i ] * dst[ i ]; 
        }
        const float oneOverLen = 1.0f / sqrtf( len );
        for ( size_t i = 0; i < dst.size(); i++ ) { 
            dst[ i ] *= oneOverLen; 
        }
    }

    template< typename val_T >
    static inline val_T normalizeRet( const val_T& dstIn ) {
        val_T dst = dstIn;
        normalize( dst );
        return dst;
    }

    template < typename vec_T >
    static float dot( const vec_T& lhs, const vec_T& rhs ) {
        float accum = lhs[ 0 ] * rhs[ 0 ];
        for ( size_t i = 1; i < lhs.size(); i++ ) {
            accum += lhs[ i ] * rhs[ i ];
        }
        return accum;
    }

    template < typename vec_T >
    static inline float distSquared(const vec_T& lhs, const vec_T& rhs) {
        vec_T diffVec;
        linAlg::sub(diffVec, lhs, rhs);
        const float distSquared = dot(diffVec, diffVec);
        return distSquared;
    }

    template < typename vec_T >
    static inline float dist(const vec_T& lhs, const vec_T& rhs) {
        return sqrtf( distSquared( lhs, rhs ) );
    }

    static void cross( vec3_t& dst, const vec3_t& lhs, const vec3_t& rhs );

    template <size_t numRows_T, size_t numCols_T > 
    using any_mat_dim_T = std::array< std::array< float, numCols_T >, numRows_T >;

    template <size_t numRows_T, size_t numCols_T >
    static void orthogonalize( any_mat_dim_T< numRows_T, numCols_T >& mat ) {
        linAlg::normalize( mat[0] );
        linAlg::cross( mat[2], mat[0], mat[1] );
        linAlg::normalize( mat[2] );
        linAlg::cross( mat[1], mat[2], mat[0] );
        linAlg::normalize( mat[1] );
        linAlg::cross( mat[0], mat[1], mat[2] );
        linAlg::normalize( mat[0] );
    }

    static float determinant( const mat2_t& mat );
    static float determinant( const mat3_t& mat );
    static float determinant( const mat4_t& mat );

    template< size_t squareDim_T >
    using square_mat_T = std::array< std::array< float, squareDim_T >, squareDim_T >;

    template<size_t squareDim_T> 
    static void transpose( square_mat_T<squareDim_T>& dst, const square_mat_T<squareDim_T>& src ) {
        for (size_t y = 0; y < squareDim_T; y++) {
            for (size_t x = 0; x < squareDim_T; x++) {
                dst[y][x] = src[x][y];
            }
        }
    }

    static void inverse( mat2_t& dst, const mat2_t& src );
    static void inverse( mat3_t& dst, const mat3_t& src );
    static void inverse( mat4_t& dst, const mat4_t& src );

    template< typename mat_T >
    static const float *const getMatrixPtr( const mat_T& matrix ) {
        return matrix[ 0 ].data();
    }

    template< typename mat_T >
    static float *const getMatrixPtr( mat_T& matrix ) {
        return matrix[ 0 ].data();
    }

    static void loadIdentityMatrix( mat3_t& matrix );
    static void loadIdentityMatrix( mat3x4_t& matrix );
    static void loadIdentityMatrix( mat4_t& matrix );

    static void loadTranslationMatrix( mat3_t& matrix, const vec2_t& translationVec );
    static void loadTranslationMatrix( mat3x4_t& matrix, const vec3_t& translationVec );

    static void loadRotationXMatrix( mat3x4_t& matrix, const float radiantsX );
    static void loadRotationYMatrix( mat3x4_t& matrix, const float radiantsY );
    static void loadRotationZMatrix( mat2_t&   matrix, const float radiantsZ );
    static void loadRotationZMatrix( mat3x4_t& matrix, const float radiantsZ );

    static void loadRotationAroundAxis( mat3x4_t& rotMat, const vec3_t& axis, const float angleRad );

    static void loadScaleMatrix( mat2_t& matrix, const vec2_t& scaleVec );
    static void loadScaleMatrix( mat3_t& matrix, const vec3_t& scaleVec );
    static void loadScaleMatrix( mat3x4_t& matrix, const vec3_t& scaleVec );
    static void loadScaleMatrix( mat4_t& matrix, const vec4_t& scaleVec );

    static void multMatrix( mat2_t& result, const mat2_t& left, const mat2_t& right );
    static void multMatrix( mat3_t& result, const mat3_t& left, const mat3_t& right );
    static void multMatrix( mat3x4_t& result, const mat3x4_t& left, const mat3x4_t& right );
    static void multMatrix( mat4_t& result, const mat4_t& left, const mat4_t& right );

    static void applyTransformationToPoint(
        const mat3_t& transformationMatrix,
        vec3_t* const pVertices,
        const size_t numVertices );
    static void applyTransformationToPoint(
        const mat3x4_t& transformationMatrix, 
        vec3_t *const pVertices, 
        const size_t numVertices );
    static void applyTransformationToVector(
        const mat3x4_t& transformationMatrix,
        vec3_t* const pVector,
        const size_t numVertices );
    static void applyTransformationToPoint( 
        const mat4_t& transformationMatrix, 
        vec4_t *const pVertices, 
        const size_t numVertices );

    static void loadPerspectiveFovYMatrix( mat4_t& matrix, const float fovY_deg, const float aspectRatio, const float zNear, const float zFar );
    static void loadPerspectiveMatrix( mat4_t& matrix, const float l, const float r, const float b, const float t, const float n, const float f );
    static void loadOrthoMatrix( mat4_t& matrix, const float l, const float r, const float b, const float t, const float n, const float f );

    static void castMatrix( mat4_t& mat4, const mat3x4_t& mat3x4 );
    static void castMatrix( mat3_t& mat3, const mat3x4_t& mat3x4 );
    static void castMatrix( mat3x4_t& mat3x4, const mat3_t& mat3 );
    static void castMatrix( mat3x4_t& mat3x4, const mat4_t& mat4 );

    template<typename val_T, std::size_t rows_T, std::size_t cols_T>
    static bool approxEqual( const linAlg::mat_t< val_T, rows_T, cols_T >& matrixLhs,
                             const linAlg::mat_t< val_T, rows_T, cols_T >& matrixRhs,
                             const val_T epsilon = val_T{ 0.001 } ) {

        for (std::size_t col = 0; col < cols_T; col++) {
            for (std::size_t row = 0; row < rows_T; row++) {
                if ( abs( matrixLhs[row][col] - matrixRhs[row][col] ) > epsilon ) { return false; }
            }
        }

        return true;
    }

    // https://sourceforge.net/p/anttweakbar/code/ci/master/tree/examples/TwSimpleGLUT.c#l59
    static void quaternionFromAxisAngle( quat_t& quat, const vec3_t& axis, float angle );
    static void quaternionToMatrix( mat3x4_t& mat, const quat_t& quat );
    static void multQuaternion( quat_t& result, const quat_t& q1, const quat_t& q2 );

    static bool isPointInsideTriangle( const vec3_t& P, const std::array< vec3_t, 3 >& triangleVertexPositions );
};

#if ( DEFINE_OPERATORS != 0 )
    #include "linAlgOperators.inl"
#endif

#endif // _LINALG_H_671dfc89_7e49_4a11_812b_1b04a6e746cc

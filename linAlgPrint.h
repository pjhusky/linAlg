#ifndef _LINALG_PRINT_H_E739A378_0E68_4401_BB14_6D23611B7068
#define _LINALG_PRINT_H_E739A378_0E68_4401_BB14_6D23611B7068

#include "linAlg.h"
//#include <string>

    static void printVec( const char* label, const linAlg::vec3_t& vec ) {
        printf( "%s: (%f|%f|%f)\n", label, vec[0], vec[1], vec[2] );
    }
    static void printVec( const char* label, const linAlg::vec4_t& vec ) {
        printf( "%s: (%f|%f|%f|%f)\n", label, vec[0], vec[1], vec[2], vec[3] );
    }

    static void printVecDebug( const char* label, const linAlg::vec3_t& vec ) {
    #if (VERBOSE_DEBUG != 0)
        printf( "%s: (%f|%f|%f)\n", label, vec[0], vec[1], vec[2] );
    #endif
    }
    static void printVecDebug( const char* label, const linAlg::vec4_t& vec ) {
    #if (VERBOSE_DEBUG != 0)
        printf( "%s: (%f|%f|%f|%f)\n", label, vec[0], vec[1], vec[2], vec[3] );
    #endif
    }

#endif // _LINALG_PRINT_H_E739A378_0E68_4401_BB14_6D23611B7068

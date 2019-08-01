#ifndef DATATYPE_H
#define DATATYPE_H

typedef unsigned long UINT64;

#define USE_DOUBLE 1

#ifdef USE_DOUBLE
typedef double DataType_t;
#else
typedef int    DataType_t;
#endif

#endif

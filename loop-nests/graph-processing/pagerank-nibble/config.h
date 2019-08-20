#ifndef CONFIG_H
#define CONFIG_H

typedef enum
{
    MODE_UNTILED                                    = 0,
    MODE_UNTILED_PARALLEL                           = 1,
    MODE_TILED_PARALLEL_REFERENCE                   = 2,
    MODE_TILED_PARALLEL                             = 3,
    MODE_UNTILED_COMPRESSED_ITER_TRACE              = 13,
    MODE_UNTILED_COMPRESSED_PARALLEL_ITER_TRACE     = 14,
    MODE_UNTILED_COMPRESSED_PARALLEL_FGEN_TRACE     = 15,
    MODE_UNTILED_COMPRESSED_PARALLEL_PCOMP_TRACE    = 16,
    MODE_ISTA_TRACE     = 17,
} RUN_MODE;
    

typedef enum
{
    STATIONARY_DST_TILE                    = 0,
    STATIONARY_DST_TILE_AND_CT             = 1,
    STATIONARY_SRC_TILE                    = 2,
    STATIONARY_SRC_TILE_AND_CT             = 3,
} STATIONARY_TYPE;
        

static bool IsDstTileStationary( STATIONARY_TYPE type_in )
{
    if( type_in == STATIONARY_DST_TILE || type_in == STATIONARY_DST_TILE_AND_CT ) 
    {
        return true;
    }
    return false;
}

static bool IsSrcTileStationary( STATIONARY_TYPE type_in )
{
    if( type_in == STATIONARY_SRC_TILE || type_in == STATIONARY_SRC_TILE_AND_CT ) 
    {
        return true;
    }
    return false;
}


typedef enum
{
    FORMAT_CSR                             = 0,
    FORMAT_CSC                             = 1,
} FORMAT_TYPE;
    
#endif

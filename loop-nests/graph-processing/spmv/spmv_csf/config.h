#ifndef CONFIG_H
#define CONFIG_H

typedef enum
{
    MODE_UNTILED                           = 0,
    MODE_TILED_PRE_PROCESSED_SERIAL        = 1,
    MODE_TILED_PRE_PROCESSED_PARALLEL      = 2,
    MODE_TILED_ONLINE                      = 3,
    MODE_TILED_ONLINE_VER3                 = 5,
    MODE_TILED_PRE_PROCESSED_V2_PARALLEL   = 6,
    MODE_TILED_ONLINE_COPLAND              = 7,
    MODE_TILED_PREPROCESSED_COPLAND        = 8,
    MODE_UNTILED_COMPRESSED_LGC_NIBBLE     = 9,
    MODE_UNTILED_COMPRESSED_PARALLEL_LGC_NIBBLE     = 10,
    MODE_UNTILED_COMPRESSED_PARALLEL_TILED_LGC_NIBBLE     = 11,
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

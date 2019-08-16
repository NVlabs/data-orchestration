#include "whoop.hpp"
#include <math.h>
#include <map>
#include "csf.h"
#include <boost/timer/timer.hpp>
#include <chrono>

// using namespace whoop;
using namespace std;

/*********************************************************************************************************************/
/************************************  COMMAND LINE ARGUMENTS ********************************************************/
/*********************************************************************************************************************/

int ARG_BUFFET_GRANULARITY = LINE_SIZE_BYTES / BYTES_PER_VERTEX;  // buffet granularity
int ARG_S_TILE_SIZE        = 32; 
int ARG_D_TILE_SIZE        = 32; 
int ARG_TILE_SIZE_OVERRIDE = 0;
int ARG_PRINT_GRAPH        = 0;
int ARG_NUM_COMPUTE_TILE   = 1;
int ARG_NUM_DOT_C          = NUM_DOT_C;
int ARG_NUM_DOT_M          = -1;
int ARG_RF_BUFFET_SIZE_KB  = 32;
int ARG_L1_BUFFET_SIZE_KB  = 256;
int ARG_L2_BUFFET_SIZE_KB  = 2048;
int ARG_RUN_SERIAL         = 0;
int ARG_RUN_MODE           = 0;
int ARG_COMPRESSION_FORMAT = 0;
int ARG_MAX_ITERATIONS     = 1024;
int ARG_USE_SEED           = 0;

void MyInit(int argc, char** argv)
{
    AddOption( &ARG_S_TILE_SIZE, "src_tile_size", "Source      Tile Size");
    AddOption( &ARG_D_TILE_SIZE, "dst_tile_size", "Destination Tile Size");
    AddOption( &ARG_TILE_SIZE_OVERRIDE, "override_tile_size", "Override The Tile Size");
    AddOption( &ARG_BUFFET_GRANULARITY, "granularity", "fetch granularity (in item count) from next level of memory");
    AddOption( &ARG_PRINT_GRAPH, "print", "print the input graph in CSF");
    AddOption( &ARG_NUM_COMPUTE_TILE, "numCT", "Number of Compute Tiles");
    AddOption( &ARG_NUM_DOT_C, "dotC", "Number of Compute Tiles");
    AddOption( &ARG_NUM_DOT_M, "dotM", "Number of Compute Tiles");
    AddOption( &ARG_RF_BUFFET_SIZE_KB, "RF", "RF Buffet Size");
    AddOption( &ARG_L1_BUFFET_SIZE_KB, "L1", "L1 Buffet Size");
    AddOption( &ARG_L2_BUFFET_SIZE_KB, "L2", "L2 Buffet Size");
    AddOption( &ARG_RUN_SERIAL, "serial", "Serial Mode");
    AddOption( &ARG_RUN_MODE, "mode", "Run Mode");
    AddOption( &ARG_COMPRESSION_FORMAT, "format", "Compression Format");
    AddOption( &ARG_MAX_ITERATIONS, "max_iter", "Max Iterations");
    AddOption( &ARG_USE_SEED, "seed", "seed");

    whoop::Init(argc, argv);
}

/*********************************************************************************************************************/
/**************************************  TILING PARAMETERS ***********************************************************/
/*********************************************************************************************************************/

void CreateBufferBasedTileSizes( int V, int& S0, int& D0, int& S1, int& D1, int& S2, int& D2 )
{
    int L1_Vertex_Capacity = ARG_L1_BUFFET_SIZE_KB * KILO / BYTES_PER_VERTEX;
    int L2_Vertex_Capacity = ARG_L2_BUFFET_SIZE_KB * KILO / BYTES_PER_VERTEX;

    S0 = L1_Vertex_Capacity;
    D0 = L2_Vertex_Capacity;

    if( S0 > V ) S0 = V;
    if( D0 > V ) D0 = V;  

    S1 = (V % S0) ? (V/S0+1) : (V/S0);
    D1 = (V % D0) ? (V/D0+1) : (V/D0);

    if( S1 < NUM_DOT_C ) 
    {
        S2 = 1;
    }
    else 
    {
        S2 = S1/NUM_DOT_C + ( (S1%NUM_DOT_C) ? 1 : 0 );
        S1 = NUM_DOT_C;
    }

//     if( D1 < NUM_DOT_M ) 
//     {
        D2 = 1;
//     }
}

bool SetupTileSizes( int V, int& S0, int& D0, int& S1, int& D1, int& S2, int& D2 )
{
    //
    bool amTiled = true;
  
    S0 = (ARG_S_TILE_SIZE == 0) ? V : ARG_S_TILE_SIZE;
    D0 = (ARG_D_TILE_SIZE == 0) ? V : ARG_D_TILE_SIZE;

    if( (ARG_S_TILE_SIZE==0 ) && (ARG_D_TILE_SIZE == 0) ) 
    {
        amTiled = false;
    }

    if( S0 > V ) S0 = V;
    if( D0 > V ) D0 = V;  

    S1 = (V % S0) ? (V/S0+1) : (V/S0);
    D1 = (V % D0) ? (V/D0+1) : (V/D0);

    if( S1 < NUM_DOT_C ) 
    {
        S2 = 1;
    }
    else 
    {
        S2 = S1/NUM_DOT_C + ( (S1%NUM_DOT_C) ? 1 : 0 );
        S1 = NUM_DOT_C;
    }


//     if( D1 < NUM_DOT_M ) 
//     {
        D2 = 1;
//     }

    return amTiled;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/

void PrintInfo( GraphAlgorithm &graphAlg, bool print_arch=true )
{
    graphAlg.PrintInfo();

    if( print_arch ) 
    {
        cout<<"================================================="<<endl;
        cout<<"SPMV On Symphony With Architecture Parameters"<<endl;
        cout<<"\tNumber of Compute Tiles:    "<<ARG_NUM_COMPUTE_TILE<<endl;
        cout<<"\tNumber of DOT-Cs:           "<<ARG_NUM_DOT_C<<endl;
        cout<<"\tNumber of DOT-Ms:           "<<ARG_NUM_DOT_M<<endl;
        cout<<"\tL1 Buffer Size (KB):        "<<ARG_L1_BUFFET_SIZE_KB<<endl;
        cout<<"\tL2 Buffer Size (KB):        "<<ARG_L2_BUFFET_SIZE_KB<<endl;
        cout<<"================================================="<<endl;
    }
    
    cout<<endl;
}


void PrintTileInfo( const vector<int>& tile_sizes )
{

    cout<<"================================================="<<endl;
    cout<<"Tile Sizes"<<endl;
    cout<<"\tTile Size  (S0):           "<<tile_sizes[0]<<endl;
    cout<<"\tTile Size  (D0):           "<<tile_sizes[1]<<endl;
    cout<<endl;
    cout<<"\t# of Tiles (S1):           "<<tile_sizes[2]<<endl;
    cout<<"\t# of Tiles (D1):           "<<tile_sizes[3]<<endl;
    cout<<endl;
    cout<<"\t# of Tiles (S2):           "<<tile_sizes[4]<<endl;
    cout<<"\t# of Tiles (D2):           "<<tile_sizes[5]<<endl;
    cout<<"\tTot # of Tiles:            "<<(tile_sizes[2]*tile_sizes[3]*tile_sizes[4]*tile_sizes[5])<<endl;
    cout<<"================================================="<<endl;
    
    cout<<endl;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/

int main(int argc, char** argv)
{

    /******************************************************/
    // Set up the graph inputs
    /******************************************************/
    VecIn  SegmentArray("segarray");
    VecIn  SrcCoords("coordarray");

    /******************************************************/
    // Initialize and read the segment and cordinate arrays
    /******************************************************/
    MyInit( argc, argv);

    /******************************************************/
    // Set up the input graph and tiled related parameters
    /******************************************************/
    int numVertices  = SegmentArray.Size()-1;
    int numEdges     = SrcCoords.Size();
 
    const int E      = numEdges;
    const int V      = numVertices;  
    const int S      = numVertices;  
    const int D      = numVertices;  

    int S0, D0, S1, D1, S2, D2;
    /******************************************************/
    // Set up the graph algorithm 
    /******************************************************/
    FORMAT_TYPE formatIn = (FORMAT_TYPE) ARG_COMPRESSION_FORMAT;
    GraphAlgorithm graphAlg( formatIn, &SegmentArray, &SrcCoords, NULL );

    /******************************************************/
    /******************************************************/

   
    if( ARG_RUN_MODE == MODE_UNTILED ) 
    {

        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Run Native
        cout<<"(+)  Running Untiled Natively ..."<<endl;    
        graphAlg.Untiled( S0, S1, S2, formatIn );

        cout<<endl;
        cout<<"(+)  Running Untiled Whoop Mode ..."<<endl;
        graphAlg.WhoopUntiled( S0, S1, S2, ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn );
    }
    else if( ARG_RUN_MODE == MODE_TILED_PRE_PROCESSED_SERIAL ) 
    {
        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Pre-Process and Create The Meta-Data
        graphAlg.SetTileSizes( {S0,D0,S1,D1,S2,D2}, formatIn, STATIONARY_DST_TILE );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Run Native
        cout<<"(+)  Running Natively ..."<<endl;    
        graphAlg.Run5( formatIn );

        // Run Whoop
        cout<<endl;
        cout<<"(+)  Running In Serial Whoop Mode ..."<<endl;
        graphAlg.RunWhoop5( ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn );
    }
    else if( ARG_RUN_MODE == MODE_TILED_PRE_PROCESSED_PARALLEL ) 
    {
        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Pre-Process and Create The Meta-Data
        graphAlg.SetTileSizes( {S0,D0,S1,D1,S2,D2}, formatIn, STATIONARY_DST_TILE );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

        // Run Native
        cout<<"(+)  Running Natively ..."<<endl;    
        graphAlg.Run5( formatIn );

        // Run Whoop
        cout<<endl;
        cout<<"(+)  Running In Parallel Whoop Mode ..."<<endl;
        graphAlg.RunWhoop5_Parallel( ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn );
    }
    else if( ARG_RUN_MODE == MODE_TILED_PRE_PROCESSED_V2_PARALLEL ) 
    {
        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Pre-Process and Create The Meta-Data
        graphAlg.SetTileSizes( {S0,D0,S1,D1,S2,D2}, formatIn, STATIONARY_DST_TILE_AND_CT );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

        if( formatIn != FORMAT_CSR )
        {
            cout<<"Unsupported Format"<<endl;
            exit(0);
        }

        // Run Native
        cout<<"(+)  Running Natively ..."<<endl;    

        auto start = chrono::steady_clock::now();
        graphAlg.Run5_CSR_ver2();
        auto end = chrono::steady_clock::now();

        auto diff = end - start;
        cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;

        // Run Whoop
        cout<<endl;
        cout<<"(+)  Running In Parallel Whoop Mode ..."<<endl;
        graphAlg.RunWhoop5_Parallel_CSR_ver2( ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB );
    }
    else if( ARG_RUN_MODE == MODE_TILED_ONLINE ) 
    {
        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

        // Run Natively
        cout<<endl;
        cout<<"(+)  Running Online Tiling Natively ..."<<endl;    
        graphAlg.OnlineTiling(D1, S2, S1, D0, S0, formatIn);

        // Run Whoop
        cout<<endl;
        cout<<"(+)  Running Online Tiling Whoop ..."<<endl;    
        graphAlg.WhoopOnlineTiling(D1, S2, S1, D0, S0,  ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn );
    }
    else if( ARG_RUN_MODE == MODE_TILED_ONLINE_VER3 ) 
    {
        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

        // Run Natively
        cout<<endl;
        cout<<"(+)  Running Online Tiling v2 Natively ..."<<endl;    
        graphAlg.OnlineTiling_ver2(D1, S2, S1, D0, S0, formatIn);

        // Run Whoop
        cout<<endl;
        cout<<"(+)  Running Online Tiling v3 Whoop ..."<<endl;    
        graphAlg.WhoopOnlineTiling_ver3(D1, S2, S1, D0, S0,  ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn );
    }
    else if( ARG_RUN_MODE == MODE_TILED_ONLINE_COPLAND ) 
    {
        // Print Setup Information
        PrintInfo( graphAlg, false );

        // Run Natively
        cout<<endl;
        cout<<"(+)  Running Online Tiling Copland ..."<<endl;    
        graphAlg.OnlineTiling_CSR_Copland();

        // Run Whoop
        cout<<endl;
        cout<<"(+)  Running Whoop Online Tiling Copland ..."<<endl;    
        graphAlg.WhoopOnlineTiling_CSR_Copland();
    }
    else if( ARG_RUN_MODE == MODE_TILED_PREPROCESSED_COPLAND )
    {
        // Print Setup Information
        PrintInfo( graphAlg, false );

        // Run Natively
        cout<<endl;
        cout<<"(+)  Running Online Tiling Copland ..."<<endl;    
        graphAlg.OnlineTiling_CSR_Copland();

    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_LGC_NIBBLE )
    {
        int seed = ARG_USE_SEED;

        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

//     graphAlg.PageRankNibble_Untiled( seed );
//     graphAlg.Whoop_PageRankNibble_Untiled(seed, ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn);


        auto start = chrono::steady_clock::now();
        graphAlg.PageRankNibble_Untiled_Compressed( seed );
        auto end = chrono::steady_clock::now();

        auto diff = end - start;
        cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;

        graphAlg.Whoop_PageRankNibble_Untiled_Compressed(seed, ARG_RF_BUFFET_SIZE_KB, ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn);
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_PARALLEL_LGC_NIBBLE )
    {
        int seed = ARG_USE_SEED;

        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

        graphAlg.PageRankNibble_Untiled_Compressed_Parallel( seed );
        graphAlg.Whoop_PageRankNibble_Untiled_Compressed_Parallel(seed, ARG_RF_BUFFET_SIZE_KB, ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn);
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_PARALLEL_TILED_LGC_NIBBLE )
    {
        int seed = ARG_USE_SEED;

        // Determine Tile Sizes Using On-Chip Buffer Sizes
        CreateBufferBasedTileSizes( V, S0, D0, S1, D1, S2, D2 );

        // Print Setup Information
        PrintInfo( graphAlg );

        // Print Tile Info
        PrintTileInfo( {S0,D0,S1,D1,S2,D2} );

        graphAlg.PageRankNibble_Untiled_Compressed_Parallel_Tiled( seed );
        graphAlg.Whoop_PageRankNibble_Untiled_Compressed_Parallel_Tiled(seed, ARG_RF_BUFFET_SIZE_KB, ARG_L1_BUFFET_SIZE_KB, ARG_L2_BUFFET_SIZE_KB, formatIn);
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_PARALLEL_REFERENCE )
    {
        int seed = ARG_USE_SEED;

        graphAlg.PageRankNibble_Untiled_Compressed_Parallel_Tiled( seed );
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_ITER_TRACE )
    {
        int seed = ARG_USE_SEED;

        graphAlg.PageRankNibble_Untiled_Compressed_TraceIter( seed );
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_PARALLEL_ITER_TRACE )
    {
        int seed = ARG_USE_SEED;

        graphAlg.PageRankNibble_Untiled_Compressed_Parallel_TraceIter( seed );
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_PARALLEL_FGEN_TRACE )
    {
        int seed = ARG_USE_SEED;

        graphAlg.PageRankNibble_Untiled_Compressed_Parallel_TraceFrontierGeneration( seed );
    }    
    else if( ARG_RUN_MODE == MODE_UNTILED_COMPRESSED_PARALLEL_PCOMP_TRACE )
    {
        int seed = ARG_USE_SEED;

        graphAlg.PageRankNibble_Untiled_Compressed_Parallel_TracePRComp( seed );
    }    
    else if( ARG_RUN_MODE == MODE_ISTA_TRACE )
    {
        int seed = ARG_USE_SEED;

        graphAlg.Whoop_PageRankNibble_Untiled_Compressed_Parallel_TraceISTAIter( seed );
    }    
    else 
    {
        cout<<"Unknown Mode"<<endl;
        exit(0);
    }
}

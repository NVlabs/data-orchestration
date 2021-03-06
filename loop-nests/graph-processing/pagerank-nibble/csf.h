#ifndef CSF_DEF
#define CSF_DEF

#include <map>
#include "whoop.hpp"
#include <algorithm>
#include <string>
#include "config.h"

using namespace whoop;
using namespace std;

#define KILO 1024
#define MEGA ((KILO)*(KILO))
#define GIGA ((MEGA)*(KILO))

#define LINE_SIZE_BYTES    128
#define BYTES_PER_VERTEX   8

#ifndef NUM_DOT_C
#define NUM_DOT_C 2
#endif

extern int ARG_BUFFET_GRANULARITY;
extern int ARG_MAX_ITERATIONS;

class CSF
{

  public:
    
    int              numDims;
    int              nnz_in;
    
    vector<int>     dim_sizes;
    vector<string>  dim_names;

    vector<int>     SA_insert_ptr;  // segment array insert pointer

    vector<int>     CA_insert_ptr;  // coordinate array insert pointer
    vector<int>     CA_last_insert; // coordinate array last insert val

    Tensor**         segArray;
    Tensor**         coordArray;

    Tensor*          valArray;

  public:

    CSF( const vector<int>& dim_sizes_, const vector<string>& dim_names_, long nnz )
    {
        numDims   = dim_sizes_.size();
        dim_sizes = dim_sizes_;
        dim_names = dim_names_;
        
        Init(nnz);        
    }


    long GetStorage( void )
    {
        long sum = 0;
        
        for(int dim=numDims-1; dim>=0; dim--) 
        {
            sum += SA_insert_ptr[dim];
            sum += CA_insert_ptr[dim];
        }

        return sum;
    }


    void TrimFat( void )
    {
        for(int dim=numDims-1; dim>=0; dim--) 
        {
            segArray[dim]->Resize( {SA_insert_ptr[dim]+1} );
            coordArray[dim]->Resize( {CA_insert_ptr[dim]+1} );
        }

        if( CA_insert_ptr[0] != nnz_in ) 
        {
            PrintCSF();
            
            cout<<"NNZ expected: "<<nnz_in<<endl;
            cout<<"CSF NNZ:      "<<CA_insert_ptr[0]<<endl;
            exit(0);
        }
    }

    void PrintCSF( int print_coords=false )
    {
        cout<<endl;
        cout<<"Compressed Sparse Fiber (CSF) Representation Of Graph:"<<endl;
        cout<<endl;
        for(int dim=numDims-1; dim>=0; dim--) 
        {
            cout<<"\tDim: "<<dim<<" SegmentArray["<<dim_names[dim]<<"] = ";
            for(int p=0; print_coords && p<SA_insert_ptr[dim]+1; p++)
            {
                cout<<segArray[dim]->At({p})<<", ";
            }
            cout<<" WrPtr: "<<SA_insert_ptr[dim]<<" Max: "<<segArray[dim]->size()<<endl;

            cout<<"\tDim: "<<dim<<" CoordinateArray["<<dim_names[dim]<<"] = ";
            for(int p=0; print_coords && p<CA_insert_ptr[dim]; p++)
            {
                cout<<coordArray[dim]->At({p})<<", ";
            }
            cout<<" WrPtr: "<<CA_insert_ptr[dim]<<" Max: "<<coordArray[dim]->size()<<endl;
            cout<<endl;
        }
    }

    void Init( int nnz )
    {
        int dim = numDims - 1;        
        int segSize, coordSize, prev_coordSize;

        segArray    = new Tensor*[ numDims ];
        coordArray  = new Tensor*[ numDims ];
        valArray    = new Tensor( "TensorValueArray" );


        //
        nnz_in = nnz;

        // allocate space for the value array equivalent to nnz
        valArray->Resize( {nnz} );

        // init insert pointer and count
        SA_insert_ptr.resize( numDims );

        CA_insert_ptr.resize( numDims );
        CA_last_insert.resize( numDims );


        // Set initial sizes for segment and coordinate arrays, later
        // it will be resized based on insertions

        segSize   = 512;
        coordSize = 512;

        // initialize the per dimension segment and coordinate arrays
        while( dim >= 0 )
        {
            // allocate the tensor and coordinate array for this dimension
            segArray[dim]   = new Tensor("SegmentArray-" + to_string(dim));
            coordArray[dim] = new Tensor("CoordArray-" + to_string(dim));

            segArray[dim]->Resize( {segSize} );
            coordArray[dim]->Resize( {coordSize} );

            // initialize the segment array 
            segArray[dim]->At({0}) = 0;
            SA_insert_ptr[dim] = 0;
            CA_insert_ptr[dim] = 0;
            CA_last_insert[dim] = -1;
            
            dim--;
        }

        // Initialize the root table of the segment array
        segArray[numDims-1]->At({1}) = 0;
        SA_insert_ptr[numDims-1] = 1;
    }

    void Insert( const vector<int>& coord, int valIn=1 )
    {
        int insert = 0;

        for(int dim=coord.size()-1; dim>=0; dim--)
        {
            if( (CA_last_insert[dim] != coord[dim]) || insert )
            {
                insert |= 1;

                coordArray[dim]->At( {CA_insert_ptr[dim]} ) = coord[dim];

                if( dim == 0 )
                {
                    valArray->At( {CA_insert_ptr[dim]} ) = valIn;
                }

                CA_last_insert[dim] = coord[dim];
                CA_insert_ptr[dim]++;

                // if inserting at anything but the leaf dimension, we
                // need to start a new segment, hence increment the
                // write pointers for the level below

                if( dim ) 
                {
                    SA_insert_ptr[dim-1]++;

                    int wr_ptr = SA_insert_ptr[dim-1];
                    
                    // if not enough space resize
                    if( wr_ptr == segArray[dim-1]->size() )
                    {
                        segArray[dim-1]->Resize( {2*SA_insert_ptr[dim-1]} );
                    }
                    
                    // set segment start pointer to be whatever it was before
                    segArray[dim-1]->At( {wr_ptr} ) = segArray[dim-1]->At( {wr_ptr-1} ) ;
                }
                
                segArray[dim]->At( {SA_insert_ptr[dim]} ) = CA_insert_ptr[dim];

                if( CA_insert_ptr[dim] == coordArray[dim]->size() )
                {
                    coordArray[dim]->Resize( {2*CA_insert_ptr[dim]} );
                }
            }
        }

    }

    bool Locate( int d1, int s1, int &os, int &oe )
    {
        int levelsSearched  = 0;
        int currDim = numDims-1;
        
        bool found = false;

        int compareCoord = d1;

        int pos_start = 0;
        int pos_end  = coordArray[currDim]->size();

//         cout<<"Locate -- D1: "<<d1<<" S1: "<<s1<<endl;

        while( pos_start < pos_end ) 
        {
            int currCoord = coordArray[currDim]->At( {pos_start} );
//             cout<<"\tComparing: "<<currCoord<<" to: "<<compareCoord<<endl;
            if( currCoord == compareCoord )
            {
//                 cout<<"\tFound Coordinate: "<<compareCoord<<endl;

                int old_pos_start = pos_start;
                levelsSearched++;
                currDim--;
                pos_start = segArray[currDim]->At( {old_pos_start} );
                pos_end = segArray[currDim]->At( {old_pos_start+1} );
                compareCoord = s1;

//                 cout<<"\tnew start: "<<pos_start<<" new end: "<<pos_end<<endl;
                if( levelsSearched == 2 )
                {
                    os = pos_start;
                    oe = pos_end;
                    return true;
                }
            }
//             else if( currCoord > compareCoord )
//             {
// //                 cout<<"Breaking Out Early: "<<levelsSearched<<endl;
//                 return false;
//             }
            else {
                pos_start++;
            }
        }
        return false;
    }

    int GetCoordinateVal( int dimArg, int pos )
    {
        return coordArray[dimArg]->At( {pos} );
    }

    int GetSegArrayVal( int dimArg, int pos )
    {
        return segArray[dimArg]->At( {pos} );
    }


//     void ReferenceAlgorithm( int V, int E, const vector<int> DimMax )
//     {
//         int D1 = DimMax[0];
//         int S1 = DimMax[1];
//         int D0 = DimMax[2];
//         int S0 = DimMax[3];
// 
//         int start_s1_pos, end_s1_pos;
//         int start_s0_pos, end_s0_pos;
//         int start_d0_pos, end_d0_pos;
// 
//         int s1, d1, s0, d0;
// 
//         int D1_max = CA_insert_ptr[3];
// 
//         for(int d1_pos=0; d1_pos<D1_max; d1_pos++)
//         {
//             d1 = coordArray[3]->At({d1_pos});
//             
//             start_s1_pos = segArray[2]->At({d1_pos});
//             end_s1_pos   = segArray[2]->At({d1_pos+1});
// 
//             for(int s1_pos=start_s1_pos; s1_pos<end_s1_pos; s1_pos++)
//             {
//                 s1 = coordArray[2]->At({s1_pos});
// 
//                 start_s0_pos = segArray[1]->At({s1_pos});
//                 end_s0_pos   = segArray[1]->At({s1_pos+1});
// 
//                 for(int s0_pos=start_s0_pos; s0_pos<end_s0_pos; s0_pos++ ) 
//                 {
//                     s0 = coordArray[1]->At({s0_pos}); 
//                     
//                     start_d0_pos = segArray[0]->At({s0_pos}); 
//                     end_d0_pos   = segArray[0]->At({s0_pos+1}); 
//                     
//                     for(int d0_pos=start_d0_pos; d0_pos<end_d0_pos; d0_pos++)
//                     {
//                         d0 = coordArray[0]->At({d0_pos}); 
// 
//                         cout<<"\t"<<d1<<","<<s1<<","<<s0<<","<<d0<<endl;
//                     }
//                     
//                 }
//             }
//         }
// 
//     }
// 
//     void WhoopAlgorithm( int V, int E, const vector<int> DimMax )
//     {
//         int D1 = DimMax[0];
//         int S1 = DimMax[1];
//         int D0 = DimMax[2];
//         int S0 = DimMax[3];
// 
//         Var start_s1_pos("start_s1_pos"), end_s1_pos("end_s1_pos");
//         Var start_d1_pos("start_d1_pos"), end_d1_pos("end_d1_pos");
//         Var start_s0_pos("start_s0_pos"), end_s0_pos("end_s0_pos");
//         Var start_d0_pos("start_d0_pos"), end_d0_pos("end_d0_pos");
// 
//         Var s1_pos("s1_pos");
//         Var d1_pos("d1_pos");
//         Var s0_pos("s0_pos");
//         Var d0_pos("d0_pos");
// 
//         Var s1("s1"), d1("d1"), s0("s0"), d0("d0");
// 
//         int D1_max = CA_insert_ptr[3];
// 
//         t_for(d1_pos, 0, D1_max);
//         {
//             d1 = coordArray[3][d1_pos];
//             
//             start_s1_pos = segArray[2][d1_pos];
//             end_s1_pos   = segArray[2][d1_pos+1];
// 
//             t_for(s1_pos, start_s1_pos, end_s1_pos);
//             {
//                 s1 = coordArray[2][s1_pos];
//                 
//                 start_s0_pos = segArray[1][s1_pos];
//                 end_s0_pos   = segArray[1][s1_pos+1];
//                 
//                 t_for(s0_pos, start_s0_pos, end_s0_pos );
//                 {
//                     s0 = coordArray[1][s0_pos];
//                     
//                     start_d0_pos = segArray[0][s0_pos];
//                     end_d0_pos   = segArray[0][s0_pos+1];
//                     
//                     t_for(d0_pos, start_d0_pos, end_d0_pos );
//                     {
//                         d0 = coordArray[0][d0_pos]; 
// 
// 
//                     }
//                     end();
//                 }
//                 end();
//             }
//             end();
//         }
//         end();
// 
// 
//         cout<<endl;
//         cout<< "RUNNING WHOOP..." <<endl;
//         whoop::Run();
//         cout<< "DONE WHOOP..." <<endl;
// 
//         whoop::Done();
//     }
// 
//     void Whoop_RunGraphAlgorithm( int V, int E, const vector<int> DimMax )
//     {
//         int D1 = DimMax[0];
//         int S1 = DimMax[1];
//         int D0 = DimMax[2];
//         int S0 = DimMax[3];
// 
//         int start_s0_pos, end_s0_pos;
//         int start_d0_pos, end_d0_pos;
// 
//         int s0, d0;
// 
//         for(int d1=0; d1<D1; d1++)
//         {
//             for(int s1=0; s1<S1; s1++)
//             {
//                 bool found = Locate( d1, s1, start_s0_pos, end_s0_pos );
// 
//                 if( found ) 
//                 {
//                     for(int s0_pos=start_s0_pos; s0_pos<end_s0_pos; s0_pos++ ) 
//                     {
//                         s0 = coordArray[1]->At({s0_pos}); //GetCoordinateVal(1, s0_pos);
//                     
//                         start_d0_pos = segArray[0]->At({s0_pos}); //GetSegArrayVal(0,s0_pos);
//                         end_d0_pos   = segArray[0]->At({s0_pos+1}); //GetSegArrayVal(0,s0_pos+1);
//                     
//                         for(int d0_pos=start_d0_pos; d0_pos<end_d0_pos; d0_pos++)
//                         {
//                             d0 = coordArray[0]->At({d0_pos}); //GetCoordinateVal(0,d0_pos);
// 
//                             cout<<"\t"<<d1<<","<<s1<<","<<s0<<","<<d0<<endl;
//                         }
//                     
//                     }
//                 }
//             }
//         }
// 
//     }

};


class GraphAlgorithm 
{

  private:
    VecIn*       SegmentArray;
    Vec*         DeltaArray;

    VecIn*       CoordinateArray;
    VecIn*       ValueArray;
    FORMAT_TYPE  inputGraphFormat;

    CSF*         adjMat_csf;

    Tensor*      pageRank;
    Tensor*      frontier;
    Tensor*      residual;
    Tensor*      residual_prime;
    double       alpha;
    double       epsilon;
    
    
    // Graph Specific
    Tensor*      outDegree;
    Tensor*      inDegree;
    bool         degreeCalc;
    
    int          V;
    int          E;
    
  public:

    void InitGraphDataStructures()
    {
        for( int i=0; i<V; i++ )
        {
            frontier->At( {i} ) = 1;
            pageRank->At( {i} ) = 0;
            residual->At( {i} ) = 0;
            residual_prime->At( {i} ) = 0;
        }
    }

    void Init()
    {
        V                = SegmentArray->Size()-1;
        E                = CoordinateArray->Size();
        adjMat_csf       = NULL;
        DeltaArray       = NULL;

        // page rank init
        pageRank->Resize( {V} );
        frontier->Resize( {V} );
        residual->Resize( {V} );
        residual_prime->Resize( {V} );

        degreeCalc = false;
        outDegree->Resize( {V} );
        inDegree->Resize( {V} );

        if( ValueArray == NULL ) 
        {
            ValueArray = new VecIn( "InputValArray");
            ValueArray->Resize( {E} );

            for( int i=0; i<E; i++ )
            {
                ValueArray->At({i}) = 1;
            }
        }

        InitGraphDataStructures();

    }

    void CalculateDegrees( bool print=false)
    {
        if( !degreeCalc ) 
        {
            degreeCalc = true;

            if( inputGraphFormat == FORMAT_CSR )
            {
                for(int s=0; s<V; s++)
                {
                    int start = SegmentArray->At(s);
                    int end   = SegmentArray->At(s+1);

                    for(int pos=start; pos<end; pos++)
                    {
                        int d = CoordinateArray->At(pos);

                        outDegree->At({s})++;
                        inDegree->At({d})++;
                    }
                }
            }
            else if( inputGraphFormat == FORMAT_CSC )
            {
                for(int d=0; d<V; d++)
                {
                    int start = SegmentArray->At(d);
                    int end   = SegmentArray->At(d+1);

                    for(int pos=start; pos<end; pos++)
                    {
                        int s = CoordinateArray->At(pos);

                        outDegree->At({s})++;
                        inDegree->At({d})++;
                    }
                }
            }
            else 
            {
                assert(0);
            }
        }
    }

    void InitDeltaArray()
    {
        if( DeltaArray == NULL ) 
        {
            DeltaArray = new Vec( "DeltaArray" );
        }
        
        DeltaArray->Resize(V);

        for(int v=0; v<V; v++)
        {
            DeltaArray->At(v) = 0;
        }

    }
    
    GraphAlgorithm( FORMAT_TYPE myFormat, VecIn *segArray, VecIn *coordArray, VecIn *valArray )
    {
        inputGraphFormat = myFormat;
        SegmentArray     = segArray;
        CoordinateArray  = coordArray;
        ValueArray       = valArray;

        // init Page Rank Nibble
        pageRank       = new Tensor("pagerank");
        frontier       = new Tensor("frontier");
        residual       = new Tensor("residual");
        residual_prime = new Tensor("residual_prime");
        alpha          = 0.15;
        epsilon        = 1e-6;
        
        // graph specific
        inDegree  = new Tensor("inDegree");
        outDegree = new Tensor("outDegree");
        
        Init();        
    }

    // CSF 4 expects tile sizes in format:  (D1, S, D0)
    void CreateCSF_3_CSR( const vector<int>& tile_sizes_ )
    {
        int D1 = tile_sizes_[2];
        int S  = tile_sizes_[1];
        int D0 = tile_sizes_[0];

        // Init Delta Array
        InitDeltaArray();

        for(int d1=0; d1<D1; d1++)
        {
            for(int s=0; s<S; s++)
            {
                int pos_start = SegmentArray->At(s) + DeltaArray->At(s);
                int pos_end = SegmentArray->At(s+1);

                for(int p=pos_start; p<pos_end; p++)
                {
                    int d = CoordinateArray->At(p);
                        
                    int tilenum = s/D0;
                        
                    if( tilenum == d1 )
                    {
                        // When inserting, the insert function expects
                        // the coordinates align with their dimension
                        adjMat_csf->Insert( {d%D0, s, d1}, ValueArray->At(p) );

                        outDegree->At({s})++;
                        inDegree->At({d})++;
                    }
                    else 
                    {
                        DeltaArray->At(s) += (p - pos_start);                                    
                        p = pos_end;
                    }
                }
            }
        }

        adjMat_csf->TrimFat();
    }

    // CSF 4 expects tile sizes in format:  (S1, D, S0)
    void CreateCSF_3_CSC( const vector<int>& tile_sizes_ )
    {
        int S1 = tile_sizes_[2];
        int D  = tile_sizes_[1];
        int S0 = tile_sizes_[0];

        // Init Delta Array
        InitDeltaArray();

        for(int s1=0; s1<S1; s1++)
        {
            for(int d=0; d<V; d++)
            {
                int pos_start = SegmentArray->At(d) + DeltaArray->At(d);
                int pos_end = SegmentArray->At(d+1);

                for(int p=pos_start; p<pos_end; p++)
                {
                    int s = CoordinateArray->At(p);
                        
                    int tilenum = s/S0;
                        
                    if( tilenum == s1 )
                    {
                        // When inserting, the insert function expects
                        // the coordinates align with their dimension
                        adjMat_csf->Insert( {s%S0, d, s1}, ValueArray->At(p) );

                        outDegree->At({s})++;
                        inDegree->At({d})++;
                    }
                    else 
                    {
                        DeltaArray->At(d) += (p - pos_start);                                    
                        p = pos_end;
                    }
                }
            }
        }

        adjMat_csf->TrimFat();
    }
    
    // CSF 4 expects tile sizes in format:  (S0, D0, S1, D1)
    void CreateCSF_4_CSC( const vector<int>& tile_sizes_ )
    {
        int D1 = tile_sizes_[3];
        int S1 = tile_sizes_[2];
        int D0 = tile_sizes_[1];
        int S0 = tile_sizes_[0];

        // Init Delta Array
        InitDeltaArray();

        for(int d1=0; d1<D1; d1++)
        {
            for(int s1=0; s1<S1; s1++)
            {
                for(int d0=0; d0<D0; d0++)
                {
                    int d = d1*D0 + d0;
                
                    if( d < V )
                    {
                    
                        int pos_start = SegmentArray->At(d) + DeltaArray->At(d);
                        int pos_end = SegmentArray->At(d+1);

                        for(int p=pos_start; p<pos_end; p++)
                        {
                            int s = CoordinateArray->At(p);
                        
                            int tilenum = s/S0;
                        
                            if( tilenum == s1 )
                            {
//                                 cout<<"Inserting: "<<d1<<","<<s1<<","<<d0<<","<<s%S0<<" ("<<d<<","<<s<<") = "<<ValueArray->At(p)<<endl;

                                // When inserting, the insert function expects
                                // the coordinates align with their dimension
                                adjMat_csf->Insert( {s%S0, d0, s1, d1}, ValueArray->At(p) );

                                outDegree->At({s})++;
                                inDegree->At({d})++;
                            }
                            else 
                            {
                                DeltaArray->At(d) += (p - pos_start);                                    
                                p = pos_end;
                            }
                        }
                    }
                }
            }
        }

        adjMat_csf->TrimFat();
    }

    // CSF 4 expects tile sizes in format:  (S0, D0, S1, D1)
    void CreateCSF_5_CSC_DstTileStationary( const vector<int>& tile_sizes_ )
    {
        int D1 = tile_sizes_[4];
        int S2 = tile_sizes_[3];
        int S1 = tile_sizes_[2];
        int D0 = tile_sizes_[1];
        int S0 = tile_sizes_[0];

        // instantiate the CSF
        adjMat_csf      = new CSF( tile_sizes_, {"S0","D0","S1","S2","D1"}, E );


//         cout<<"D1: "<<D1<<" S2: "<<S2<<" S1: "<<S1<<" D0: "<<D0<<" S0: "<<S0<<endl;

        // Init Delta Array
        InitDeltaArray();

        for(int d1=0; d1<D1; d1++)
        {
            for(int s2=0; s2<S2; s2++)
            {
                for(int s1=0; s1<S1; s1++)
                {
                    for(int d0=0; d0<D0; d0++)
                    {
                        int d = d1*D0 + d0;
                
                        if( d < V )
                        {
                    
                            int pos_start = SegmentArray->At(d) + DeltaArray->At(d);
                            int pos_end = SegmentArray->At(d+1);

                            for(int p=pos_start; p<pos_end; p++)
                            {
                                int s = CoordinateArray->At(p);
                        
                                int s2_tile = s/(S0*S1);
                                int s1_tile = (s/S0)%S1;
                        
                                if( (s1_tile == s1) && (s2_tile == s2) )
                                {
//                                     cout<<"Inserting: "<<d1<<","<<s2<<","<<s1<<","<<d0<<","<<s%S0<<" ("<<d<<","<<s<<") = "<<ValueArray->At(p)<<" ("<<(d1*D0+d0)<<","<<(s2*S1+s1*S0+(s%S0))<<")"<<endl;

                                    // When inserting, the insert function expects
                                    // the coordinates align with their dimension
                                    adjMat_csf->Insert( {s%S0, d0, s1, s2, d1}, ValueArray->At(p) );

                                    outDegree->At({s})++;
                                    inDegree->At({d})++;

                                }
                                else 
                                {
//                                     cout<<"\tSkipping: "<<d1<<","<<s2<<","<<s1%S1<<","<<d0<<","<<s%S0<<" ("<<d<<","<<s<<") = "<<ValueArray->At(p)<<endl;
                                    DeltaArray->At(d) += (p - pos_start);
                                    p = pos_end;
                                }
                            }
                        }
                    }
                }
            }
        }

        adjMat_csf->TrimFat();
    }

    void CreateCSF_5_CSR_DstTileStationary( const vector<int>& tile_sizes_ )
    {
        int D1 = tile_sizes_[4];
        int S2 = tile_sizes_[3];
        int S1 = tile_sizes_[2];
        int S0 = tile_sizes_[1];
        int D0 = tile_sizes_[0];

        // instantiate the CSF
        adjMat_csf      = new CSF( tile_sizes_, {"D0","S0","S1","S2","D1"}, E );

        // Init Delta Array
        InitDeltaArray();

        for(int d1=0; d1<D1; d1++)
        {
            for(int s2=0; s2<S2; s2++)
            {
                for(int s1=0; s1<S1; s1++)
                {
                    for(int s0=0; s0<S0; s0++)
                    {
                        int s = s2*S1*S0 + s1*S0 + s0;
                
                        if( s < V )
                        {
                            int pos_start = SegmentArray->At(s) + DeltaArray->At(s);
                            int pos_end = SegmentArray->At(s+1);

                            for(int p=pos_start; p<pos_end; p++)
                            {
                                int d = CoordinateArray->At(p);
                        
                                int d1_of_p = (d/D0);
                        
                                if( d1_of_p == d1 )
                                {
                                    // When inserting, the insert function expects
                                    // the coordinates align with their dimension
                                    adjMat_csf->Insert( {d%D0, s0, s1, s2, d1}, ValueArray->At(p) );

                                    outDegree->At({s})++;
                                    inDegree->At({d})++;
                                }
                                else 
                                {
                                    DeltaArray->At(s) += (p - pos_start);
                                    p = pos_end;
                                }
                            }
                        }
                    }
                }
            }
        }

        adjMat_csf->TrimFat();
    }

    void CreateCSF_5_CSR_DstTile_CT_Stationary( const vector<int>& tile_sizes_ )
    {
        int S2 = tile_sizes_[4];
        int D1 = tile_sizes_[3];
        int S1 = tile_sizes_[2];
        int S0 = tile_sizes_[1];
        int D0 = tile_sizes_[0];

        // instantiate the CSF
        adjMat_csf      = new CSF( tile_sizes_, {"D0","S0","S1","D1","S2"}, E );

        // Init Delta Array
        InitDeltaArray();

        for(int s2=0; s2<S2; s2++)
        {
            for(int d1=0; d1<D1; d1++)
            {
                for(int s1=0; s1<S1; s1++)
                {
                    for(int s0=0; s0<S0; s0++)
                    {
                        int s = s2*S1*S0 + s1*S0 + s0;
                
                        if( s < V )
                        {
                            int pos_start = SegmentArray->At(s) + DeltaArray->At(s);
                            int pos_end = SegmentArray->At(s+1);

                            for(int p=pos_start; p<pos_end; p++)
                            {
                                int d = CoordinateArray->At(p);
                        
                                int d1_of_p = (d/D0);
                        
                                if( d1_of_p == d1 )
                                {
                                    // When inserting, the insert function expects
                                    // the coordinates align with their dimension
                                    adjMat_csf->Insert( {d%D0, s0, s1, d1, s2}, ValueArray->At(p) );

                                    outDegree->At({s})++;
                                    inDegree->At({d})++;
                                }
                                else 
                                {
                                    DeltaArray->At(s) += (p - pos_start);
                                    p = pos_end;
                                }
                            }
                        }
                    }
                }
            }
        }

        adjMat_csf->TrimFat();
    }

    int GetCSFstorage()
    {
        return adjMat_csf->GetStorage();
    }
    

    // Adjacency Matrix Tile sizes are expected in format:  (S0, D0, S1, D1)
    void SetTileSizes ( const vector<int>& tile_sizes_, FORMAT_TYPE in_format, STATIONARY_TYPE type_in  )
    {
        int S0 = tile_sizes_[0];
        int D0 = tile_sizes_[1];
        int S1 = tile_sizes_[2];
        int D1 = tile_sizes_[3];
        int S2 = tile_sizes_[4];
        int D2 = tile_sizes_[5];

        if( IsDstTileStationary(type_in) ) 
        {
            if( in_format == FORMAT_CSR )
            {
                if( type_in == STATIONARY_DST_TILE ) 
                    CreateCSF_5_CSR_DstTileStationary( {D0,S0,S1,S2,D1} );
                else if( type_in == STATIONARY_DST_TILE_AND_CT )
                    CreateCSF_5_CSR_DstTile_CT_Stationary( {D0,S0,S1,D1,S2} );
            }
            else if( in_format == FORMAT_CSC )
            {
                CreateCSF_5_CSC_DstTileStationary( {S0,D0,S1,S2,D1} );
            }
        }
        else 
        {
            assert(0);
        }
    }

    ~GraphAlgorithm()
    {
        delete adjMat_csf;
    }

    void PrintInfo()
    {
        cout<<endl<<endl;
        cout<<"<Graph Algorithm Name>"<<endl;
        cout<<"==========================================="<<endl;
        cout<<endl;
        cout<<"Graph Attributes:"<<endl;
        cout<<"\tGraph Name:                "<<SegmentArray->GetFileName()<<endl;
        cout<<"\tNumber of Vertices:        "<<V<<endl;
        cout<<"\tNumber of Edges:           "<<E<<endl;
        cout<<"\tInput Storage (MB):        "<<((double)(((1*(V+1))+E)*8)/(double)MEGA)<<endl;
        cout<<"\tAverage Degree:            "<<(double)E / (double)V<<endl;
        cout<<endl;
        if( adjMat_csf ) 
        {
            cout<<"Tiling Parameters:"<<endl;
            cout<<"\tAm Tiled?                  "<<((adjMat_csf->numDims>2)?"YES":"NO")<<endl;
            cout<<endl;
            cout<<"\tTile Size  ("<<adjMat_csf->dim_names[0]<<"):           "<<adjMat_csf->dim_sizes[0]<<endl;
            cout<<"\tTile Size  ("<<adjMat_csf->dim_names[1]<<"):           "<<adjMat_csf->dim_sizes[1]<<endl;
            cout<<endl;
            cout<<"\t# of Tiles ("<<adjMat_csf->dim_names[2]<<"):           "<<adjMat_csf->dim_sizes[2]<<endl;
            cout<<"\t# of Tiles ("<<adjMat_csf->dim_names[3]<<"):           "<<adjMat_csf->dim_sizes[3]<<endl;
            cout<<"\t# of Tiles ("<<adjMat_csf->dim_names[4]<<"):           "<<adjMat_csf->dim_sizes[4]<<endl;
            cout<<"\tTot # of Tiles:            "<<(adjMat_csf->dim_sizes[2]*adjMat_csf->dim_sizes[3]*adjMat_csf->dim_sizes[4])<<endl;
            cout<<"\tCSF Storage (MB):          "<<((double)(adjMat_csf->GetStorage()*8)/(double)MEGA)<<endl;

            cout<<endl;
            adjMat_csf->PrintCSF(0);
        }
    }

    void Validate()
    {
//         // Validate
//         for(int v=0; v<V; v++)
//         {
//             DataType_t neighbors        = outDegree->At({v});
//             DataType_t whoop_neighbors  = dstData->At({v});
// 
//             if( neighbors != whoop_neighbors )
//             {
// //                 cout<<"v: "<<v<<" Neighbors: "<<neighbors<<" Whoop: "<<whoop_neighbors<<" srcData: "<<srcData->At({v})<<endl;
//                 cout<<"Validation Failed"<<endl;
//                 exit(0);
//             }
//             else 
//             {
// //                 cout<<"v: "<<v<<" Neighbors: "<<neighbors<<" Whoop: "<<whoop_neighbors<<" srcData: "<<srcData->At({v})<<endl;
//             }
//             
// 
//             // Reset it
//             dstData->At({v}) = 0;
//         }
    }

    void Whoop_Untiled( int seed, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int FRONTIER_BUFFET          = LLB_SIZE;
        int RESIDUAL_BUFFET          = 1;
        int RESIDUAL_PRIME_BUFFET    = L1_SIZE;
        int OUTDEGREE_BUFFET        = L1_SIZE;
        int PAGERANK_BUFFET          = 1;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p");
        Var iters;
        
        Vec frontier_size("frontier_size");
        frontier_size.Resize({1});

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_empty = 1;

        // Initialize The State
        for(int i=0; i<V; i++)
        {
            residual_prime->At({i}) = 0;
            residual->At({i}) = 0;
            pageRank->At({i}) = 0;
            frontier->At({i}) = 0;
        }

        // Sett Up For Seed In
        frontier->At({seed}) = 1;
        residual->At({seed}) = 1;
        residual_prime->At({seed}) = 1;

        //////////////////////////////////////////
        // Set Up The Whoop State And Start Runs
        //////////////////////////////////////////

        frontier_empty       = 0;
        frontier_size[0]     = 1;
        iters = 0;
        
        w_while( frontier_empty == 0 );
        {
            iters += 1;

//             pageRank->AddTileLevel( PAGERANK_BUFFET, PAGERANK_BUFFET, BUFFET_LINE_SIZE );
//             residual->AddTileLevel( RESIDUAL_BUFFET, RESIDUAL_BUFFET, BUFFET_LINE_SIZE );
            
            // Update The Page Rank
            t_for(v, 0, V);
            {
                w_if( (*frontier)[v] == 1 );
                {
                    (*pageRank)[v] += weight1 * (*residual)[v];
                    (*residual_prime)[v] = 0;
                }
                end();
            }
            end();

            // Propogate The Residuals To Neighbors
            t_for(s, 0, V);
            {
                w_if( (*frontier)[s] == 1 );
                {
                    update =  weight2 * (*residual)[s] / (*outDegree)[s];

                    pos_start = (*SegmentArray)[s];
                    pos_end   = (*SegmentArray)[s+1];

                    t_for(p,pos_start,pos_end);
                    {
                        d = (*CoordinateArray)[p];
                        (*residual_prime)[d] += update;
                    }
                    end();
                }
                end();
            }
            end();

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size[0]  = 0;
            
            t_for(v, 0, V);
            {
                // copy the update residuals
                (*residual)[v] = (*residual_prime)[v];

                // Generate the new frontier
                w_if( (*outDegree)[v] && ((*residual)[v] >= ((*outDegree)[v] * epsilon)) );
                {
                    (*frontier)[v]    = 1;
                    frontier_empty    = 0;
                    frontier_size[0] += 1;
                }
                w_else();
                {
                    (*frontier)[v] = 0;
                }
                end();
            }
            end();

#ifdef ERROR_CHECK
            /////////////////////////////////////////////////////
            ////  THE BELOW IS FOR DEBUGGING TO MATCH STATE    //
            /////////////////////////////////////////////////////
            w_if( (iters == 1) && (frontier_size[0] != 95) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 2) && (frontier_size[0] != 1682) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 3) && (frontier_size[0] != 2040) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 11) && (frontier_size[0] != 113) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 8) && (frontier_size[0] != 348) );
            {
                frontier_empty = 1;
            }
            end();
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
#endif

            w_if( (iters > ARG_MAX_ITERATIONS) );
            {
                frontier_empty = 1;
            }
            end();

        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Iterations: "<<iters.Access(0,0)<<" Frontier Size: "<<frontier_size.At(0)<<std::endl;

        whoop::Done();
    }

    void Whoop_Untiled_Compressed_TraceFrontierGeneration()
    {
        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        
        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");

        // Generate The New Frontier
        frontier_empty    = 1;
        frontier_cnt      = 0;
            
        t_for(v, 0, V);
        {

            // Assume everything fits on chip LLB
            residual->AddTileLevel( V );
            residual_prime->AddTileLevel( V );
            outDegree->AddTileLevel( V );
            frontier->AddTileLevel( V );
            SegmentArray->AddTileLevel( V+1 );
            CoordinateArray->AddTileLevel( E );

            // copy the update residuals
//             (*residual)[v] = (*residual_prime)[v] + (*residual)[v]*0;

            // Generate the new frontier
            w_if( (*outDegree)[v] && ((*residual_prime)[v] >= ((*outDegree)[v] * epsilon)) );
            {
                (*frontier)[frontier_cnt] = v; // + (*frontier)[frontier_cnt]*0;
                frontier_empty    = 0;
                frontier_cnt     += 1;
            }
            end();
        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;
        cout<< "Generated Frontier Size: "<<frontier_cnt.Access(0,0)<<endl;
        
        whoop::Done();
    }

    void Whoop_Untiled_Compressed_TracePRComp( int neighbors )
    {
        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");

        
//         Vec frontier_size("frontier_size");
// 
//         for(int v=0; v<neighbors; v++) 
//         {
//             frontier->At({v})          = rand()%V;
//         }

        std::cout<<"Tracing Iteration With Neighbors: "<<neighbors<<endl;

        frontier_cnt = neighbors;

        // Update The Page Rank
        t_for(p, 0, frontier_cnt);
        {

            // Assume everything fits on chip LLB
            residual->AddTileLevel( V );
            residual_prime->AddTileLevel( V );
            outDegree->AddTileLevel( V );
            frontier->AddTileLevel( V );
            SegmentArray->AddTileLevel( V+1 );
            CoordinateArray->AddTileLevel( E );

            v = (*frontier)[p];
            (*pageRank)[v] += weight1 * (*residual)[v];
            (*residual_prime)[v] = 0 + (*residual_prime)[v]*0;
        }
        end();


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;
        cout<< "Generated Frontier Size: "<<frontier_cnt.Access(0,0)<<endl;
        
        whoop::Done();
    }    


    void Whoop_Untiled_Compressed_TraceIter( int neighbors )
    {
        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");

        
//         Vec frontier_size("frontier_size");
// 
//         for(int v=0; v<neighbors; v++) 
//         {
//             frontier->At({v})          = rand()%V;
//         }

        std::cout<<"Tracing Iteration With Neighbors: "<<neighbors<<endl;

        frontier_cnt = neighbors;

        // Propogate The Residuals To Neighbors
        t_for(q, 0, frontier_cnt);
        {
            // Assume everything fits on chip LLB
            residual->AddTileLevel( V );
            residual_prime->AddTileLevel( V );
            outDegree->AddTileLevel( V );
            frontier->AddTileLevel( V );
            SegmentArray->AddTileLevel( V+1 );
            CoordinateArray->AddTileLevel( E );

            s = (*frontier)[q];
            update = weight2 * (*residual)[s] / (*outDegree)[s];
            
            pos_start = (*SegmentArray)[s];
            pos_end   = (*SegmentArray)[s+1];
                
            t_for(p,pos_start,pos_end);
            {
                d = (*CoordinateArray)[p];
                (*residual_prime)[d] += update;
            }
            end();
        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Neighbors: "<<neighbors<<std::endl;

        whoop::Done();

    }
    
    void Whoop_Untiled_Compressed( int seed, int RF_KB, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int RF_SIZE                  = (RF_KB*KILO        / BYTES_PER_VERTEX);
        int L1_SIZE                  = (BufferL1_KB*KILO  / BYTES_PER_VERTEX);
        int LLB_SIZE                 = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int FRONTIER_BUFFET          = LLB_SIZE;
        int RESIDUAL_BUFFET          = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int RESIDUAL_PRIME_BUFFET    = L1_SIZE;
        int OUTDEGREE_BUFFET         = L1_SIZE;
        int PAGERANK_BUFFET          = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int SEG_ARRAY_BUFFET         = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET       = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");

        
        Vec frontier_size("frontier_size");
        
        frontier_size.Resize({1});

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_empty = 1;

        // Initialize The State
        for(int i=0; i<V; i++)
        {
            residual_prime->At({i}) = 0;
            residual->At({i}) = 0;
            pageRank->At({i}) = 0;
            frontier->At({i}) = 0;
        }

        // Set Up For Seed In
        frontier->At({0})          = seed;
        residual->At({seed})       = 1;
        residual_prime->At({seed}) = 1;

        //////////////////////////////////////////
        // Set Up The Whoop State And Start Runs
        //////////////////////////////////////////

        frontier_empty       = 0;
        frontier_cnt         = 1;
        iters                = 0;


        // Assume everything fits on chip LLB
        pageRank->SetBackingGranularity(8);
        residual->SetBackingGranularity(8);
        residual_prime->SetBackingGranularity(8);
        outDegree->SetBackingGranularity(8);
        frontier->SetBackingGranularity(8);
//         dstData->SetBackingGranularity(8);

        SegmentArray->SetBackingGranularity(8);
        CoordinateArray->SetBackingGranularity(8);

        
        w_while( frontier_empty == 0 );
        {
            iters += 1;
//             (*dstData)[0] += iters;

            // Assume everything fits on chip LLB
            pageRank->AddTileLevel( V );
            residual->AddTileLevel( V );
            residual_prime->AddTileLevel( V );
            outDegree->AddTileLevel( V );
            frontier->AddTileLevel( V );
//             dstData->AddTileLevel(8);
            SegmentArray->AddTileLevel( V+1 );
            CoordinateArray->AddTileLevel( E );

            // Update The Page Rank
            t_for(p, 0, frontier_cnt);
            {
                v = (*frontier)[p];
                (*pageRank)[v] += weight1 * (*residual)[v];
                (*residual_prime)[v] = 0 + (*residual_prime)[v]*0;
            }
            end();

            // Propogate The Residuals To Neighbors
            t_for(q, 0, frontier_cnt);
            {
                s = (*frontier)[q];

                pos_start = (*SegmentArray)[s];
                pos_end   = (*SegmentArray)[s+1];
                
                t_for(p,pos_start,pos_end);
                {
                    d = (*CoordinateArray)[p];
                    (*residual_prime)[d] += weight2 * (*residual)[s] / (*outDegree)[s];
                }
                end();
            }
            end();

            // Generate The New Frontier
            frontier_empty    = 1;
            frontier_cnt      = 0;
            
            t_for(v, 0, V);
            {
                // copy the update residuals
                (*residual)[v] = (*residual_prime)[v] + (*residual)[v]*0;

                // Generate the new frontier
                w_if( (*outDegree)[v] && ((*residual)[v] >= ((*outDegree)[v] * epsilon)) );
                {
                    (*frontier)[frontier_cnt] = v + (*frontier)[frontier_cnt]*0;
                    frontier_empty    = 0;
                    frontier_cnt     += 1;
                }
                end();
            }
            end();

#ifdef ERROR_CHECK
            /////////////////////////////////////////////////////
            ////  THE BELOW IS FOR DEBUGGING TO MATCH STATE    //
            /////////////////////////////////////////////////////
            w_if( (iters == 1) && (frontier_cnt != 95) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 2) && (frontier_cnt != 1682) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 3) && (frontier_cnt != 2040) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 11) && (frontier_cnt != 113) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 8) && (frontier_cnt != 348) );
            {
                frontier_empty = 1;
            }
            end();
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
#endif

            w_if( (iters > ARG_MAX_ITERATIONS) );
            {
                frontier_empty = 1;
            }
            end();

        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Iterations: "<<iters.Access(0,0)<<" Frontier Size: "<<frontier_cnt.Access(0,0)<<" max: "<<ARG_MAX_ITERATIONS<<std::endl;

        whoop::Done();
    }

    #define VIVALDI_NUM_CT    4
    #define VIVALDI_NUM_DOT_C 2
    void Whoop_Untiled_Compressed_Parallel( int seed, int RF_KB, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int RF_SIZE                  = (RF_KB*KILO        / BYTES_PER_VERTEX);
        int L1_SIZE                  = (BufferL1_KB*KILO  / BYTES_PER_VERTEX);
        int LLB_SIZE                 = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int FRONTIER_BUFFET          = LLB_SIZE;
        int RESIDUAL_BUFFET          = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int RESIDUAL_PRIME_BUFFET    = L1_SIZE;
        int OUTDEGREE_BUFFET         = L1_SIZE;
        int PAGERANK_BUFFET          = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int SEG_ARRAY_BUFFET         = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET       = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        cout<<endl;

        Var D1("D1"), S4("S4"), S3("S3"), S2("S2"), S1("S1"), D0("D0"), S0("S0"), tmp("tmp");
        Var d1("d1"), s4("s4"), s3("s3"), s2("s2"), s1("s1"), d0("d0"), s0("s0");


        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");
        
        Vec frontier_size("frontier_size");
        frontier_size.Resize({1});

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_empty = 1;

        // Initialize The State
        for(int i=0; i<V; i++)
        {
            residual_prime->At({i}) = 0;
            residual->At({i}) = 0;
            pageRank->At({i}) = 0;
            frontier->At({i}) = 0;
        }

        // Set Up For Seed In
        frontier->At({0})          = seed;
        residual->At({seed})       = 1;
        residual_prime->At({seed}) = 1;

        //////////////////////////////////////////
        // Set Up The Whoop State And Start Runs
        //////////////////////////////////////////

        S1 = VIVALDI_NUM_CT;
        S2 = VIVALDI_NUM_DOT_C;

        frontier_empty       = 0;
        frontier_cnt         = 1;
        iters                = 0;

        // Assume everything fits on chip LLB
        pageRank->SetBackingGranularity(8);
        residual->SetBackingGranularity(8);
        residual_prime->SetBackingGranularity(8);
        outDegree->SetBackingGranularity(8);
        frontier->SetBackingGranularity(8);
//         dstData->SetBackingGranularity(8);

        SegmentArray->SetBackingGranularity(8);
        CoordinateArray->SetBackingGranularity(8);
        
        w_while( frontier_empty == 0 );
        {
            iters += 1;
//             (*dstData)[0] += iters;

            // Assume everything fits on chip LLB
            pageRank->AddTileLevel( V, V, 8, 1024 );
            residual->AddTileLevel( V, V, 8, 1024 );
            residual_prime->AddTileLevel( V, V, 8, 1024 );
            outDegree->AddTileLevel( V, V, 8, 1024 );
            frontier->AddTileLevel( V, V, 8, 1024 );
//             dstData->AddTileLevel(1, 1, 8, 1024);

            SegmentArray->AddTileLevel( V+1, V+1, 8, 1024 );
            CoordinateArray->AddTileLevel( E, E, 8, 1024 );

            // Update The Page Rank
            t_for(p, 0, frontier_cnt);
            {
                v = (*frontier)[p];
                (*pageRank)[v] += weight1 * (*residual)[v];
                (*residual_prime)[v] = 0 + (*residual_prime)[v]*0;
            }
            end();


            // Divide the work equally among all the compute elements

            S0 = frontier_cnt / (VIVALDI_NUM_DOT_C * VIVALDI_NUM_CT);
            S0 = S0 & S0;
            w_if( S0 < 1 );
            {
                S0 += 1;
            }
            end();

            S3 = frontier_cnt / (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT);
            S3 = S3 & S3;
            w_if( frontier_cnt % (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT) );
            {
                S3 += 1;
            }
            end();
            
            // Propogate The Residuals To Neighbors
            t_for(s3, 0, S3);
            {
                s_for(s2, 0, VIVALDI_NUM_DOT_C);
                {
                    s_for(s1, 0, VIVALDI_NUM_CT);
                    {
                        t_for(s0, 0, S0);
                        {
                            q = s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;

                            w_if( q < frontier_cnt);
                            {
                                s = (*frontier)[q];

                                pos_start = (*SegmentArray)[s];
                                pos_end   = (*SegmentArray)[s+1];
                
                                t_for(p,pos_start,pos_end);
                                {
                                    d = (*CoordinateArray)[p];
                                    (*residual_prime)[d] += weight2 * (*residual)[s] / (*outDegree)[s];
                                }
                                end();
                            }
                            end();
                            
                        }
                        end();
                    }
                    end();
                }
                end();
            }
            end();

            // Generate The New Frontier
            frontier_empty    = 1;
            frontier_cnt      = 0;
            
            t_for(v, 0, V);
            {
                // copy the update residuals
                (*residual)[v] = (*residual_prime)[v] + (*residual)[v]*0;

                // Generate the new frontier
                w_if( (*outDegree)[v] && ((*residual)[v] >= ((*outDegree)[v] * epsilon)) );
                {
                    (*frontier)[frontier_cnt] = v + (*frontier)[frontier_cnt]*0;
                    frontier_empty    = 0;
                    frontier_cnt     += 1;
                }
                end();
            }
            end();

#ifdef ERROR_CHECK
            /////////////////////////////////////////////////////
            ////  THE BELOW IS FOR DEBUGGING TO MATCH STATE    //
            /////////////////////////////////////////////////////
            w_if( (iters == 1) && (frontier_cnt != 95) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 2) && (frontier_cnt != 1682) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 3) && (frontier_cnt != 2040) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 11) && (frontier_cnt != 113) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 8) && (frontier_cnt != 348) );
            {
                frontier_empty = 1;
            }
            end();
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
#endif

            w_if( (iters > ARG_MAX_ITERATIONS) );
            {
                frontier_empty = 1;
            }
            end();

        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
//         BindCompute(2, 0, "DOTML2");
//         BindCompute(2, 1, "DOTML2");
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Iterations: "<<iters.Access(0,0)<<" Frontier Size: "<<frontier_cnt.Access(0,0)<<std::endl;
        std::cout<<"S3: "<<S3.Access(0,0)<<" ";
        std::cout<<"S2: "<<S2.Access(0,0)<<" ";
        std::cout<<"S1: "<<S1.Access(0,0)<<" ";
        std::cout<<"S0: "<<S0.Access(0,0)<<" ";
        std::cout<<endl;

        whoop::Done();
    }


    void Whoop_Untiled_Compressed_Parallel_TraceIter( int neighbors )
    {
        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var d1("d1"), s4("s4"), s3("s3"), s2("s2"), s1("s1"), d0("d0"), s0("s0");
        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");


        int S0 = neighbors / (VIVALDI_NUM_DOT_C * VIVALDI_NUM_CT);

        if( S0 < 1 ) S0++;
        int S1 = VIVALDI_NUM_CT;
        int S2 = VIVALDI_NUM_DOT_C;
        int S3 = neighbors / (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT);
        if( neighbors % (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT) ) 
        {
            S3 += 1;
        }
        
        std::cout<<"Parallel Tracing Iteration With Neighbors: "<<neighbors<<endl;

        frontier_cnt = neighbors;

        // Propogate The Residuals To Neighbors
        t_for(s3, 0, S3);
        {
            // Assume everything fits on chip LLB
            residual->AddTileLevel( V );
            residual_prime->AddTileLevel( V );
            outDegree->AddTileLevel( V );
            frontier->AddTileLevel( V );
            SegmentArray->AddTileLevel( V+1 );
            CoordinateArray->AddTileLevel( E );

            s_for(s2, 0, VIVALDI_NUM_DOT_C);
            {
                s_for(s1, 0, VIVALDI_NUM_CT);
                {
                    frontier->AddTileLevel( S0 );

                    t_for(s0, 0, S0);
                    {
                        q = s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;

                        w_if( q < frontier_cnt);
                        {
                            s = (*frontier)[q];

                            update = weight2 * (*residual)[s] / (*outDegree)[s];

                            pos_start = (*SegmentArray)[s];
                            pos_end   = (*SegmentArray)[s+1];
                
                            t_for(p,pos_start,pos_end);
                            {
                                d = (*CoordinateArray)[p];
                                (*residual_prime)[d] += update;
                            }
                            end();
                        }
                        end();
                            
                    }
                    end();
                }
                end();
            }
            end();
        }
        end();

        
//         // Propogate The Residuals To Neighbors
//         t_for(q, 0, frontier_cnt);
//         {
// 
//             // Assume everything fits on chip LLB
//             residual->AddTileLevel( V );
//             residual_prime->AddTileLevel( V );
//             outDegree->AddTileLevel( V );
//             frontier->AddTileLevel( V );
//             SegmentArray->AddTileLevel( V+1 );
//             CoordinateArray->AddTileLevel( E );
// 
//             t_for(p, (*SegmentArray)[ (*frontier)[q] ] , (*SegmentArray)[ (*frontier)[q]+1]);
//             {
//                 (*residual_prime)[ (*CoordinateArray)[p] ] += (*residual)[ (*frontier)[q] ] / (*outDegree)[ (*frontier)[q] ];
//             }
//             end();
//         }
//         end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Neighbors: "<<neighbors<<std::endl;

        whoop::Done();

    }


    void Whoop_Untiled_Compressed_Parallel_TraceISTAIter( int neighbors )
    {
        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var d1("d1"), s4("s4"), s3("s3"), s2("s2"), s1("s1"), d0("d0"), s0("s0");
        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q");
        Var iters("iters");


        int S0 = neighbors / (VIVALDI_NUM_DOT_C * VIVALDI_NUM_CT);

        if( S0 < 1 ) S0++;
        int S1 = VIVALDI_NUM_CT;
        int S2 = VIVALDI_NUM_DOT_C;
        int S3 = neighbors / (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT);
        if( neighbors % (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT) ) 
        {
            S3 += 1;
        }
        
        std::cout<<"ISTA Tracing Creating Neighbors: "<<neighbors<<endl;

        CalculateDegrees();

        frontier_cnt = neighbors;

        for(int v=0; v<neighbors; v++) 
        {
            frontier->At({v})          = rand()%V;
        }


        // Propogate The Residuals To Neighbors
        t_for(s3, 0, S3);
        {
            // Assume everything fits on chip LLB
            residual->AddTileLevel( V );
            residual_prime->AddTileLevel( V );
            outDegree->AddTileLevel( V );
            frontier->AddTileLevel( V );
            SegmentArray->AddTileLevel( V+1 );
            CoordinateArray->AddTileLevel( E );
            ValueArray->AddTileLevel( E );

            s_for(s2, 0, VIVALDI_NUM_DOT_C);
            {
                s_for(s1, 0, VIVALDI_NUM_CT);
                {
//                     frontier->AddTileLevel( S0 );

                    t_for(s0, 0, S0);
                    {
                        q = s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;

                        w_if( q < frontier_cnt);
                        {
                            s = (*frontier)[q];

                            update = weight2 * (*residual)[s] / (*outDegree)[s];

                            pos_start = (*SegmentArray)[s];
                            pos_end   = (*SegmentArray)[s+1];
                
                            t_for(p,pos_start,pos_end);
                            {
                                d = (*CoordinateArray)[p];
                                update = (*ValueArray)[p];
                                (*residual_prime)[d] += update;
                            }
                            end();
                        }
                        end();
                            
                    }
                    end();
                }
                end();
            }
            end();
        }
        end();

        
//         // Propogate The Residuals To Neighbors
//         t_for(q, 0, frontier_cnt);
//         {
// 
//             // Assume everything fits on chip LLB
//             residual->AddTileLevel( V );
//             residual_prime->AddTileLevel( V );
//             outDegree->AddTileLevel( V );
//             frontier->AddTileLevel( V );
//             SegmentArray->AddTileLevel( V+1 );
//             CoordinateArray->AddTileLevel( E );
// 
//             t_for(p, (*SegmentArray)[ (*frontier)[q] ] , (*SegmentArray)[ (*frontier)[q]+1]);
//             {
//                 (*residual_prime)[ (*CoordinateArray)[p] ] += (*residual)[ (*frontier)[q] ] / (*outDegree)[ (*frontier)[q] ];
//             }
//             end();
//         }
//         end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        whoop::Done();

    }

    void Whoop_Tiled_Compressed_Parallel( int seed, int RF_KB, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int RF_SIZE                  = (RF_KB*KILO        / BYTES_PER_VERTEX);
        int L1_SIZE                  = (BufferL1_KB*KILO  / BYTES_PER_VERTEX);
        int LLB_SIZE                 = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int FRONTIER_BUFFET          = LLB_SIZE;
        int RESIDUAL_BUFFET          = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int RESIDUAL_PRIME_BUFFET    = L1_SIZE;
        int OUTDEGREE_BUFFET         = L1_SIZE;
        int PAGERANK_BUFFET          = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int SEG_ARRAY_BUFFET         = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET       = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        Var D1("D1"), S4("S4"), S3("S3"), S2("S2"), S1("S1"), D0("D0"), S0("S0"), tmp("tmp");
        Var d1("d1"), s4("s4"), s3("s3"), s2("s2"), s1("s1"), d0("d0"), s0("s0");

        Var d1_of_p("d1_of_p"), crossed_tile_boundary("crossed_tile_boundary");
        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update"), frontier_cnt("frontier_cnt");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p"), q("q"), iters("iters");

        
        Vec frontier_size("frontier_size");
        frontier_size.Resize({1});

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_empty = 1;

        // Initialize The State
        for(int i=0; i<V; i++)
        {
            residual_prime->At({i}) = 0;
            residual->At({i}) = 0;
            pageRank->At({i}) = 0;
            frontier->At({i}) = 0;
        }

        // Set Up For Seed In
        frontier->At({0})          = seed;
        residual->At({seed})       = 1;
        residual_prime->At({seed}) = 1;

        //////////////////////////////////////////
        // Set Up The Whoop State And Start Runs
        //////////////////////////////////////////
        D0 = RF_SIZE;
        D1 = V/D0;
        w_if( V % D0 );
        {
            D1 += 1;
            D1  = D1 & D1;
        }
        end();

        S1 = VIVALDI_NUM_CT;
        S2 = VIVALDI_NUM_DOT_C;

        frontier_empty       = 0;
        frontier_cnt         = 1;
        iters                = 0;

        // Assume everything fits on chip LLB
        pageRank->SetBackingGranularity(8);
        residual->SetBackingGranularity(8);
        residual_prime->SetBackingGranularity(8);
        outDegree->SetBackingGranularity(8);
        frontier->SetBackingGranularity(8);
//         dstData->SetBackingGranularity(8);
        DeltaArray->SetBackingGranularity(8);
        
        SegmentArray->SetBackingGranularity(8);
        CoordinateArray->SetBackingGranularity(8);
        
        w_while( frontier_empty == 0 );
        {
            iters += 1;
//             (*dstData)[0] += iters;

            // Assume everything fits on chip LLB
            pageRank->AddTileLevel( V ); //, V, 8, 1024 );
            residual->AddTileLevel( V ); //, V, 8, 1024 );
            residual_prime->AddTileLevel( V ); //, V, 8, 1024 );
            outDegree->AddTileLevel( V ); //, V, 8, 1024 );
            frontier->AddTileLevel( V ); //, V, 8, 1024 );
//             dstData->AddTileLevel(1); //, 1, 8, 1024);

            SegmentArray->AddTileLevel( V+1 ); //, V+1, 8, 1024 );
            CoordinateArray->AddTileLevel( E ); //, E, 8, 1024 );
            DeltaArray->AddTileLevel(V+1); //, V+1, 8, 1024 );

            // Update The Page Rank
            t_for(p, 0, frontier_cnt);
            {
                v = (*frontier)[p];
                (*pageRank)[v] += weight1 * (*residual)[v];
                (*residual_prime)[v] = 0 + (*residual_prime)[v]*0;
            }
            end();


            // Divide the work equally among all the compute elements

            S0 = frontier_cnt / (VIVALDI_NUM_DOT_C * VIVALDI_NUM_CT);
            S0 = S0 & S0;
            w_if( S0 < 1 );
            {
                S0 += 1;
            }
            end();

            S4 = 1; // this is dependent on how many S0 tiles can fit in the LLB
            S3 = frontier_cnt / (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT);
            S3 = S3 & S3;
            w_if( frontier_cnt % (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT) );
            {
                S3 += 1;
            }
            end();

            // Propogate The Residuals To Neighbors
            t_for(s4, 0, S4);
            {
                t_for(d1, 0, D1);
                {
                    t_for(s3, 0, S3);
                    {
                        s_for(s2, 0, VIVALDI_NUM_DOT_C);
                        {
                            // What do we store in the DOT-C buffer?
                            residual_prime->AddTileLevel( RF_SIZE ); //, RF_SIZE, 8, 1024 );
                            DeltaArray->AddTileLevel(L1_SIZE ); //, L1_SIZE, 8, 1024);

                            s_for(s1, 0, VIVALDI_NUM_CT);
                            {
                                // What do we store in the CT buffer?
                                residual_prime->AddTileLevel( RF_SIZE ); //, RF_SIZE, 8, 1024 );

                                t_for(s0, 0, S0);
                                {
                                    q = s4*S3*S2*S1*S0 + s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;

                                    w_if( q < frontier_cnt);
                                    {
                                        s = (*frontier)[q];

                                        pos_start = (*SegmentArray)[s] + (*DeltaArray)[s];
                                        pos_end = (*SegmentArray)[s+1];

                                        crossed_tile_boundary = 0;
                                        p = pos_start;
                                        w_while( (p<pos_end) && (crossed_tile_boundary != 1) );
                                        {
                                            d = (*CoordinateArray)[p];
                        
                                            d1_of_p = (d/D0);
                                            d1_of_p = d1_of_p & d1_of_p;
                                
                                            w_if( (d1_of_p == d1) );
                                            {
                                                (*residual_prime)[d] += weight2 * (*residual)[s] / (*outDegree)[s];
                                                p += 1;
                                            }
                                            w_else();
                                            {
                                                crossed_tile_boundary = 1;
                                            }
                                            end();
                                        }
                                        end();
 
                                        (*DeltaArray)[s] += p-pos_start;
                                    }
                                    end();
                            
                                }
                                end();
                            }
                            end();
                        }
                        end();
                    }
                    end();
                }
                end();
            }
            end();

            // Generate The New Frontier
            frontier_empty    = 1;
            frontier_cnt      = 0;
            
            t_for(v, 0, V);
            {
                // copy the update residuals
                (*residual)[v] = (*residual_prime)[v] + (*residual)[v]*0;

                // Reset the delta array
                (*DeltaArray)[v] = 0;

                // Generate the new frontier
                w_if( (*outDegree)[v] && ((*residual)[v] >= ((*outDegree)[v] * epsilon)) );
                {
                    (*frontier)[frontier_cnt] = v + (*frontier)[frontier_cnt]*0;
                    frontier_empty    = 0;
                    frontier_cnt     += 1;
                }
                end();
            }
            end();

#ifdef ERROR_CHECK
            /////////////////////////////////////////////////////
            ////  THE BELOW IS FOR DEBUGGING TO MATCH STATE    //
            /////////////////////////////////////////////////////
            w_if( (iters == 1) && (frontier_cnt != 95) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 2) && (frontier_cnt != 1682) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 3) && (frontier_cnt != 2040) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 11) && (frontier_cnt != 113) );
            {
                frontier_empty = 1;
            }
            end();

            w_if( (iters == 8) && (frontier_cnt != 348) );
            {
                frontier_empty = 1;
            }
            end();
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
#endif


            w_if( (iters > ARG_MAX_ITERATIONS) );
            {
                frontier_empty = 1;
            }
            end();

        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Iterations: "<<iters.Access(0,0)<<" Frontier Size: "<<frontier_cnt.Access(0,0)<<std::endl;
        std::cout<<"S3: "<<S3.Access(0,0)<<" ";
        std::cout<<"S2: "<<S2.Access(0,0)<<" ";
        std::cout<<"S1: "<<S1.Access(0,0)<<" ";
        std::cout<<"S0: "<<S0.Access(0,0)<<" ";
        std::cout<<endl;

        whoop::Done();
    }

    void Untiled( int seed )
    {
        CalculateDegrees();

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({seed}) = 1;
        residual->At({seed}) = 1;
        residual_prime->At({seed}) = 1;
        
        frontier_empty       = 0;
        frontier_size++;
        
        while( frontier_empty == 0 )
        {
            iters++;
            
            // Update The Page Rank
            for(int v=0; v<V; v++)
            {
                if( frontier->At({v}) == 1 )
                {
                    pageRank->At({v}) += weight1 * residual->At({v});
                    residual_prime->At({v}) = 0;
//                     cout<<"\tPage Rank of: "<<v<<" is: "<<pageRank->At({v})<<" weight1: "<<weight1<<" residual: "<<residual->At({v})<<endl;
                }
            }

            // Propogate The Residuals To Neighbors
            for(int s=0; s<V; s++)
            {
                if( frontier->At({s}) == 1 )
                {
                    update =  weight2 * residual->At({s}) / outDegree->At({s});

                    pos_start = SegmentArray->At({s});
                    pos_end   = SegmentArray->At({s+1});

                    for(p=pos_start; p<pos_end; p++)
                    {
                        d = CoordinateArray->At({p});
                        residual_prime->At({d}) += update;

//                         cout<<"\tResidual_prime of: "<<d<<" is: "<<residual_prime->At({d})<<endl;

                    }
                }
            }

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({v}) = 1;
                    frontier_empty = 0;
                    frontier_size++;
                }
                else
                {
                    frontier->At({v}) = 0;
                }
            }

            std::cout<<iters<<" -- Frontier Size: "<<frontier_size<<std::endl;
        }
    }

    void Untiled_Compressed_TraceIter( int seed )
    {
        CalculateDegrees();

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;
        
        while( frontier_empty == 0 )
        {
            iters++;
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;
            }

            // Propogate The Residuals To Neighbors
            for(int q=0; q<frontier_size; q++)
            {
                s = frontier->At({q});

                pos_start = SegmentArray->At({s});
                pos_end   = SegmentArray->At({s+1});

                for(p=pos_start; p<pos_end; p++)
                {
                    d = CoordinateArray->At({p});
                    residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                }
            }

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<std::endl;

            if( iters == ARG_MAX_ITERATIONS )
            {
                Whoop_Untiled_Compressed_TraceIter( frontier_size );
                return;
            }
        }
    }

    void Untiled_Compressed_Parallel_TraceIter( int seed )
    {
        CalculateDegrees();

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;
        
        while( frontier_empty == 0 )
        {
            iters++;
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;
            }

            // Propogate The Residuals To Neighbors
            for(int q=0; q<frontier_size; q++)
            {
                s = frontier->At({q});

                pos_start = SegmentArray->At({s});
                pos_end   = SegmentArray->At({s+1});

                for(p=pos_start; p<pos_end; p++)
                {
                    d = CoordinateArray->At({p});
                    residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                }
            }

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<std::endl;

            if( iters == ARG_MAX_ITERATIONS )
            {
                Whoop_Untiled_Compressed_Parallel_TraceIter( frontier_size );
                return;
            }
        }
    }

    void Untiled_Compressed_Parallel_TraceFrontierGeneration( int seed )
    {
        CalculateDegrees();

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;
        
        while( frontier_empty == 0 )
        {
            iters++;
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;
            }

            // Propogate The Residuals To Neighbors
            for(int q=0; q<frontier_size; q++)
            {
                s = frontier->At({q});

                pos_start = SegmentArray->At({s});
                pos_end   = SegmentArray->At({s+1});

                for(p=pos_start; p<pos_end; p++)
                {
                    d = CoordinateArray->At({p});
                    residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<std::endl;

            if( iters == ARG_MAX_ITERATIONS )
            {
                Whoop_Untiled_Compressed_TraceFrontierGeneration();
                return;
            }

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

        }
    }

    void Untiled_Compressed_Parallel_TracePRComp( int seed )
    {
        CalculateDegrees();

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;
        
        while( frontier_empty == 0 )
        {
            iters++;

            if( iters == ARG_MAX_ITERATIONS )
            {
                Whoop_Untiled_Compressed_TracePRComp( frontier_size );
                return;
            }
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;
            }

            // Propogate The Residuals To Neighbors
            for(int q=0; q<frontier_size; q++)
            {
                s = frontier->At({q});

                pos_start = SegmentArray->At({s});
                pos_end   = SegmentArray->At({s+1});

                for(p=pos_start; p<pos_end; p++)
                {
                    d = CoordinateArray->At({p});
                    residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<std::endl;


            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

        }
    }

    void Untiled_Compressed( int seed )
    {
        CalculateDegrees();

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;
        
        while( frontier_empty == 0 )
        {
            iters++;
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;
            }

            // Propogate The Residuals To Neighbors
            for(int q=0; q<frontier_size; q++)
            {
                s = frontier->At({q});

                pos_start = SegmentArray->At({s});
                pos_end   = SegmentArray->At({s+1});

                for(p=pos_start; p<pos_end; p++)
                {
                    d = CoordinateArray->At({p});
                    residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                }
            }

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<std::endl;

            if( iters > ARG_MAX_ITERATIONS )
            {
                frontier_empty = 1;
            }
        }
    }

    void Untiled_Compressed_Parallel( int seed )
    {
        CalculateDegrees();

        int S1 = VIVALDI_NUM_CT;
        int S2 = VIVALDI_NUM_DOT_C;

        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;
        
        while( frontier_empty == 0 )
        {
            iters++;

//             std::cout<<iters<<" -- Frontier Size: "<<frontier_size<<std::endl;
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;
            }

            int S0 = frontier_size / (VIVALDI_NUM_DOT_C * VIVALDI_NUM_CT);

            if( S0 < 1 ) S0++;

            int parallelCnt = S0*VIVALDI_NUM_CT*VIVALDI_NUM_DOT_C;

            int S3 = frontier_size / (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT);
            if( frontier_size % (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT) ) 
            {
                S3 += 1;
            }
            
            // Propogate The Residuals To Neighbors
            for(int s3=0; s3<S3; s3++) 
            {
                for(int s2=0; s2<VIVALDI_NUM_DOT_C; s2++)
                {
                    for(int s1=0; s1<VIVALDI_NUM_CT; s1++) 
                    {
                        for(int s0=0; s0<S0; s0++) 
                        {
                            int q = s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;
                            
                            if( q < frontier_size ) 
                            {
                                s = frontier->At({q});

                                pos_start = SegmentArray->At({s});
                                pos_end   = SegmentArray->At({s+1});

                                for(p=pos_start; p<pos_end; p++)
                                {
                                    d = CoordinateArray->At({p});
                                    residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                                }
                            }
                        }
                    }
                }
            }

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});
            
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<" \t\tS3: "<<S3<<" S2: "<<S2<<" S1: "<<S1<<" S0: "<<S0<<endl;

            if( iters > ARG_MAX_ITERATIONS )
            {
                frontier_empty = 1;
            }
        }
    }

    void Tiled_Compressed_Parallel( int seed )
    {
        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int S1 = VIVALDI_NUM_CT;
        int S2 = VIVALDI_NUM_DOT_C;
        int D0 = 4096;
        
        int iters = 0;
        int v, s, d, frontier_empty, pos_start, pos_end, p, frontier_size;
        double weight1, weight2, update;

        weight1 = (2*alpha) / (1+alpha);
        weight2 = (1 - alpha) / (1 +  alpha);

        frontier_size  = 0;
        frontier_empty = 1;

        // Initialize For This Seed
        for(int v=0; v<V; v++)
        {
            residual_prime->At({v}) = 0;
            residual->At({v}) = 0;
            pageRank->At({v}) = 0;
            frontier->At({v}) = 0;
        }
        
        // Start Page Rank Computation
        frontier->At({frontier_size}) = seed;
        residual->At({seed})          = 1;
        residual_prime->At({seed})    = 1;
        
        frontier_empty       = 0;
        frontier_size        = 1;

        int ops = 0;
        
        while( frontier_empty == 0 )
        {
            iters++;

//             std::cout<<iters<<" -- Frontier Size: "<<frontier_size<<std::endl;
            
            // Update The Page Rank
            for(int p=0; p<frontier_size; p++)
            {
                v = frontier->At({p});
                pageRank->At({v}) += weight1 * residual->At({v});
                residual_prime->At({v}) = 0;

                ops += 2;
            }

            int S0 = frontier_size / (VIVALDI_NUM_DOT_C * VIVALDI_NUM_CT);

            if( S0 < 1 ) S0++;

            int parallelCnt = S0*VIVALDI_NUM_CT*VIVALDI_NUM_DOT_C;

            int S3 = frontier_size / (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT);
            if( frontier_size % (S0*VIVALDI_NUM_DOT_C*VIVALDI_NUM_CT) ) 
            {
                S3 += 1;
            }

            // Propogate The Residuals To Neighbors
            int D1 = V%D0 ? (V/D0+1) : V/D0;
            for(int d1=0; d1<D1; d1++)
            {
                for(int s3=0; s3<S3; s3++) 
                {
                    for(int s2=0; s2<VIVALDI_NUM_DOT_C; s2++)
                    {
                        for(int s1=0; s1<VIVALDI_NUM_CT; s1++) 
                        {
                            for(int s0=0; s0<S0; s0++) 
                            {
                                int q = s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;
                            
                                if( q < frontier_size ) 
                                {
                                    s = frontier->At({q});
                                    pos_start = SegmentArray->At({s}) + DeltaArray->At({s});
                                    pos_end   = SegmentArray->At({s+1});
                                    double update = weight2 * residual->At({s}) / outDegree->At({s});

                                    ops += 2;
                                    
                                    bool crossed_tile_boundary = 0;
                                    p = pos_start;
                                    
                                    while( (p<pos_end) && (!crossed_tile_boundary) )
                                    {
                                        d = CoordinateArray->At({p});
                                        
                                        int d1_of_p = (d/D0);

                                        if( d1_of_p == d1 ) 
                                        {
//                                             residual_prime->At({d}) += weight2 * residual->At({s}) / outDegree->At({s});
                                            residual_prime->At({d}) += update;
                                            p++;
                                            ops++;
                                        }
                                        else
                                        {
                                            crossed_tile_boundary = 1;
                                        }
                                    }
                                    
                                    DeltaArray->At({s}) += p-pos_start;
                                }
                            }
                        }
                    }
                }
            }
            

            // Generate The New Frontier
            frontier_empty = 1;
            frontier_size  = 0;
            
            for(int v=0; v<V; v++)
            {
                // reset delta array
                DeltaArray->At({v}) = 0;
                
                // copy the update residuals
                residual->At({v}) = residual_prime->At({v});

                if( outDegree->At({v}) ) ops += 2;
                // Generate the new frontier
                if( outDegree->At({v}) && (residual->At({v}) >= (outDegree->At({v}) * epsilon)) )
                {
//                     std::cout<<"\tAdding: "<<v<<" to next frontier"<<endl;
                    frontier->At({frontier_size}) = v;
                    frontier_empty = 0;
                    frontier_size++;
                }
            }

            cout<<"Iteration: "<<iters<<" Frontier Size: "<<frontier_size<<" \t\tS3: "<<S3<<" S2: "<<S2<<" S1: "<<S1<<" S0: "<<S0<<endl;
            
            if( iters > ARG_MAX_ITERATIONS )
            {
                frontier_empty = 1;
            }
        }

        cout<<"Total Ops: "<<ops<<endl;

    }
};



#endif


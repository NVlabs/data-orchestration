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
// #define NUM_DOT_C 32
#define NUM_DOT_C 1
#endif

extern int ARG_BUFFET_GRANULARITY;

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

    CSF*         adjMat_csf;

    // SPMV Specific
    Tensor*      srcData;
    Tensor*      dstData;

    // Page Rank Nibble Specific
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
            srcData->At( {i} ) = 1;
            dstData->At( {i} ) = 0;
        }
    }

    void Init()
    {
        V                = SegmentArray->Size()-1;
        E                = CoordinateArray->Size();
        adjMat_csf       = NULL;
        DeltaArray       = NULL;

        // spmv init
        srcData->Resize( {V} );
        dstData->Resize( {V} );

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

    void CalculateDegrees()
    {
        if( !degreeCalc ) 
        {
            degreeCalc = true;
            
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
    
    GraphAlgorithm( VecIn *segArray, VecIn *coordArray, VecIn *valArray )
    {
        SegmentArray    = segArray;
        CoordinateArray = coordArray;
        ValueArray      = valArray;

        // init SPMV
        srcData   = new Tensor("srcData");
        dstData   = new Tensor("dstData");

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
//     cout<<"\tAverage Degree:            "<<avgDegree<<endl;
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
        // Validate
        for(int v=0; v<V; v++)
        {
            DataType_t neighbors        = outDegree->At({v});
            DataType_t whoop_neighbors  = dstData->At({v});

            if( neighbors != whoop_neighbors )
            {
                cout<<"v: "<<v<<" Neighbors: "<<neighbors<<" Whoop: "<<whoop_neighbors<<" srcData: "<<srcData->At({v})<<endl;
                cout<<"Validation Failed"<<endl;
                exit(0);
            }
            else 
            {
                cout<<"v: "<<v<<" Neighbors: "<<neighbors<<" Whoop: "<<whoop_neighbors<<" srcData: "<<srcData->At({v})<<endl;
            }
            

            // Reset it
            dstData->At({v}) = 0;
        }
    }

    void Whoop_PageRankNibble_Untiled( int seed )
    {
        CalculateDegrees();

        Var v("v"), s("s"), d("d");
        Var weight1("weight1"), weight2("weight2"), frontier_empty("frontier_empty"), update("update");
        

        Var pos_start("pos_start"), pos_end("pos_end"), p("p");

        
        Vec iters("iters"), frontier_size("frontier_size");
        iters.Resize({1});
        iters.At(0) = 0;
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
        
        w_while( frontier_empty == 0 );
        {
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

            iters[0] += 1;

            /////////////////////////////////////////////////////
            ////  THE BELOW IS FOR DEBUGGING TO MATCH STATE    //
            /////////////////////////////////////////////////////
//             w_if( (iters[0] == 1) && (frontier_size[0] != 95) );
//             {
//                 frontier_empty = 1;
//             }
//             end();
// 
//             w_if( (iters[0] == 2) && (frontier_size[0] != 1682) );
//             {
//                 frontier_empty = 1;
//             }
//             end();
// 
//             w_if( (iters[0] == 3) && (frontier_size[0] != 2040) );
//             {
//                 frontier_empty = 1;
//             }
//             end();
// 
//             w_if( (iters[0] == 11) && (frontier_size[0] != 113) );
//             {
//                 frontier_empty = 1;
//             }
//             end();
// 
//             w_if( (iters[0] == 8) && (frontier_size[0] != 348) );
//             {
//                 frontier_empty = 1;
//             }
//             end();
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////
            /////////////////////////////////////////////////////

// 
//             w_if( (iters[0] > 4) );
//             {
//                 frontier_empty = 1;
//             }
//             end();

        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        std::cout<<"Number of Iterations: "<<iters.At(0)<<" Frontier Size: "<<frontier_size.At(0)<<std::endl;

        whoop::Done();
    }

    void PageRankNibble_Untiled( int seed )
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
            std::cout<<iters<<" -- Frontier Size: "<<frontier_size<<std::endl;
            
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
            iters++;
        }
    }

    void Untiled( int X0, int X1, int X2, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            Untiled_CSR( X0, X1, X2);
        }
        else if( format == FORMAT_CSC )
        {
            Untiled_CSC();
        }
    }
    

    void Untiled_CSR( int S0, int S1, int S2 )
    {
        int d, s;

        CalculateDegrees();

        for(int s2=0; s2<S2; s2++)
        {
            for(int s1=0; s1<S1; s1++)
            {
                for(int s0=0; s0<S0; s0++)
                {
                    s = s2*S1*S0 + s1*S0 + s0;
                    
                    if( s < V ) 
                    {
                        int pos_start = SegmentArray->At(s);
                        int pos_end = SegmentArray->At(s+1);
                    
                        for(int p=pos_start; p<pos_end; p++)
                        {
                            d = CoordinateArray->At(p);
                            dstData->At({s}) += srcData->At({d})*3.5;
                        }
                    }
                    
                }
            }
        }
        
        cout<<"\tValidating Results... ";
        Validate();        
        cout<<"DONE!"<<endl;
        cout<<endl;
    }


    void Untiled_CSC()
    {
        int d, s;

        CalculateDegrees();

        for(d=0; d<V; d++)
        {
            int pos_start = SegmentArray->At(d);
            int pos_end = SegmentArray->At(d+1);
            
            for(int p=pos_start; p<pos_end; p++)
            {
                s = CoordinateArray->At(p);
                        
                dstData->At({s}) += srcData->At({d});
            }
        }

        cout<<"\tValidating Results... ";
        Validate();        
        cout<<"DONE!"<<endl;
        cout<<endl;
    }

    void WhoopUntiled( int X0, int X1, int X2, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR )
        {
            WhoopUntiled_CSR( X0, X1, X2, BufferL1_KB, BufferL2_KB );
        }
        else if( format == FORMAT_CSC )
        {
            WhoopUntiled_CSC( BufferL1_KB, BufferL2_KB );
        }
    }


    void WhoopUntiled_CSR( int S0, int S1, int S2, int BufferL1_KB, int BufferL2_KB )
    {

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int D0                 = L1_SIZE;
        int D1                 = (V%D0) ? (V/D0+1) : (V/D0);
        int D2                 = 1;

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<endl;
        cout<<endl;

        Var s2("s2"), s1("s1"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p");

        assert( S1 <= NUM_DOT_C );

        t_for(s2, 0, S2);
        {
            SegmentArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            CoordinateArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );

            s_for(s1,0,NUM_DOT_C);
            {
                dstData->AddTileLevel( DST_DATA_BUFFET, 1, BUFFET_LINE_SIZE );
                SegmentArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                CoordinateArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);

                t_for(s0,0,S0);
                {
                    s = s2*S1*S0 + s1*S0 + s0;

//                     w_if( s < V );
//                     {
                        t_for(p,(*SegmentArray)[s],(*SegmentArray)[s+1]);
                        {
                            d = (*CoordinateArray)[p];
                        
                            (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
                        }
                        end();
//                     }
//                     end();
                }
                end();
            }
            end();
        }
        end();
        
        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void WhoopUntiled_CSC( int BufferL1_KB, int BufferL2_KB )
    {

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int D0                 = L1_SIZE;
        int D1                 = (V%D0) ? (V/D0+1) : (V/D0);
        int D2                 = 1;
        
        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<endl;
        cout<<endl;

        Var d2("d2"), d1("d1"), d0("d0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p");

        t_for(d,0,V);
        {
            SegmentArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            CoordinateArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            
            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE );

            t_for(p,(*SegmentArray)[d],(*SegmentArray)[d+1]);
            {
                s = (*CoordinateArray)[p];
                        
                (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
            }
            end();
        }
        end();

        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void OnlineTiling_CSC_ver2( int D1, int S2, int S1, int D0, int S0 )
    {

        int d1, s2, s1, d0, s0, d, s;

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        for(int s2=0; s2<S2; s2++)
        {
            for(int d1=0; d1<D1; d1++)
            {
                for(int s1=0; s1<S1; s1++)
                {
                    for(int d0=0; d0<D0; d0++)
                    {
                        int d = d1*D0 + d0;
                
                        if( d < V )
                        {
                            int p;
                            int pos_start = SegmentArray->At(d) + DeltaArray->At(d);
                            int pos_end = SegmentArray->At(d+1);

                            bool crossed_tile_boundary = false;

                            for(p=pos_start; p<pos_end; p++)
                            {
                                int s = CoordinateArray->At(p);
                        
                                int s2_tile = s/(S0*S1);
                                int s1_tile = (s/S0)%S1;
                        
                                if( (s1_tile == s1) && (s2_tile == s2) )
                                {
                                    dstData->At({s}) += srcData->At({d});
                                }
                                else 
                                {
                                    crossed_tile_boundary = true;
                                    DeltaArray->At(d) += (p - pos_start);
                                    p = pos_end;
                                }                                
                            }

                            if( crossed_tile_boundary == false )
                            {
                                DeltaArray->At(d) += (p-pos_start);
                            }

                        }
                    }
                }
            }
        }


        // Validate
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }

    void OnlineTiling_CSR_ver2( int D1, int S2, int S1, int D0, int S0 )
    {

        int d1, s2, s1, d0, s0, d, s;

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

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

                            bool crossed_tile_boundary = false;
                            
                            int p = pos_start;
                            while( (p < pos_end) && !crossed_tile_boundary )
                            {
                                int d = CoordinateArray->At(p);
                        
                                int d1_of_p = d/D0;
                        
                                if( (d1_of_p == d1) )
                                {
                                    dstData->At({s}) += srcData->At({d});
                                    p++;
                                }
                                else 
                                {
                                    crossed_tile_boundary = true;
                                }
                            }

                            DeltaArray->At(s) += (p - pos_start);
                        }
                    }
                }
            }
        }


        // Validate
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }

    void WhoopOnlineTiling_CSC_ver2( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB )
    {

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<endl;

        Var d1("d1"), s2("s2"), s1("s1"), d0("d0"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p");
        Var s2_tile("s2_tile");
        Var s1_tile("s1_tile");

        assert( S1 <= NUM_DOT_C );

        t_for(s2,0,S2);
        {            
            t_for(d1,0,D1);
            {
                /**********************************************************************************/
                /******************************* Setup LLC Buffet Sizes ***************************/
                /**********************************************************************************/
                SegmentArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
                DeltaArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );

                CoordinateArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

                srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
                dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                /**********************************************************************************/
                s_for(s1,0,NUM_DOT_C); //t_for(s1,0,S1);
                {
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);
                    DeltaArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    SegmentArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    CoordinateArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    t_for(d0,0,D0);
                    {
                        d = d1*D0 + d0;
                
                        w_if( d < V );
                        {
                    
                            pos_start = (*SegmentArray)[d] + (*DeltaArray)[d];
                            pos_end = (*SegmentArray)[d+1];

                            t_for(p,pos_start,pos_end);
                            {
                                s = (*CoordinateArray)[p];
                        
                                s2_tile = s/(S0*S1);
                                s1_tile = (s/S0)%S1;
                                
                                w_if( (s1_tile == s1) && (s2_tile == s2) );
                                {
                                    (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
                                }
                                w_else();
                                {
                                    w_if( pos_start != p );
                                    {
                                        (*DeltaArray)[d] += (p - pos_start);
                                    }
                                    end();
                                    
                                    p = pos_end;
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void WhoopOnlineTiling_CSC_ver3( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB )
    {

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<endl;

        Var d1("d1"), s2("s2"), s1("s1"), d0("d0"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p"), crossed_tile_boundary("crossed_tile_boundary");
        Var s2_tile("s2_tile");
        Var s1_tile("s1_tile");

        assert( S1 <= NUM_DOT_C );

        t_for(s2,0,S2);
        {            
            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            SegmentArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            DeltaArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            CoordinateArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            ValueArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
            /**********************************************************************************/

            t_for(d1,0,D1);
            {
                s_for(s1,0,S1); //t_for(s1,0,S1);
                {
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);
                    DeltaArray->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE );
                    SegmentArray->AddTileLevel( DST_DATA_BUFFET+1, DST_DATA_BUFFET+1, BUFFET_LINE_SIZE );
                    CoordinateArray->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE );
                    ValueArray->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE );

                    t_for(d0,0,D0);
                    {
                        d = d1*D0 + d0;
                
                        w_if( d < V );
                        {
                    
                            pos_start = (*SegmentArray)[d] + (*DeltaArray)[d];
                            pos_end = (*SegmentArray)[d+1];

                            crossed_tile_boundary = 0;
                            p = pos_start;
                            w_while( (p < pos_end) && (crossed_tile_boundary != 1) );
                            {
                                s = (*CoordinateArray)[p];
                        
                                s2_tile = s/(S0*S1);
                                s1_tile = (s/S0)%S1;
                                
                                w_if( (s1_tile == s1) && (s2_tile == s2) );
                                {
                                    (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
                                    p += 1;
                                }
                                w_else();
                                {
                                    crossed_tile_boundary = 1;
                                }
                                end();
                            }
                            end();

                            (*DeltaArray)[d] += p-pos_start;

//                             t_for(p,pos_start,pos_end);
//                             {
//                                 s = (*CoordinateArray)[p];
//                         
//                                 s2_tile = s/(S0*S1);
//                                 s1_tile = (s/S0)%S1;
//                                 
//                                 w_if( (s1_tile == s1) && (s2_tile == s2) );
//                                 {
//                                     (*dstData)[s] += (*srcData)[d];
//                                     p += 1;
//                                 }
//                                 w_else();
//                                 {
//                                     crossed_tile_boundary = 1;
//                                     (*DeltaArray)[d] += (p-pos_start);
//                                     p = pos_end;
//                                 }
//                                 end();
//                             }
//                             end();
// 
//                              // a while loop will clean this up
//                              w_if( crossed_tile_boundary == 0 );
//                              {
//                                 (*DeltaArray)[d] += p-pos_start;
//                              }
//                              end();

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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void WhoopOnlineTiling_CSR_ver3( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB )
    {

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<endl;

        Var d1("d1"), s2("s2"), s1("s1"), d0("d0"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p"), crossed_tile_boundary("crossed_tile_boundary");
        Var s2_tile("s2_tile");
        Var s1_tile("s1_tile");
        Var d1_of_p("d1_of_p");

        assert( S1 <= NUM_DOT_C );

        t_for(s2,0,S2);
        {            
            t_for(d1,0,D1);
            {

                /**********************************************************************************/
                /******************************* Setup LLC Buffet Sizes ***************************/
                /**********************************************************************************/
                DeltaArray->AddTileLevel( (DST_DATA_BUFFET*NUM_DOT_C), (DST_DATA_BUFFET*NUM_DOT_C), BUFFET_LINE_SIZE );
                SegmentArray->AddTileLevel( ((DST_DATA_BUFFET*NUM_DOT_C)+1), ((DST_DATA_BUFFET*NUM_DOT_C)+1), BUFFET_LINE_SIZE );

                // this should be implemented as a cache because a given
                // older 's0' coordinate may be reused much later than a new
                // 's0' coordinate, so we don't want to discard
                CoordinateArray->AddTileLevel( (DST_DATA_BUFFET*NUM_DOT_C), (DST_DATA_BUFFET*NUM_DOT_C), BUFFET_LINE_SIZE );

                            
                srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
                dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                
                /**********************************************************************************/

                s_for(s1,0,NUM_DOT_C); //t_for(s1,0,S1);
                {
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);

                    SegmentArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    DeltaArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    CoordinateArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE ); 
                    
                    t_for(s0,0,S0); 
                    {
                        s = s2*S1*S0 + s1*S0 + s0;
                
                        w_if( s < V );
                        {
                    
                            pos_start = (*SegmentArray)[s] + (*DeltaArray)[s];
                            pos_end = (*SegmentArray)[s+1];

                            crossed_tile_boundary = 0;
                            p = pos_start;
                            w_while( (p<pos_end) && (crossed_tile_boundary != 1) );
                            {
                                d = (*CoordinateArray)[p];
                        
                                d1_of_p = (d/D0);
                                
                                w_if( (d1_of_p == d1) );
                                {
                                    (*dstData)[s] += (*ValueArray)[p] * (*srcData)[d];
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

//                             t_for(p,pos_start,pos_end);
//                             {
//                                 d = (*CoordinateArray)[p];
//                         
//                                 d1_of_p = (d/D0);
//                                 
//                                 w_if( (d1_of_p == d1) );
//                                 {
//                                     (*dstData)[s] += (*srcData)[d];
//                                 }
//                                 w_else();
//                                 {
//                                     crossed_tile_boundary = 1;
//                                     (*DeltaArray)[s] += p-pos_start;
//                                     
//                                     p = pos_end;
//                                 }
//                                 end();
//                             }
//                             end();
// 
//                             // a while loop will clean this up
//                             w_if( crossed_tile_boundary == 0 );
//                             {
//                                 (*DeltaArray)[s] += p-pos_start;
//                             }
//                             end();
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }


#define COPLAND_NUM_DOT_M    1
#define COPLAND_NUM_DOT_C    2
#define COPLAND_NUM_CT       4
#define COPLAND_VECTOR_WIDTH 8

    void OnlineTiling_CSR_Copland()
    {

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int RF_SIZE            = (4*KILO     / BYTES_PER_VERTEX);
        int L1_SIZE            = (64*KILO    / BYTES_PER_VERTEX);
        int LLB_SIZE           = (1024*KILO  / BYTES_PER_VERTEX);

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int S0 = RF_SIZE;
        int D0 = L1_SIZE;
        int D1 = (V % D0) ? (V/D0+1) : (V/D0);

        int S1 = COPLAND_NUM_CT;
        int S2 = COPLAND_NUM_DOT_C;
        int S3 = LLB_SIZE / (S0*S1*S2);
        int S4 = (V % (S3*S2*S1*S0)) ? (V/(S3*S2*S1*S0)+1) : (V/(S3*S2*S1*S0));

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tRF  Buffer:  "<<RF_SIZE<<" (S0: "<<S0<<")"<<" Count: S1*S2 where S1: "<<S1<<" S2: "<<S2<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<" Count: S2 where S2: "<<S2<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (S3: "<<S3<<")"<<endl;
        cout<<"\t\tOut Iters S4: "<<S4<<endl;

        cout <<endl;
        
        int d1, s4, s3, s2, s1, d0, s0, d, s;

        int pos_start;
        int pos_end;
        int p, crossed_tile_boundary;
        int s2_tile;
        int s1_tile;
        int d1_of_p;
        int li;
        int lane_iter_cnt;
        int lane;
        int val;
        
        for(s4=0; s4<S4; s4++)
        {
            for(d1=0; d1<D1; d1++)
            {
                for(s3=0; s3<S3; s3++)
                {            
                    for(s2=0; s2<COPLAND_NUM_DOT_C; s2++)
                    {
                        for(s1=0; s1<COPLAND_NUM_CT; s1++) 
                        {
                            for(s0=0; s0<S0; s0++)
                            {
                                s = s4*S3*S2*S1*S0 + s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;                
                                if( s < V )
                                {
                                    pos_start = SegmentArray->At(s) + DeltaArray->At(s);
                                    pos_end = SegmentArray->At(s+1);

                                    // VECTOR SETUP
                                    lane_iter_cnt = (pos_end - pos_start) / COPLAND_VECTOR_WIDTH;
                                    if( (pos_end - pos_start) % COPLAND_VECTOR_WIDTH )
                                    {
                                        lane_iter_cnt += 1;
                                    }

                                    p = pos_start;
                                    // VECTOR EXECUTE
                                    for(li=0; li<lane_iter_cnt; li++)
                                    {
                                        for(lane=0; lane<COPLAND_VECTOR_WIDTH; lane++)                     // VECTOR PARALLELISM
                                        {
                                            p = (li*COPLAND_VECTOR_WIDTH + lane)+pos_start;

                                            if( p<pos_end )
                                            {
                                                d   = CoordinateArray->At(p);
                                                val = ValueArray->At(p);

                                                // d1_of_p = (d/D0);
                                                if( ((d/D0) == d1) )
                                                {
                                                    dstData->At({s}) += val*srcData->At({d});
                                                }
                                                else
                                                {
                                                    lane = COPLAND_VECTOR_WIDTH;
                                                    li = lane_iter_cnt;
                                                }
                                            }
                                        }
                                    }
                                    
                                    // track which neighbors have already been processed
                                    DeltaArray->At(s) += p-pos_start;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        cout<<endl;

        // Validate
        cout<<"\tValidating Copland Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
    }
    
    void WhoopOnlineTiling_CSR_Copland()
    {

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int RF_SIZE            = (4*KILO     / BYTES_PER_VERTEX);
        int L1_SIZE            = (64*KILO    / BYTES_PER_VERTEX);
        int LLB_SIZE           = (1024*KILO  / BYTES_PER_VERTEX);

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        int S0 = RF_SIZE;
        int D0 = L1_SIZE;
        int D1 = (V % D0) ? (V/D0+1) : (V/D0);

        int S1 = COPLAND_NUM_CT;
        int S2 = COPLAND_NUM_DOT_C;
        int S3 = LLB_SIZE / (S0*S1*S2);
        int S4 = (V % (S3*S2*S1*S0)) ? (V/(S3*S2*S1*S0)+1) : (V/(S3*S2*S1*S0));

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tRF  Buffer:  "<<RF_SIZE<<" (S0: "<<S0<<")"<<" Count: S1*S2 where S1: "<<S1<<" S2: "<<S2<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<" Count: S2 where S2: "<<S2<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (S3: "<<S3<<")"<<endl;
        cout<<"\t\tOut Iters S4: "<<S4<<endl;
        
        cout<<endl;

        Var d1("d1"), s4("s4"), s3("s3"), s2("s2"), s1("s1"), d0("d0"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p"), crossed_tile_boundary("crossed_tile_boundary");
        Var s2_tile("s2_tile");
        Var s1_tile("s1_tile");
        Var d1_of_p("d1_of_p");
        Var li("li");
        Var lane_iter_cnt("lane_iter_cnt");
        Var lane("lane");
        Var val("val");

        t_for(s4,0,S4);
        {
            t_for(d1,0,D1);
            {
                t_for(s3,0,S3);
                {            
                    /**********************************************************************************/
                    /******************************* Setup DOT-M Buffet Sizes *************************/
                    /**********************************************************************************/
                    DeltaArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    SegmentArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    CoordinateArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    ValueArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    dstData->AddTileLevel( LLB_SIZE, LLB_SIZE, BUFFET_LINE_SIZE );
                    /**********************************************************************************/

                    s_for(s2,0,COPLAND_NUM_DOT_C);
                    {
                        /**********************************************************************************/
                        /******************************* Setup DOT-C Buffet Sizes ***************************/
                        /**********************************************************************************/
                        srcData->AddTileLevel( L1_SIZE, L1_SIZE, BUFFET_LINE_SIZE );
                        /**********************************************************************************/

                        s_for(s1,0,COPLAND_NUM_CT); 
                        {
                            /**********************************************************************************/
                            /******************************* Setup RF Buffet Sizes ****************************/
                            /**********************************************************************************/
                            dstData->AddTileLevel( RF_SIZE, RF_SIZE, 1 );
                            /**********************************************************************************/

                            t_for(s0,0,S0); 
                            {
                                s = s4*S3*S2*S1*S0 + s3*S2*S1*S0 + s2*S1*S0 + s1*S0 + s0;                

                                w_if( s < V );
                                {
                                    pos_start = (*SegmentArray)[s] + (*DeltaArray)[s];
                                    pos_end = (*SegmentArray)[s+1];

                                    // VECTOR SETUP
                                    lane_iter_cnt = (pos_end - pos_start) / COPLAND_VECTOR_WIDTH;
                                    w_if( (pos_end - pos_start) % COPLAND_VECTOR_WIDTH );
                                    {
                                        lane_iter_cnt += 1;
                                    }
                                    end();

                                    // VECTOR EXECUTE
                                    p = pos_start;
                                    t_for(li,0,lane_iter_cnt);
                                    {
                                        
                                        /**********************************************************************************/
                                        /******************************* Setup Vector Width Sizes ************************/
                                        /**********************************************************************************/
                                        srcData->AddTileLevel( COPLAND_VECTOR_WIDTH, COPLAND_VECTOR_WIDTH, COPLAND_VECTOR_WIDTH );
                                        ValueArray->AddTileLevel( COPLAND_VECTOR_WIDTH, COPLAND_VECTOR_WIDTH, COPLAND_VECTOR_WIDTH );
                                        dstData->AddTileLevel( 1, 1, 1 );
                                        /**********************************************************************************/

                                        s_for(lane,0,COPLAND_VECTOR_WIDTH);                     // VECTOR PARALLELISM
                                        {
                                            p = (li*COPLAND_VECTOR_WIDTH + lane) + pos_start;
                                            w_if( p<pos_end );
                                            {
                                                d   = (*CoordinateArray)[p];

                                                // d1_of_p = (d/D0);
                                                w_if( ((d/D0) == d1) );
                                                {
                                                    (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
                                                }
                                                w_else();
                                                {
                                                    lane = COPLAND_VECTOR_WIDTH;
                                                    li = lane_iter_cnt;
                                                }
                                                end();
                                            }
                                            end();
                                        }
                                        end();
                                    }
                                    end();

                                    // track which neighbors have already been processed
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
        
        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void OnlineTiling_CSC( int D1, int S2, int S1, int D0, int S0 )
    {

        int d1, s2, s1, d0, s0, d, s;

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

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
                                    dstData->At({s}) += srcData->At({d});
                                }
                                else 
                                {
                                    if( pos_start != p ) 
                                    {
                                        DeltaArray->At(d) += (p - pos_start);
                                    }
                                    p = pos_end;
                                }
                            }
                        }
                    }
                }
            }
        }


        // Validate
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }

    void OnlineTiling_CSR( int D1, int S2, int S1, int D0, int S0 )
    {

        int d1, s2, s1, d0, s0, d, s;

        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

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

                            bool crossed_tile_boundary = false;
                            
                            int p = pos_start;
                            while( (p < pos_end) && !crossed_tile_boundary )
                            {
                                int d = CoordinateArray->At(p);
                        
                                int d1_of_p = d/D0;
                        
                                if( (d1_of_p == d1) )
                                {
                                    dstData->At({s}) += srcData->At({d});
                                    p++;
                                }
                                else 
                                {
                                    crossed_tile_boundary = true;
                                }
                            }

                            DeltaArray->At(s) += (p - pos_start);
                        }
                    }
                }
            }
        }


        // Validate
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }

    void WhoopOnlineTiling( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            WhoopOnlineTiling_CSR( D1, S2, S1, D0, S0, BufferL1_KB, BufferL2_KB );
        }
        else if( format == FORMAT_CSC )
        {
            WhoopOnlineTiling_CSC( D1, S2, S1, D0, S0, BufferL1_KB, BufferL2_KB );
        }
    }

    void OnlineTiling( int D1, int S2, int S1, int D0, int S0, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            OnlineTiling_CSR( D1, S2, S1, D0, S0 );
        }
        else if( format == FORMAT_CSC )
        {
            OnlineTiling_CSC( D1, S2, S1, D0, S0 );
        }
    }

    void OnlineTiling_ver2( int D1, int S2, int S1, int D0, int S0, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            OnlineTiling_CSR_ver2( D1, S2, S1, D0, S0 );
        }
        else if( format == FORMAT_CSC )
        {
            OnlineTiling_CSC_ver2( D1, S2, S1, D0, S0 );
        }
    }

    void WhoopOnlineTiling_ver3( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            WhoopOnlineTiling_CSR_ver3( D1, S2, S1, D0, S0, BufferL1_KB, BufferL2_KB );
        }
        else if( format == FORMAT_CSC )
        {
            WhoopOnlineTiling_CSC_ver3( D1, S2, S1, D0, S0, BufferL1_KB, BufferL2_KB );
        }
    }


    void WhoopOnlineTiling_CSC( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB )
    {
        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<endl;

        Var d1("d1"), s2("s2"), s1("s1"), d0("d0"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p");
        Var s2_tile("s2_tile");
        Var s1_tile("s1_tile");

        assert( S1 <= NUM_DOT_C );

        t_for(d1,0,D1);
        {
            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            SegmentArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            DeltaArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            CoordinateArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

            /**********************************************************************************/

            t_for(s2,0,S2);
            {
                s_for(s1,0,NUM_DOT_C); //t_for(s1,0,S1);
                {
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);
                    DeltaArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    SegmentArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    CoordinateArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    
                    t_for(d0,0,D0);
                    {
                        d = d1*D0 + d0;
                
                        w_if( d < V );
                        {
                    
                            pos_start = (*SegmentArray)[d] + (*DeltaArray)[d];
                            pos_end = (*SegmentArray)[d+1];

                            t_for(p,pos_start,pos_end);
                            {
                                s = (*CoordinateArray)[p];
                        
                                s2_tile = s/(S0*S1);
                                s1_tile = (s/S0)%S1;
                                
                                w_if( (s1_tile == s1) && (s2_tile == s2) );
                                {
                                    (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
                                }
                                w_else();
                                {
                                    w_if( pos_start != p );
                                    {
                                        (*DeltaArray)[d] += p-pos_start;
                                    }
                                    end();
                                    
                                    p = pos_end;
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S1*S0} );
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void WhoopOnlineTiling_CSR( int D1, int S2, int S1, int D0, int S0, int BufferL1_KB, int BufferL2_KB )
    {
        // Init Delta Array
        InitDeltaArray();

        CalculateDegrees();

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;

        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<endl;

        Var d1("d1"), s2("s2"), s1("s1"), d0("d0"), s0("s0"), d("d"), s("s");

        Var pos_start("pos_start");
        Var pos_end("pos_end");
        Var p("p");
        Var d1_of_p("d1_of_p");

        assert( S1 <= NUM_DOT_C );

        t_for(d1,0,D1);
        {
            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            SegmentArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            DeltaArray->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            CoordinateArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

            /**********************************************************************************/

            t_for(s2,0,S2);
            {
                s_for(s1,0,NUM_DOT_C); //t_for(s1,0,S1);
                {
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    DeltaArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    SegmentArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    CoordinateArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    
                    t_for(s0,0,S0);
                    {
                        s = s2*S1*S0 + s1*S0 + s0;
                
                        w_if( s < V );
                        {
                    
                            pos_start = (*SegmentArray)[s] + (*DeltaArray)[s];
                            pos_end = (*SegmentArray)[s+1];

                            t_for(p,pos_start,pos_end);
                            {
                                d = (*CoordinateArray)[p];
                        
                                d1_of_p = d/D0;
                                
                                w_if( d1_of_p == d1 );
                                {
                                    (*dstData)[s] += (*ValueArray)[p]*(*srcData)[d];
                                }
                                w_else();
                                {
                                    w_if( pos_start != p );
                                    {
                                        (*DeltaArray)[s] += p-pos_start;
                                    }
                                    end();
                                    
                                    p = pos_end;
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void RunWhoop5_CSC( int BufferL1_KB, int BufferL2_KB )
    {

        int D1 = adjMat_csf->dim_sizes[4];
        int S2 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int D0 = adjMat_csf->dim_sizes[1];
        int S0 = adjMat_csf->dim_sizes[0];

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<endl;
        

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        Var start_s2_pos("start_s2_pos"), end_s2_pos("end_s2_pos");
        Var start_s1_pos("start_s1_pos"), end_s1_pos("end_s1_pos");
        Var start_d1_pos("start_d1_pos"), end_d1_pos("end_d1_pos");
        Var start_s0_pos("start_s0_pos"), end_s0_pos("end_s0_pos");
        Var start_d0_pos("start_d0_pos"), end_d0_pos("end_d0_pos");

        Var s2_pos("s2_pos");
        Var s1_pos("s1_pos");
        Var d1_pos("d1_pos");
        Var s0_pos("s0_pos");
        Var d0_pos("d0_pos");

        Var s2("s2"), s1("s1"), d1("d1"), s0("s0"), d0("d0"), tensor_val("tensor_val");

        int D1_max = adjMat_csf->CA_insert_ptr[4];

        t_for(d1_pos, 0, D1_max);
        {

            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            adjMat_csf->coordArray[4]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[3]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[2]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[1]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[0]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            adjMat_csf->segArray[4]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[3]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[2]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[1]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[0]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

            /**********************************************************************************/

            d1 = (*adjMat_csf->coordArray[4])[d1_pos];
            
            start_s2_pos = (*adjMat_csf->segArray[3])[d1_pos];
            end_s2_pos   = (*adjMat_csf->segArray[3])[d1_pos+1];

            t_for(s2_pos, start_s2_pos, end_s2_pos);
            {
                s2 = (*adjMat_csf->coordArray[3])[s2_pos];
                
                start_s1_pos = (*adjMat_csf->segArray[2])[s2_pos];
                end_s1_pos   = (*adjMat_csf->segArray[2])[s2_pos+1];
                
                t_for(s1_pos, start_s1_pos, end_s1_pos);
                {
                    // Dest data buffet size
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);
                    
                    s1 = (*adjMat_csf->coordArray[2])[s1_pos];
                    
                    start_d0_pos = (*adjMat_csf->segArray[1])[s1_pos];
                    end_d0_pos   = (*adjMat_csf->segArray[1])[s1_pos+1];
                    
                    t_for(d0_pos, start_d0_pos, end_d0_pos );
                    {
                        d0 = (*adjMat_csf->coordArray[1])[d0_pos];
                        
                        start_s0_pos = (*adjMat_csf->segArray[0])[d0_pos];
                        end_s0_pos   = (*adjMat_csf->segArray[0])[d0_pos+1];
                        
                        t_for(s0_pos, start_s0_pos, end_s0_pos );
                        {
                            s0         = (*adjMat_csf->coordArray[0])[s0_pos]; 
                            tensor_val = (*adjMat_csf->valArray)[s0_pos]; 

                            (*dstData)[s2][s1][s0] += tensor_val*(*srcData)[d1][d0];
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;


        whoop::Done();
    }

    void RunWhoop5_CSR( int BufferL1_KB, int BufferL2_KB )
    {

        int D1 = adjMat_csf->dim_sizes[4];
        int S2 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int D0 = adjMat_csf->dim_sizes[1];
        int S0 = adjMat_csf->dim_sizes[0];

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<endl;
        

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        Var start_s2_pos("start_s2_pos"), end_s2_pos("end_s2_pos");
        Var start_s1_pos("start_s1_pos"), end_s1_pos("end_s1_pos");
        Var start_d1_pos("start_d1_pos"), end_d1_pos("end_d1_pos");
        Var start_s0_pos("start_s0_pos"), end_s0_pos("end_s0_pos");
        Var start_d0_pos("start_d0_pos"), end_d0_pos("end_d0_pos");

        Var s2_pos("s2_pos");
        Var s1_pos("s1_pos");
        Var d1_pos("d1_pos");
        Var s0_pos("s0_pos");
        Var d0_pos("d0_pos");

        Var s2("s2"), s1("s1"), d1("d1"), s0("s0"), d0("d0"), tensor_val("tensor_val");

        int D1_max = adjMat_csf->CA_insert_ptr[4];

        t_for(d1_pos, 0, D1_max);
        {

            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            adjMat_csf->coordArray[4]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[3]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[2]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[1]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[0]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            adjMat_csf->segArray[4]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[3]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[2]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[1]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[0]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

            /**********************************************************************************/

            d1 = (*adjMat_csf->coordArray[4])[d1_pos];
            
            start_s2_pos = (*adjMat_csf->segArray[3])[d1_pos];
            end_s2_pos   = (*adjMat_csf->segArray[3])[d1_pos+1];

            t_for(s2_pos, start_s2_pos, end_s2_pos);
            {
                s2 = (*adjMat_csf->coordArray[3])[s2_pos];
                
                start_s1_pos = (*adjMat_csf->segArray[2])[s2_pos];
                end_s1_pos   = (*adjMat_csf->segArray[2])[s2_pos+1];
                
                t_for(s1_pos, start_s1_pos, end_s1_pos);
                {
                    // Dest data buffet size
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);
                    
                    s1 = (*adjMat_csf->coordArray[2])[s1_pos];
                    
                    start_s0_pos = (*adjMat_csf->segArray[1])[s1_pos];
                    end_s0_pos   = (*adjMat_csf->segArray[1])[s1_pos+1];
                    
                    t_for(s0_pos, start_s0_pos, end_s0_pos );
                    {
                        s0 = (*adjMat_csf->coordArray[1])[s0_pos];
                        
                        start_d0_pos = (*adjMat_csf->segArray[0])[s0_pos];
                        end_d0_pos   = (*adjMat_csf->segArray[0])[s0_pos+1];
                        
                        t_for(d0_pos, start_d0_pos, end_d0_pos );
                        {
                            d0         = (*adjMat_csf->coordArray[0])[d0_pos]; 
                            tensor_val = (*adjMat_csf->valArray)[d0_pos]; 

                            (*dstData)[s2][s1][s0] += tensor_val*(*srcData)[d1][d0];
                            (*dstData)[s2][s1][s0] += (*srcData)[d1][d0];
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;

        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;


        whoop::Done();
    }



    void RunWhoop5( int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            RunWhoop5_Parallel_CSR( BufferL1_KB, BufferL2_KB );
        }
        else if( format == FORMAT_CSC )
        {
            RunWhoop5_Parallel_CSC( BufferL1_KB, BufferL2_KB );
        }
    }

    void RunWhoop5_Parallel( int BufferL1_KB, int BufferL2_KB, FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            RunWhoop5_Parallel_CSR( BufferL1_KB, BufferL2_KB );
        }
        else if( format == FORMAT_CSC )
        {
            RunWhoop5_Parallel_CSC( BufferL1_KB, BufferL2_KB );
        }
    }
    


    void RunWhoop5_Parallel_CSC( int BufferL1_KB, int BufferL2_KB )
    {

        int D1 = adjMat_csf->dim_sizes[4];
        int S2 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int D0 = adjMat_csf->dim_sizes[1];
        int S0 = adjMat_csf->dim_sizes[0];

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<endl;
        

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        Var start_s2_pos("start_s2_pos"), end_s2_pos("end_s2_pos");
        Var start_s1_pos("start_s1_pos"), end_s1_pos("end_s1_pos");
        Var start_d1_pos("start_d1_pos"), end_d1_pos("end_d1_pos");
        Var start_s0_pos("start_s0_pos"), end_s0_pos("end_s0_pos");
        Var start_d0_pos("start_d0_pos"), end_d0_pos("end_d0_pos");

        Var s2_pos("s2_pos");
        Var s1_pos("s1_pos");
        Var d1_pos("d1_pos");
        Var s0_pos("s0_pos");
        Var d0_pos("d0_pos");

        Var ct("ct"), s2("s2"), s1("s1"), d1("d1"), s0("s0"), d0("d0"), tensor_val("tensor_val");

        int D1_max = adjMat_csf->CA_insert_ptr[4];

        t_for(d1_pos, 0, D1_max);
        {

            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            adjMat_csf->coordArray[4]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[3]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[2]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[1]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[0]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->valArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            adjMat_csf->segArray[4]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[3]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[2]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[1]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[0]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );


            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

            /**********************************************************************************/

            d1 = (*adjMat_csf->coordArray[4])[d1_pos];
            
            start_s2_pos = (*adjMat_csf->segArray[3])[d1_pos];
            end_s2_pos   = (*adjMat_csf->segArray[3])[d1_pos+1];

            t_for(s2_pos, start_s2_pos, end_s2_pos);
            {
                s2 = (*adjMat_csf->coordArray[3])[s2_pos];
                
                start_s1_pos = (*adjMat_csf->segArray[2])[s2_pos];
                end_s1_pos   = (*adjMat_csf->segArray[2])[s2_pos+1];
                
                s_for(ct, 0, NUM_DOT_C);
                {
                    // Dest data buffet size
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);

                    adjMat_csf->segArray[1]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    adjMat_csf->segArray[0]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    adjMat_csf->coordArray[2]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    adjMat_csf->coordArray[1]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    adjMat_csf->coordArray[0]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    adjMat_csf->valArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    s1_pos = ct + start_s1_pos;
                    w_if( s1_pos < end_s1_pos );
                    {
                        s1 = (*adjMat_csf->coordArray[2])[s1_pos];
                    
                        start_d0_pos = (*adjMat_csf->segArray[1])[s1_pos];
                        end_d0_pos   = (*adjMat_csf->segArray[1])[s1_pos+1];
                    
                        t_for(d0_pos, start_d0_pos, end_d0_pos );
                        {
                            d0 = (*adjMat_csf->coordArray[1])[d0_pos];
                        
                            start_s0_pos = (*adjMat_csf->segArray[0])[d0_pos];
                            end_s0_pos   = (*adjMat_csf->segArray[0])[d0_pos+1];
                        
                            t_for(s0_pos, start_s0_pos, end_s0_pos );
                            {
                                s0         = (*adjMat_csf->coordArray[0])[s0_pos]; 
                                tensor_val = (*adjMat_csf->valArray)[s0_pos]; 

                                (*dstData)[s2][s1][s0] += tensor_val*(*srcData)[d1][d0];
//                                 (*dstData)[s2][s1][s0] += (*srcData)[d1][d0];
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;


        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void RunWhoop5_Parallel_CSR( int BufferL1_KB, int BufferL2_KB )
    {

        int D1 = adjMat_csf->dim_sizes[4];
        int S2 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int S0 = adjMat_csf->dim_sizes[1];
        int D0 = adjMat_csf->dim_sizes[0];

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<endl;
        

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        Var start_s2_pos("start_s2_pos"), end_s2_pos("end_s2_pos");
        Var start_s1_pos("start_s1_pos"), end_s1_pos("end_s1_pos");
        Var start_d1_pos("start_d1_pos"), end_d1_pos("end_d1_pos");
        Var start_s0_pos("start_s0_pos"), end_s0_pos("end_s0_pos");
        Var start_d0_pos("start_d0_pos"), end_d0_pos("end_d0_pos");

        Var s2_pos("s2_pos");
        Var s1_pos("s1_pos");
        Var d1_pos("d1_pos");
        Var s0_pos("s0_pos");
        Var d0_pos("d0_pos");

        Var ct("ct"), s2("s2"), s1("s1"), d1("d1"), s0("s0"), d0("d0"), tensor_val("tensor_val");

        int D1_max = adjMat_csf->CA_insert_ptr[4];

        t_for(d1_pos, 0, D1_max);
        {

            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            adjMat_csf->coordArray[4]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[3]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[2]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[1]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[0]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->valArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            adjMat_csf->segArray[4]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[3]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[2]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[1]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[0]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );


            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
            dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

            /**********************************************************************************/

            d1 = (*adjMat_csf->coordArray[4])[d1_pos];
            
            start_s2_pos = (*adjMat_csf->segArray[3])[d1_pos];
            end_s2_pos   = (*adjMat_csf->segArray[3])[d1_pos+1];

            t_for(s2_pos, start_s2_pos, end_s2_pos);
            {
                s2 = (*adjMat_csf->coordArray[3])[s2_pos];
                
                start_s1_pos = (*adjMat_csf->segArray[2])[s2_pos];
                end_s1_pos   = (*adjMat_csf->segArray[2])[s2_pos+1];
                
                s_for(ct, 0, NUM_DOT_C);
                {
                    // Dest data buffet size
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
                    srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);

                    adjMat_csf->segArray[1]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    adjMat_csf->segArray[0]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    adjMat_csf->coordArray[2]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    adjMat_csf->coordArray[1]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
                    adjMat_csf->coordArray[0]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    adjMat_csf->valArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    s1_pos = ct + start_s1_pos;
                    w_if( s1_pos < end_s1_pos );
                    {
                        s1 = (*adjMat_csf->coordArray[2])[s1_pos];
                    
                        start_s0_pos = (*adjMat_csf->segArray[1])[s1_pos];
                        end_s0_pos   = (*adjMat_csf->segArray[1])[s1_pos+1];
                    
                        t_for(s0_pos, start_s0_pos, end_s0_pos );
                        {
                            s0 = (*adjMat_csf->coordArray[1])[s0_pos];
                        
                            start_d0_pos = (*adjMat_csf->segArray[0])[s0_pos];
                            end_d0_pos   = (*adjMat_csf->segArray[0])[s0_pos+1];
                        
                            t_for(d0_pos, start_d0_pos, end_d0_pos );
                            {
                                d0         = (*adjMat_csf->coordArray[0])[d0_pos]; 
                                tensor_val = (*adjMat_csf->valArray)[d0_pos]; 

                                (*dstData)[s2][s1][s0] += tensor_val*(*srcData)[d1][d0];
//                                 (*dstData)[s2][s1][s0] += (*srcData)[d1][d0];
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;


        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void RunWhoop5_Parallel_CSR_ver2( int BufferL1_KB, int BufferL2_KB )
    {

        int S2 = adjMat_csf->dim_sizes[4];
        int D1 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int S0 = adjMat_csf->dim_sizes[1];
        int D0 = adjMat_csf->dim_sizes[0];

        int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;

        int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
        int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);

        int SRC_DATA_BUFFET    = LLB_SIZE;
        int DST_DATA_BUFFET    = L1_SIZE;

        int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
        int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;


        cout<<endl;
        cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
        cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
        cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
        cout<<endl;
        

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        Var start_s2_pos("start_s2_pos"), end_s2_pos("end_s2_pos");
        Var start_s1_pos("start_s1_pos"), end_s1_pos("end_s1_pos");
        Var start_d1_pos("start_d1_pos"), end_d1_pos("end_d1_pos");
        Var start_s0_pos("start_s0_pos"), end_s0_pos("end_s0_pos");
        Var start_d0_pos("start_d0_pos"), end_d0_pos("end_d0_pos");

        Var s2_pos("s2_pos");
        Var s1_pos("s1_pos");
        Var d1_pos("d1_pos");
        Var s0_pos("s0_pos");
        Var d0_pos("d0_pos");

        Var ct("ct"), s2("s2"), s1("s1"), d1("d1"), s0("s0"), d0("d0"), tensor_val("tensor_val");

        int S2_max = adjMat_csf->CA_insert_ptr[4];

        t_for(s2_pos, 0, S2_max);
        {

            /**********************************************************************************/
            /******************************* Setup LLC Buffet Sizes ***************************/
            /**********************************************************************************/
            adjMat_csf->coordArray[4]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[3]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[2]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[1]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->coordArray[0]->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->valArray->AddTileLevel( COORD_ARRAY_BUFFET, COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );

            adjMat_csf->segArray[4]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[3]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[2]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[1]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
            adjMat_csf->segArray[0]->AddTileLevel( SEG_ARRAY_BUFFET, SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );


            srcData->AddTileLevel( SRC_DATA_BUFFET, SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
//             dstData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);
            /**********************************************************************************/

            s2 = (*adjMat_csf->coordArray[4])[s2_pos];
            
            start_d1_pos = (*adjMat_csf->segArray[3])[s2_pos];
            end_d1_pos   = (*adjMat_csf->segArray[3])[s2_pos+1];

            t_for(d1_pos, start_d1_pos, end_d1_pos);
            {
                d1 = (*adjMat_csf->coordArray[3])[d1_pos];
                
                start_s1_pos = (*adjMat_csf->segArray[2])[d1_pos];
                end_s1_pos   = (*adjMat_csf->segArray[2])[d1_pos+1];
                
                s_for(ct, 0, NUM_DOT_C);
                {
                    // Dest data buffet size
                    dstData->AddTileLevel( DST_DATA_BUFFET, DST_DATA_BUFFET, BUFFET_LINE_SIZE);
//                     srcData->AddTileLevel( 1, 1, BUFFET_LINE_SIZE);

//                     adjMat_csf->segArray[4]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->segArray[3]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->segArray[2]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->segArray[1]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->segArray[0]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
// 
//                     adjMat_csf->coordArray[4]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->coordArray[3]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->coordArray[2]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->coordArray[1]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
//                     adjMat_csf->coordArray[0]->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );
// 
//                     adjMat_csf->valArray->AddTileLevel( 1, 1, BUFFET_LINE_SIZE );

                    s1_pos = ct + start_s1_pos;
                    w_if( s1_pos < end_s1_pos );
                    {
                        s1 = (*adjMat_csf->coordArray[2])[s1_pos];
                    
                        start_s0_pos = (*adjMat_csf->segArray[1])[s1_pos];
                        end_s0_pos   = (*adjMat_csf->segArray[1])[s1_pos+1];
                    
                        t_for(s0_pos, start_s0_pos, end_s0_pos );
                        {
                            s0 = (*adjMat_csf->coordArray[1])[s0_pos];
                        
                            start_d0_pos = (*adjMat_csf->segArray[0])[s0_pos];
                            end_d0_pos   = (*adjMat_csf->segArray[0])[s0_pos+1];
                        
                            t_for(d0_pos, start_d0_pos, end_d0_pos );
                            {
                                d0         = (*adjMat_csf->coordArray[0])[d0_pos]; 
                                tensor_val = (*adjMat_csf->valArray)[d0_pos]; 

                                (*dstData)[s2][s1][s0] += tensor_val*(*srcData)[d1][d0];
//                                 (*dstData)[s2][s1][s0] += (*srcData)[d1][d0];
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


        cout<<endl;
        cout<< "\tStarting WHOOP Mode..." <<endl;
        whoop::Run();
        cout<< "\tFinished WHOOP Mode..." <<endl;


        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating WHOOP Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;

        whoop::Done();
    }

    void Run5( FORMAT_TYPE format )
    {
        if( format == FORMAT_CSR ) 
        {
            Run5_CSR();
        }
        else if( format == FORMAT_CSC ) 
        {
            Run5_CSC();
        }
    }

    void Run5_CSR()
    {

        int D1 = adjMat_csf->dim_sizes[4];
        int S2 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int S0 = adjMat_csf->dim_sizes[1];
        int D0 = adjMat_csf->dim_sizes[0];

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        int start_s2_pos, end_s2_pos;
        int start_s1_pos, end_s1_pos;
        int start_d1_pos, end_d1_pos;
        int start_s0_pos, end_s0_pos;
        int start_d0_pos, end_d0_pos;

        int s2_pos, s1_pos, d1_pos, s0_pos, d0_pos;
        
        int s2, s1, d1, s0, d0, tensor_val;

        int D1_max = adjMat_csf->CA_insert_ptr[4];

        for(d1_pos=0; d1_pos<D1_max; d1_pos++)
        {

            d1 = adjMat_csf->coordArray[4]->At({d1_pos});

            start_s2_pos = adjMat_csf->segArray[3]->At({d1_pos});
            end_s2_pos   = adjMat_csf->segArray[3]->At({d1_pos+1});

            for(s2_pos=start_s2_pos; s2_pos<end_s2_pos; s2_pos++)
            {
                s2 = adjMat_csf->coordArray[3]->At({s2_pos});

                start_s1_pos = adjMat_csf->segArray[2]->At({s2_pos});
                end_s1_pos   = adjMat_csf->segArray[2]->At({s2_pos+1});

                for(s1_pos=start_s1_pos; s1_pos<end_s1_pos; s1_pos++)
                {
                    s1 = adjMat_csf->coordArray[2]->At({s1_pos});
                
                    start_s0_pos = adjMat_csf->segArray[1]->At({s1_pos});
                    end_s0_pos   = adjMat_csf->segArray[1]->At({s1_pos+1});
                
                    for(s0_pos=start_s0_pos; s0_pos<end_s0_pos; s0_pos++ )
                    {
                        s0 = adjMat_csf->coordArray[1]->At({s0_pos});

                        start_d0_pos = adjMat_csf->segArray[0]->At({s0_pos});
                        end_d0_pos   = adjMat_csf->segArray[0]->At({s0_pos+1});
                    
                        for(d0_pos=start_d0_pos; d0_pos<end_d0_pos; d0_pos++ )
                        {
                            d0         = adjMat_csf->coordArray[0]->At({d0_pos}); 
                            tensor_val = adjMat_csf->valArray->At({d0_pos}); 

//                             cout<<"Exec: d1: "<<d1<<" s2: "<<s2<<" s1: "<<s1<<" s0: "<<s0<<" d0: "<<d0<<" ("<<(d1*D0+d0)<<","<<(s2*S1*S0+s1*S0+s0)<<")"<<endl;
                            
                            dstData->At({s2,s1,s0}) += tensor_val * srcData->At({d1,d0});
//                             dstData->At({s2,s1,s0}) += srcData->At({d1,d0});
                        }
                    }
                }
            }
        }
        

        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }

    void Run5_CSR_ver2()
    {
//         map<int,int> d0_uniq;
//         map<int,int> s0_uniq;
        
        int S2 = adjMat_csf->dim_sizes[4];
        int D1 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int S0 = adjMat_csf->dim_sizes[1];
        int D0 = adjMat_csf->dim_sizes[0];

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        int start_s2_pos, end_s2_pos;
        int start_s1_pos, end_s1_pos;
        int start_d1_pos, end_d1_pos;
        int start_s0_pos, end_s0_pos;
        int start_d0_pos, end_d0_pos;

        int s2_pos, s1_pos, d1_pos, s0_pos, d0_pos;
        
        int s2, s1, d1, s0, d0, tensor_val;

        int S2_max = adjMat_csf->CA_insert_ptr[4];

        for(s2_pos=0; s2_pos<S2_max; s2_pos++)
        {

            s2 = adjMat_csf->coordArray[4]->At({s2_pos});

            start_d1_pos = adjMat_csf->segArray[3]->At({s2_pos});
            end_d1_pos   = adjMat_csf->segArray[3]->At({s2_pos+1});

            for(d1_pos=start_d1_pos; d1_pos<end_d1_pos; d1_pos++)
            {
                d1 = adjMat_csf->coordArray[3]->At({d1_pos});

                start_s1_pos = adjMat_csf->segArray[2]->At({d1_pos});
                end_s1_pos   = adjMat_csf->segArray[2]->At({d1_pos+1});

                for(s1_pos=start_s1_pos; s1_pos<end_s1_pos; s1_pos++)
                {
                    s1 = adjMat_csf->coordArray[2]->At({s1_pos});
                
                    start_s0_pos = adjMat_csf->segArray[1]->At({s1_pos});
                    end_s0_pos   = adjMat_csf->segArray[1]->At({s1_pos+1});

                    for(s0_pos=start_s0_pos; s0_pos<end_s0_pos; s0_pos++ )
                    {
                        s0 = adjMat_csf->coordArray[1]->At({s0_pos});

                        start_d0_pos = adjMat_csf->segArray[0]->At({s0_pos});
                        end_d0_pos   = adjMat_csf->segArray[0]->At({s0_pos+1});
                    
                        for(d0_pos=start_d0_pos; d0_pos<end_d0_pos; d0_pos++ )
                        {
                            d0         = adjMat_csf->coordArray[0]->At({d0_pos}); 
                            tensor_val = adjMat_csf->valArray->At({d0_pos}); 

                            dstData->At({s2,s1,s0}) += tensor_val * srcData->At({d1,d0});
                        }
                    }
                }
            }
        }
        

        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }

    // Expects dimension sizes in format:  (S0, D0, S1, S2, D1)
    void Run5_CSC()
    {

        int D1 = adjMat_csf->dim_sizes[4];
        int S2 = adjMat_csf->dim_sizes[3];
        int S1 = adjMat_csf->dim_sizes[2];
        int D0 = adjMat_csf->dim_sizes[1];
        int S0 = adjMat_csf->dim_sizes[0];

        srcData->Resize( {D1,D0} );
        dstData->Resize( {S2,S1,S0} );

        int start_s2_pos, end_s2_pos;
        int start_s1_pos, end_s1_pos;
        int start_d1_pos, end_d1_pos;
        int start_s0_pos, end_s0_pos;
        int start_d0_pos, end_d0_pos;

        int s2_pos, s1_pos, d1_pos, s0_pos, d0_pos;
        
        int s2, s1, d1, s0, d0, tensor_val;

        int D1_max = adjMat_csf->CA_insert_ptr[4];

        for(d1_pos=0; d1_pos<D1_max; d1_pos++)
        {

            d1 = adjMat_csf->coordArray[4]->At({d1_pos});

            start_s2_pos = adjMat_csf->segArray[3]->At({d1_pos});
            end_s2_pos   = adjMat_csf->segArray[3]->At({d1_pos+1});

            for(s2_pos=start_s2_pos; s2_pos<end_s2_pos; s2_pos++)
            {
                s2 = adjMat_csf->coordArray[3]->At({s2_pos});

                start_s1_pos = adjMat_csf->segArray[2]->At({s2_pos});
                end_s1_pos   = adjMat_csf->segArray[2]->At({s2_pos+1});

                for(s1_pos=start_s1_pos; s1_pos<end_s1_pos; s1_pos++)
                {
                    s1 = adjMat_csf->coordArray[2]->At({s1_pos});
                
                    start_d0_pos = adjMat_csf->segArray[1]->At({s1_pos});
                    end_d0_pos   = adjMat_csf->segArray[1]->At({s1_pos+1});
                
                    for(d0_pos=start_d0_pos; d0_pos<end_d0_pos; d0_pos++ )
                    {
                        d0 = adjMat_csf->coordArray[1]->At({d0_pos});

                        start_s0_pos = adjMat_csf->segArray[0]->At({d0_pos});
                        end_s0_pos   = adjMat_csf->segArray[0]->At({d0_pos+1});
                    
                        for(s0_pos=start_s0_pos; s0_pos<end_s0_pos; s0_pos++ )
                        {
                            s0         = adjMat_csf->coordArray[0]->At({s0_pos}); 
                            tensor_val = adjMat_csf->valArray->At({s0_pos}); 

//                             cout<<"Exec: d1: "<<d1<<" s2: "<<s2<<" s1: "<<s1<<" s0: "<<s0<<" d0: "<<d0<<" ("<<(d1*D0+d0)<<","<<(s2*S1*S0+s1*S0+s0)<<")"<<endl;
                            
                            dstData->At({s2,s1,s0}) += tensor_val * srcData->At({d1,d0});
//                             dstData->At({s2,s1,s0}) += srcData->At({d1,d0});
                        }
                    }
                }
            }
        }
        

        // Validate
        srcData->Resize( {D1*D0} );
        dstData->Resize( {S2*S1*S0} );
        cout<<"\tValidating Results... ";
        Validate();
        cout<<"DONE!"<<endl;
        cout<<endl;
        
    }
};

#endif


//     // Expects dimension sizes in format:  (S0, D0, S1, D1)
//     void Run4()
//     {
// 
//         int D1 = adjMat_csf->dim_sizes[3];
//         int S1 = adjMat_csf->dim_sizes[2];
//         int D0 = adjMat_csf->dim_sizes[1];
//         int S0 = adjMat_csf->dim_sizes[0];
// 
//         srcData->Resize( {D1,D0} );
//         dstData->Resize( {S1,S0} );
// 
//         int start_s1_pos, end_s1_pos;
//         int start_d1_pos, end_d1_pos;
//         int start_s0_pos, end_s0_pos;
//         int start_d0_pos, end_d0_pos;
// 
//         int s1_pos, d1_pos, s0_pos, d0_pos;
//         
//         int s1, d1, s0, d0, tensor_val;
// 
//         int D1_max = adjMat_csf->CA_insert_ptr[3];
// 
//         for(d1_pos=0; d1_pos<D1_max; d1_pos++)
//         {
// 
//             d1 = adjMat_csf->coordArray[3]->At({d1_pos});
//             
//             start_s1_pos = adjMat_csf->segArray[2]->At({d1_pos});
//             end_s1_pos   = adjMat_csf->segArray[2]->At({d1_pos+1});
// 
//             for(s1_pos=start_s1_pos; s1_pos<end_s1_pos; s1_pos++)
//             {
//                 s1 = adjMat_csf->coordArray[2]->At({s1_pos});
//                 
//                 start_d0_pos = adjMat_csf->segArray[1]->At({s1_pos});
//                 end_d0_pos   = adjMat_csf->segArray[1]->At({s1_pos+1});
//                 
//                 for(d0_pos=start_d0_pos; d0_pos<end_d0_pos; d0_pos++ )
//                 {
//                     d0 = adjMat_csf->coordArray[1]->At({d0_pos});
// 
//                     start_s0_pos = adjMat_csf->segArray[0]->At({d0_pos});
//                     end_s0_pos   = adjMat_csf->segArray[0]->At({d0_pos+1});
//                     
//                     for(s0_pos=start_s0_pos; s0_pos<end_s0_pos; s0_pos++ )
//                     {
//                         s0         = adjMat_csf->coordArray[0]->At({s0_pos}); 
//                         tensor_val = adjMat_csf->valArray->At({s0_pos}); 
// 
// 
//                         dstData->At({s1,s0}) += tensor_val * srcData->At({d1,d0});
//                     }
//                 }
//             }
//         }
// 
//         srcData->Resize( {D1*D0} );
//         dstData->Resize( {S1*S0} );
// 
//         cout<<"\tValidating Results... ";
//         Validate();
//         cout<<"DONE!"<<endl;
//         cout<<endl;
//     }
//
//     // Expects dimension sizes in format:  (S0, D0, S1, D1)
//     void RunWhoop4( int BufferL1_KB, int BufferL2_KB )
//     {
// 
//         int D1 = adjMat_csf->dim_sizes[3];
//         int S1 = adjMat_csf->dim_sizes[2];
//         int D0 = adjMat_csf->dim_sizes[1];
//         int S0 = adjMat_csf->dim_sizes[0];
// 
//         int BUFFET_LINE_SIZE   = ARG_BUFFET_GRANULARITY;
// 
//         int L1_SIZE            = (BufferL1_KB*KILO / BYTES_PER_VERTEX);
//         int LLB_SIZE           = (BufferL2_KB*KILO  / BYTES_PER_VERTEX);
// 
//         int SRC_DATA_BUFFET    = LLB_SIZE;
//         int DST_DATA_BUFFET    = L1_SIZE;
// 
//         int SEG_ARRAY_BUFFET   = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
//         int COORD_ARRAY_BUFFET = LINE_SIZE_BYTES / BYTES_PER_VERTEX;
// 
// 
//         cout<<endl;
//         cout<<"\tBuffer Hierarchy (Vertex Count):"<<endl;
//         cout<<"\t\tL1  Buffer:  "<<L1_SIZE<<" (S0: "<<S0<<")"<<endl;
//         cout<<"\t\tLLC Buffer:  "<<LLB_SIZE<<" (D0: "<<D0<<")"<<endl;
//         cout<<endl;
//         
// 
//         srcData->Resize( {D1,D0} );
//         dstData->Resize( {S1,S0} );
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
//         Var s1("s1"), d1("d1"), s0("s0"), d0("d0"), tensor_val("tensor_val");
// 
//         int D1_max = adjMat_csf->CA_insert_ptr[3];
// 
//         t_for(d1_pos, 0, D1_max);
//         {
// 
//             /**********************************************************************************/
//             /******************************* Setup LLC Buffet Sizes ***************************/
//             /**********************************************************************************/
//             adjMat_csf->coordArray[3]->AddTileLevel( COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
//             adjMat_csf->coordArray[2]->AddTileLevel( COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
//             adjMat_csf->coordArray[1]->AddTileLevel( COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
//             adjMat_csf->coordArray[0]->AddTileLevel( COORD_ARRAY_BUFFET, BUFFET_LINE_SIZE );
// 
//             adjMat_csf->segArray[3]->AddTileLevel( SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
//             adjMat_csf->segArray[2]->AddTileLevel( SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
//             adjMat_csf->segArray[1]->AddTileLevel( SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
//             adjMat_csf->segArray[0]->AddTileLevel( SEG_ARRAY_BUFFET, BUFFET_LINE_SIZE );
// 
//             srcData->AddTileLevel( SRC_DATA_BUFFET, BUFFET_LINE_SIZE );
// 
//             /**********************************************************************************/
// 
//             d1 = (*adjMat_csf->coordArray[3])[d1_pos];
//             
//             start_s1_pos = (*adjMat_csf->segArray[2])[d1_pos];
//             end_s1_pos   = (*adjMat_csf->segArray[2])[d1_pos+1];
// 
//             t_for(s1_pos, start_s1_pos, end_s1_pos);
//             {
//                 
//                 // Dest data buffet size
//                 dstData->AddTileLevel( DST_DATA_BUFFET, BUFFET_LINE_SIZE);
// 
//                 s1 = (*adjMat_csf->coordArray[2])[s1_pos];
//                 
//                 start_d0_pos = (*adjMat_csf->segArray[1])[s1_pos];
//                 end_d0_pos   = (*adjMat_csf->segArray[1])[s1_pos+1];
//                 
//                 t_for(d0_pos, start_d0_pos, end_d0_pos );
//                 {
//                     d0 = (*adjMat_csf->coordArray[1])[d0_pos];
//                     
//                     start_s0_pos = (*adjMat_csf->segArray[0])[d0_pos];
//                     end_s0_pos   = (*adjMat_csf->segArray[0])[d0_pos+1];
//                     
//                     t_for(s0_pos, start_s0_pos, end_s0_pos );
//                     {
//                         s0         = (*adjMat_csf->coordArray[0])[s0_pos]; 
//                         tensor_val = (*adjMat_csf->valArray)[s0_pos]; 
// 
//                         (*dstData)[s1][s0] += tensor_val*(*srcData)[d1][d0];
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
//         cout<< "\tStarting WHOOP Mode..." <<endl;
//         whoop::Run();
//         cout<< "\tFinished WHOOP Mode..." <<endl;
// 
//         cout<<"\tValidating WHOOP Results... ";
//         srcData->Resize( {D1*D0} );
//         dstData->Resize( {S1*S0} );
//         Validate();
//         cout<<"DONE!"<<endl;
//         cout<<endl;
// 
//         whoop::Done();
//     }
//     

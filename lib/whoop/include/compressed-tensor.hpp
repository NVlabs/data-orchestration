/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace whoop
{

class DomainIterator;
class FullyConcordantScanner;

class CompressedTensor
{
 public:

  enum class Type
  {
   Normal,
   In,
   Out
  };

  std::string name_;
  Type my_type_;

  int             num_dims_;
  int             nnz_in_; // Number of expected NNZ for sanity checking.

  std::vector<int>          dim_sizes_;
  std::vector<std::string>  dim_names_;
  std::vector<int>          last_inserted_coordinate_; // Last inserted point for sanity checking.

 public:

  std::vector<std::shared_ptr<Vec>> segments_;
  std::vector<std::shared_ptr<Vec>> coordinates_;
  std::shared_ptr<Vec> values_;

  CompressedTensor(const std::string& name, const int num_dims, const std::vector<int>& dim_sizes, const std::vector<std::string>& dim_names, long nnz, const Type& my_type = Type::Normal) :
    name_(name),
    my_type_(my_type),
    num_dims_(num_dims),
    dim_sizes_(dim_sizes),
    dim_names_(dim_names),
    last_inserted_coordinate_(dim_sizes_.size(), -1),
    coordinates_(num_dims, NULL),
    segments_(num_dims, NULL)
  {
    //if (my_type_ != Type::Out) Neal: we need to init output as well (have override for init)
      Init(nnz);        
  }

  std::vector<int> DimensionSizes()
  {
    return dim_sizes_;
  }


  UINT64 Size(int dim)
  {
      return dim_sizes_[dim];
  }

  UINT64 size()
  {
      return values_->size();
  }
  
  UINT64 PrimSize()
  {
    return values_->size();
  }

  long GetStorage( void )
  {
    long sum = 0;

    for(int dim=num_dims_ - 1; dim>=0; dim--) 
    {
      sum += segments_[dim]->Size();
      sum += coordinates_[dim]->Size();
    }
    
    sum += values_->Size();

    return sum;
  }
  
  void Reshape(int num_dims, int nnz = 0)
  {
    num_dims_ = num_dims;
    last_inserted_coordinate_.resize(num_dims_, -1);
    coordinates_.resize(num_dims_, NULL);
    segments_.resize(num_dims_, NULL);
    Init(nnz);
  }

  void Resize(const std::vector<int>& dim_sizes, int nnz = 0)
  {
    // Reverse the dimension sizes.
    dim_sizes_.assign(dim_sizes.rbegin(), dim_sizes.rend());
    num_dims_ = dim_sizes_.size();
    last_inserted_coordinate_.resize(num_dims_, -1);
    coordinates_.resize(num_dims_, NULL);
    segments_.resize(num_dims_, NULL);
    Init(nnz);
  }

  void TrimFat( void )
  {

      for(int dim=num_dims_ - 1; dim>=0; dim--) 
      {
          coordinates_[dim]->PrimShrinkToFit();
          segments_[dim]->PrimShrinkToFit();
      }

      if( coordinates_[0]->Size() != nnz_in_ && nnz_in_ != 0) 
      {
          PrintCompressedTensor();

          T(4) <<"NNZ expected: "<<nnz_in_<< EndT;;
          T(4) <<"CompressedTensor NNZ:      "<<coordinates_[0]->Size()<< EndT;;
          exit(0);
      }
  }

  void PrintCompressedTensor( int print_coords=false )
  {

      whoop::T(0) << "  Printing Compressed tensor - nnz: " << coordinates_[0]->Size() << whoop::EndT;

      T(4) << EndT;
      T(4) <<"Compressed Sparse Fiber (CompressedTensor) Representation Of Graph:"<< EndT;
      T(4) << EndT;

      std::cout <<" Print Compressed Tracer " << std::endl;
      for(int dim=0; dim < num_dims_; dim++)
      {
        std::cout <<"dim: " << dim << std::endl;
        std::cout <<"segment (size " << segments_[dim]->size() << "): ";
        for (int x = 0; x < segments_[dim]->PrimSize(); x++)
        {
          std::cout << segments_[dim]->PrimAt(x) << " ";
        }
        std::cout << std::endl;

        std::cout <<"coord (size " << coordinates_[dim]->size() << "): ";
        for (int x = 0; x < coordinates_[dim]->PrimSize(); x++)
        {
          std::cout << coordinates_[dim]->PrimAt(x) << " ";
        }
        std::cout << std::endl;

        if (dim == (num_dims_-1)) {
          std::cout <<"val (size " << values_->size() << "): ";
          for (int x = 0; x < values_->PrimSize(); x++)
          {
           std::cout << values_->PrimAt(x) << " ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }

      

      /*
      for(int dim=num_dims_ - 1; dim>=0; dim--) 
      {
          T(4) <<"\tDim: "<<dim<<" SegmentArray["<<dim_names_[dim]<<"] = ";
          for(int p=0; print_coords && p<segment_insert_ptr_[dim]+1; p++)
          {
              T(4) <<segments_[dim]->At({p})<<", ";
          }
          T(4) <<" WrPtr: "<<segment_insert_ptr_[dim]<<" Max: "<<segments_[dim]->size()<< EndT;;

          T(4) <<"\tDim: "<<dim<<" CoordinateArray["<<dim_names_[dim]<<"] = ";
          for(int p=0; print_coords && p<coordinates_[dim].size(); p++)
          {
              T(4) <<coordinates_[dim]->At({p})<<", ";
          }
          T(4) <<" Max: "<<coordinates_[dim]->size()<< EndT;;
          T(4) << EndT;;
      }*/
  }

  void Init( int nnz )
  {
      int dim = num_dims_ - 1;        
      int kSegSize, kCoordSize, prev_kCoordSize;
      std::cout << "compressed tensor init for " << name_ << " num_dims_ " << num_dims_ << std::endl;
      values_ = std::shared_ptr<Vec>(new Vec(name_ + "_values"));

      for (int x = 0; x < num_dims_; x++)
      {

        std::cout << "  creating structures for " << name_ << " dim " << x << std::endl;
        segments_[x] = std::shared_ptr<Vec>(new Vec(name_ + "_segments_" + std::to_string(x)));
        coordinates_[x] = std::shared_ptr<Vec>(new Vec(name_ + "_coordinates_" + std::to_string(x)));
      }

      nnz_in_ = nnz;

      // allocate space for the value array equivalent to nnz
      values_->Resize(nnz);

      // Set initial sizes for segment arrays, later
      // it will be resized based on insertions

      for (int x = 0; x < num_dims_; x++)
      {  
        //initialize root segment to be empty (with two values)
        if (x == 0) {
          segments_[x]->Resize(2);
          segments_[x]->At(0) = 0;
          segments_[x]->At(1) = 0;      
        }
        else {
          segments_[x]->Resize(1);
          segments_[x]->At(0) = 0;
        }

        coordinates_[x]->Resize(0);
      }
  }

  void Insert( const std::vector<int>& coord, int val_in=1 )
  {
      //whoop::T(0) << "  DS: Insert begin at : "  << whoop::EndT;

      ASSERT(coord.size() == num_dims_) << "Wrong number of coordinates in point. Expected: " << num_dims_ << ", Got: " << coord.size() << EndT;
      
      // Check that we are inserting points in correct sorted order.
      int strictly_greater = 0;
      for(int dim=coord.size()-1; dim>=0; dim--)
      {
        if (coord[dim] > last_inserted_coordinate_[dim])
        {
          strictly_greater = 1;
          break;
        }
      }
      ASSERT(strictly_greater || values_->PrimSize() == 0) << "Inserting points in unsorted order!" << EndT;

      bool attempt_compress = true;
      for(int dim=0; dim < coord.size(); dim++)
      {
        int index_dim = (coord.size()-1)-dim;
        //whoop::T(0) << "      Coord insert at: " << coord[index_dim] << whoop::EndT;

        //don't duplicate like coordinates of the CSF tree (compress)
        //  traverse the tree until we don't find a suitable coord node
        //    (we work from root down toward leaves with each iteration)
        //  compare the coord to insert with the last item in coord array
        //  we stop compressing when we no longer have a match
        //  this works as insertions must be done in strict ascending order
        if (attempt_compress == false || (coordinates_[dim]->size() == 0) || (coord[index_dim] != coordinates_[dim]->PrimAt(coordinates_[dim]->size()-1)))
        {
          //whoop::T(0) << "      Coord added " << whoop::EndT;
          //add new coordinate
          coordinates_[dim]->PushBack(coord[index_dim]);
  
          //we found a new value at our fiber, increase count local to our fiber
          //whoop::T(0) << "      Segment adjust at : " << (segments_[dim]->PrimSize() - 1) << " newval: " << segments_[dim]->At(segments_[dim]->PrimSize() - 1) << whoop::EndT;
          segments_[dim]->At(segments_[dim]->PrimSize() - 1) += 1; // = coordinates_[dim]->PrimSize();

          //we found a new value - new fiber at next rank.
          //add new segment value to next dim, equal to next dim's current tail 
          if (dim != (num_dims_-1)) 
          {
            segments_[dim+1]->PushBack(segments_[dim+1]->At(segments_[dim+1]->PrimSize() - 1));
            //whoop::T(0) << "      Segment added in next dim at : " << (segments_[dim+1]->PrimSize() - 1) << " newval: " << segments_[dim+1]->At(segments_[dim+1]->PrimSize() -1) << whoop::EndT;
          }

          attempt_compress = false; //we didn't find a match and shouldn't continue to compress subsequent dims
        }
        else {
          //whoop::T(0) << "      Coord existing " << whoop::EndT;
        }


        

        //whoop::T(0) << "  Printing Compressed tensor - nnz: " << coordinates_[dim]->size() << whoop::EndT;

        if( dim == (num_dims_-1) )
        {
            //whoop::T(0) << "      Value added " << whoop::EndT;
            values_->PushBack(val_in);
        }

      }

      last_inserted_coordinate_ = coord;
      //whoop::T(0) << "    DS: Insert end at : " << whoop::EndT;

  }

/*
  bool Locate(const std::vector<Var>& query_coord, const std::vector<Var>& start_position, std::vector<Var>& query_position){

  }

  bool Locate(const std::vector<Var>& query_coord, std::vector<Var>& query_position)
  {
      //whoop::T(0) << "  DS: Insert begin at : "  << whoop::EndT;

      ASSERT(coord.size() == num_dims_) << "Wrong number of coordinates in point. Expected: " << num_dims_ << ", Got: " << coord.size() << EndT;
      
      for(int dim=0; dim < coord.size(); dim++)
      {
        int index_dim = (coord.size()-1)-dim;
        whoop::T(0) << "      Coord insert at: " << coord[index_dim] << whoop::EndT;

        //don't duplicate like coordinates of the CSF tree (compress)
        //  traverse the tree until we don't find a suitable coord node
        //    (we work from root down toward leaves with each iteration)
        //  compare the coord to insert with the last item in coord array
        //  we stop compressing when we no longer have a match
        //  this works as insertions must be done in strict ascending order
        if (attempt_compress == false || (coordinates_[dim]->size() == 0) || (coord[index_dim] != coordinates_[dim]->PrimAt(coordinates_[dim]->size()-1)))
        {
          whoop::T(0) << "      Coord added " << whoop::EndT;
          //add new coordinate
          coordinates_[dim]->PushBack(coord[index_dim]);
  
          //we found a new value at our fiber, increase count local to our fiber
          whoop::T(0) << "      Segment adjust at : " << (segments_[dim]->PrimSize() - 1) << " newval: " << segments_[dim]->At(segments_[dim]->PrimSize() - 1) << whoop::EndT;
          segments_[dim]->At(segments_[dim]->PrimSize() - 1) += 1; // = coordinates_[dim]->PrimSize();

          //we found a new value - new fiber at next rank.
          //add new segment value to next dim, equal to next dim's current tail 
          if (dim != (num_dims_-1)) 
          {
            segments_[dim+1]->PushBack(segments_[dim+1]->At(segments_[dim+1]->PrimSize() - 1));
            whoop::T(0) << "      Segment added in next dim at : " << (segments_[dim+1]->PrimSize() - 1) << " newval: " << segments_[dim+1]->At(segments_[dim+1]->PrimSize() -1) << whoop::EndT;
          }

          attempt_compress = false; //we didn't find a match and shouldn't continue to compress subsequent dims
        }
        else {
          whoop::T(0) << "      Coord existing " << whoop::EndT;
        }


        

        //whoop::T(0) << "  Printing Compressed tensor - nnz: " << coordinates_[dim]->size() << whoop::EndT;

        if( dim == (num_dims_-1) )
        {
            whoop::T(0) << "      Value added " << whoop::EndT;
            values_->PushBack(val_in);
        }

      }

      last_inserted_coordinate_ = coord;
      //whoop::T(0) << "    DS: Insert end at : " << whoop::EndT;

  }

*/
  bool Locate( int d1, int s1, int &os, int &oe )
  {
      int levelsSearched  = 0;
      int currDim = num_dims_ - 1;

      bool found = false;

      int compareCoord = d1;

      int pos_start = 0;
      int pos_end  = coordinates_[currDim]->size();

      while( pos_start < pos_end ) 
      {
          int currCoord = coordinates_[currDim]->At( pos_start );
          if( currCoord == compareCoord )
          {

              int old_pos_start = pos_start;
              levelsSearched++;
              currDim--;
              pos_start = segments_[currDim]->At( old_pos_start );
              pos_end = segments_[currDim]->At( old_pos_start+1 );
              compareCoord = s1;

              if( levelsSearched == 2 )
              {
                  os = pos_start;
                  oe = pos_end;
                  return true;
              }
          }
          else {
              pos_start++;
          }
      }
      return false;
  }


  int AddSegment( int dim_arg, int pos )
  {
      return segments_[dim_arg]->At( {pos} );
  }

  int GetSegmentAt( int dim_arg, int pos )
  {
      return segments_[dim_arg]->At( {pos} );
  }


  int GetSegmentBeginAt(int dim_arg, int pos)
  {
      return segments_[dim_arg]->At( {pos} );
  }

  int GetSegmentEndAt(int dim_arg, int pos)
  {
      return segments_[dim_arg]->At( {pos+1} );
  }


  TensorDisambiguator GetSegmentRootBegin(Var& dummy)
  {
      return (*(segments_[0]))[dummy*0+0];
  }

  TensorDisambiguator GetSegmentRootEnd(Var& dummy)
  {

      return (*(segments_[0]))[dummy*0+1];
  }

  TensorDisambiguator GetSegmentBegin( int dim_arg, Var& pos )
  {
      std::cout << "  SegBeg: " << name_ << " " << dim_arg << std::endl;
      return (*(segments_[dim_arg]))[pos];
  }

  TensorDisambiguator GetSegmentEnd( int dim_arg, Var& pos )
  {
      std::cout << "  SegEnd: " << name_ << " " << dim_arg << std::endl;
      return (*(segments_[dim_arg]))[pos+1];
  }

  

  int GetCoordinateAt( int dim_arg, int pos )
  {
      return coordinates_[dim_arg]->At( {pos} );
  }

  inline TensorDisambiguator GetCoordinate(int dim_arg, Var& pos )
  {
      std::cout << "  Coord: " << name_ << " " << dim_arg << std::endl;
      //return (*(coordinates_[dim_arg]))[pos];
      return (*this->coordinates_[dim_arg])[pos];
  }



  int GetValueAt( int pos )
  {
    return values_->At({pos});
  }

  TensorDisambiguator GetValue( Var& pos )
  {
    std::cout << "  Value: " <<  name_ << std::endl;
    return (*values_)[pos];
  }

  TensorDisambiguator SetValue( Var& pos )
  {
    std::cout << "  Value: " <<  name_ << std::endl;
    return (*values_)[pos];
  }


  void AddSegmentTileLevel(int dim_arg, int size, int shrink_granularity = 0, int granularity = 1, int port = 0)
  {
     (*segments_[dim_arg]).AddTileLevel(size, shrink_granularity, granularity, port);
  }

  void AddCoordinateTileLevel(int dim_arg, int size, int shrink_granularity = 0, int granularity = 1, int port = 0)
  {
     (*coordinates_[dim_arg]).AddTileLevel(size, shrink_granularity, granularity, port);
  }

  void AddValueTileLevel(int size, int shrink_granularity = 0, int granularity = 1, int port = 0)
  {
     (*values_).AddTileLevel(size, shrink_granularity, granularity, port);
  }



  FullyConcordantScanner GetFullyConcordantScanner();
  
  friend class boost::serialization::access;

/*  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & num_dims_;
    ar & nnz_in_;

    ar & dim_sizes_;
    ar & dim_names_;

    ar & last_inserted_coordinate_;

    // Note: serialization of std::shared_ptr by default serializes
    // the underlying object correctly. Thanks boost!
    ar & segments_;
    ar & coordinates_;
    ar & values_;
  }*/
  
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
//      ar & num_dims_; constructor must provide, used for checking later
      ar & nnz_in_;

      ar & dim_sizes_;
      ar & dim_names_;

      ar & last_inserted_coordinate_;

    // Note: serialization of std::shared_ptr by default serializes
    // the underlying object correctly. Thanks boost!

      for (int x = 0; x < num_dims_; x++)
      {  
          ar & segments_[x]->dim_sizes_;
          ar & segments_[x]->vals_;
      }

      for (int x = 0; x < num_dims_; x++)
      {  
          ar & coordinates_[x]->dim_sizes_;
          ar & coordinates_[x]->vals_;
      }

      ar & values_->dim_sizes_;
      ar & values_->vals_;
      
  }  
};

class CompressedTensorIn : public CompressedTensor, public InFromFile
{
 public:
  std::string filename_;
 
  CompressedTensorIn(const std::string& nm, const int num_dims) :
    CompressedTensor(nm, num_dims, {}, {}, 0)
  {
    filename_ = nm + ".in.txt";
    AddOption(&filename_, "tensor_" + nm + "_file", "Input datafile name for CompressedTensorIn " + nm);
    need_input.push_back(this);
  }
  
  void ReadInput(bool is_ref = false)
  {

      std::cout << "ReadInput before for " << name_ << " num_dims_ " << num_dims_ << std::endl;
    std::ifstream ifs;
    ifs.open(filename_);
    if (ifs.fail())
    {
      if (is_ref)
      {
        std::cerr << "Error: CompressedTensorOut " << name_ << " when attempting to read reference data from file: " << filename_ << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        std::cerr << "Note: file name can be set with option --ref_" << name_ << "_file=<filename>" << std::endl;
      }
      else
      {
        std::cerr << "Error: CompressedTensorIn " << name_ << " when attempting to read data from file: " << filename_ << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        std::cerr << "Note: file name can be set with option --tensor_" << name_ << "_file=<filename>" << std::endl;
      }
      std::exit(1);
    }
    boost::archive::text_iarchive ia(ifs);
    ia >> (*this);
    TrimFat();

    assert(dim_sizes_.size() == num_dims_ && "ReadInput: Compressed Tensor mismatch!");

    for (int x = 0; x < num_dims_; x++)
    {
      std::cout << "  fixing up buffermodel sizes" << std::endl;
      segments_[x]->FixupSize();
      coordinates_[x]->FixupSize();
    }
    values_->FixupSize();
  }
};

class CompressedTensorOut : public CompressedTensor, OutToFile
{
 public:
  std::string filename_;
  std::string ref_filename_;

  CompressedTensorOut(const std::string& name, const int num_dims) :
     CompressedTensor(name, num_dims, {}, {}, 0, Type::Out) 
  {
    Init(); //Neal calls local Init only
  }

  void Init()
  {
    whoop::T(0) << "  Init Compressed Tensor Out: " <<  whoop::EndT;
    filename_ = name_ + ".out.txt";
    ref_filename_ = name_ + ".ref.txt";

    AddOption(&filename_, "tensor_" + name_ + "_file", "Output datafile name for CompressedTensorOut " + name_);
    AddOption(&ref_filename_, "ref_" + name_ + "_file", "Reference datafile name for CompressedTensorOut " + name_);
    
    need_output.push_back(this);
  }

  
  void DuplicateFromIn(CompressedTensorIn ref)
  {
    std::ifstream ifs;
    ifs.open(ref.filename_);
    if (ifs.fail())
    {
      std::cerr << "Error: CompressedTensorIn " << name_ << " when attempting to read data from file: " << filename_ << std::endl;
      std::cerr << "Error: " << strerror(errno) << std::endl;
      std::cerr << "Note: file name can be set with option --tensor_" << name_ << "_file=<filename>" << std::endl;
      
      std::exit(1);
    }
    boost::archive::text_iarchive ia(ifs);
    ia >> (*this);
    TrimFat();

    assert(dim_sizes_.size() == num_dims_ && "ReadInput: Compressed Tensor mismatch!");


    for (int x = 0; x < num_dims_; x++)
    {
      std::cout << "  fixing up buffermodel sizes" << std::endl;
      segments_[x]->FixupSize();
      coordinates_[x]->FixupSize();
    }
    values_->FixupSize();

  }

  
  void DumpOutput()
  {
    whoop::T(0) << "  Dumping to File: " <<  whoop::EndT;
    std::ofstream ofs(filename_);
    boost::archive::text_oarchive oa(ofs);
    oa << (*this);
  }
  
  void CheckOutput()
  {
    /* 
    TODO: Write this
    CompressedTensorIn ref_in{name_};
    ref_in.filename_ = ref_filename_;
    ref_in.ReadInput(true);
    
    user_tracer_.T(1) << "Beginning to check CompressedTensorOut " << name_ << " against reference data." << EndT;
    
    user_tracer_.ASSERT_WARNING(ref_in.dim_sizes_.size() == dim_sizes_.size()) << "Number of dimensions mismatch versus reference tensor. Ref: " << ref_in.dim_sizes_.size() << ", This: " << ref_in.dim_sizes_.size() << EndT;

    auto dim_it = dim_sizes_.begin();
    int dim = 0;
    for (auto ref_dim_it = ref_in.dim_sizes_.begin(); ref_dim_it != ref_in.dim_sizes_.end(); ref_dim_it++)
    {
      user_tracer_.ASSERT_WARNING((*dim_it) == (*ref_dim_it)) << "Size mismatch versus reference tensor in dimension: " << ref_in.dim_sizes_.size() - dim << ". Ref: " << ref_in.dim_sizes_.size() << ", This: " << (*dim_it) << EndT;
      dim_it++;
      dim++;
    }

    for (int x = 0; x < ref_in.vals_.size(); x++)
    {
      int ref_v = static_cast<PrimTensor>(ref_in).PrimAt(x);
      int my_v = PrimTensor::PrimAt(x);
      // This construction looks weird but is faster in the common case of a match.
      if (my_v != ref_v)
      {
        std::vector<int> full_idx = UnflattenIndex(x);
        user_tracer_.ASSERT_WARNING(false) << "Data mismatch versus reference tensor, index: " << ast::ShowIndices(full_idx) << ". Ref: " << ref_v << ", This: " << my_v << EndT;
      }
    }
    
    user_tracer_.T(0) << "Reference check done." << EndT;
    */
  }
};


using Point = std::vector<int>;
using PointAndValue = std::pair<Point, int>;



class DomainIterator
{
 protected:
  FullyConcordantScanner* scanner_ = NULL;
  PointAndValue current_;

 public:
   
  explicit DomainIterator(FullyConcordantScanner* s) : scanner_(s) {}
 
  bool operator !=(const DomainIterator& other);
  /*
  {
    // Purposely ignore other, it's a dummy.
    return !scanner_->IsDone();
  }*/
  
  void operator ++();
  /*
  {
    current_ = scanner_->Advance();
  }*/
  
  PointAndValue operator *()
  {
    return current_;
  }
};


class FullyConcordantScanner
{
 public:
  // Target info
  CompressedTensor* target_;
  int num_dims_;
/*
  // hyper-rectangle definition.
  Point base_;
  Point scope_;
  Point tile_repeats_;
*/
  // Iteration state tracking.
  std::vector<int> cur_positions_;
  std::vector<int> cur_segment_bounds_;
  Point cur_point_;
  
  FullyConcordantScanner(CompressedTensor* target) :
    target_(target),
    num_dims_(target->num_dims_),
  /*  base_(num_dims_, 0),
    scope_(target->Sizes()),
    tile_repeats(num_dims_, 1),*/
    cur_positions_(num_dims_),
    cur_segment_bounds_(num_dims_),
    cur_point_(num_dims_, 0)
  {
  }
/*
  void SetRepeat(const std::vector<int>& tile_repeats)
  {
    assert(tile_repeats.size() = cur_point_.size());
    tile_repeats_ =  tile_repeats;
  }
*/
  DomainIterator begin()
  {
    cur_positions_[num_dims_ - 1] = 0;
    cur_segment_bounds_[num_dims_ - 1] = 1;
    // TODO: currently assumes whole tensor traversal.
    // 1) Find base position for each dim
    // 2) Find bound position for each dim
    // 3) Set current position for each dim to base
    // 4) Set segment_bound for each dim
    
    DomainIterator dit(this);
    return dit;
  }

  DomainIterator end()
  {
    // unused dummy
    DomainIterator dit(NULL);
    return dit;
  }
  
  PointAndValue Advance()
  {
    for (int dim = 0; dim < num_dims_; dim++)
    {
      cur_positions_[dim]++;
      cur_point_[dim] = target_->GetCoordinateAt(dim, cur_positions_[dim]);
      if (cur_positions_[dim] != cur_segment_bounds_[dim])
      {
        break; // If any inner dim wasn't done, the remaining outer dims won't be done either.
      }
      cur_segment_bounds_[dim] = target_->GetSegmentAt(dim, cur_positions_[dim+1]+1);
    }

    int cur_value = target_->GetValueAt(cur_positions_[0]);

    return std::make_pair(cur_point_, cur_value);
  }
  
  bool IsDone()
  {
    return cur_positions_[num_dims_ - 1] == cur_segment_bounds_[num_dims_ - 1];
  }

  
};


FullyConcordantScanner CompressedTensor::GetFullyConcordantScanner()
{
  return FullyConcordantScanner(this);
}



bool DomainIterator::operator !=(const DomainIterator& other)
{
  // Purposely ignore other, it's a dummy.
  return !scanner_->IsDone();
}

void DomainIterator::operator ++()
{
  current_ = scanner_->Advance();
}

} // namespace whoop

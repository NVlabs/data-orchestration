/* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION & AFFILIATES nor the names of its
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


#include <queueda.hpp>
#include <simple-tensor.hpp>
using namespace queueda;


// Static options
static const Coordinate K0 = 8; // UNROLLING_FACTOR


// Dynamic options, shared between host and device,
// with default values.
namespace opt {

struct DynamicOptions {
  Coordinate M1 = 8; // WARPS_PER_BLOCK
  Coordinate N2 = 1; // BLOCKS_PER_GPU
  Coordinate N0 = 32; // THREADS_PER_WARP
};

DynamicOptions* host_;
__CUDA_DEVICE__ DynamicOptions* device_;

}

// Globals
__CUDA_DEVICE__
Coordinate M__;
__CUDA_DEVICE__
Coordinate K__;
__CUDA_DEVICE__
Coordinate N__;

class Loop1 : public Node {
 
 Coordinate n2_;
 Coordinate m1_;
 Coordinate n0_;

 Value* a_;
 Value* b_;
 Value* z_;
 
 public:
 
  __CUDA_DEVICE__
  Loop1 (
    Coordinate n2,
    Coordinate m1,
    Coordinate n0,
    Value* a,
    Value* b,
    Value* z,
    Name instance_name = "Loop1"
  ) 
  : Node(instance_name),
    n2_(n2),
    m1_(m1),
    n0_(n0),
    a_(a),
    b_(b),
    z_(z) {
    
    //assert(M__ % M1 == 0);
    //assert(N__ % (N2*N0) == 0);
  }
  
  ~Loop1() = default;

  __CUDA_DEVICE__
  virtual void Run() {
    Trace(2, "Begin Run.");
    Coordinate n0 = n0_; // Thread ID
    Coordinate m1 = m1_; // Warp ID
    Coordinate n2 = n2_; // Block ID
    
    Coordinate M1 = opt::device_->M1;
    Coordinate N2 = opt::device_->N2;
    Coordinate N0 = opt::device_->N0;
  
    Coordinate M0(M__/M1);
    Coordinate N1(N__/(N2 * N0));
    Coordinate K1(K__/K0);
    
    // First Loop: High compute intensity
    for (Coordinate n1 = 0; n1 < N1; n1++) {
      Coordinate n = n2 * N1 * N0 + n1 * N0 + n0;
      for (Coordinate m0 = 0; m0 < M0; m0++) {
        Coordinate m = m1 * M0 + m0;
        Value tmp = 0;
        for (Coordinate k1 = 0; k1 < K1; k1++) {
          #pragma unroll
          for (Coordinate k0 = 0; k0 < K0; k0++) {
            Coordinate k = k1 * K0 + k0;
            Trace(4, "Loop 1: Iteration %d, %d, %d: %d += %d * %d", m, n, k, tmp, a_[m * K__ + k], b_[k * N__ + n]);
            tmp += a_[m * K__ + k] * b_[k * N__ + n];
          }
        }
        z_[m * N__ + n] = tmp;
        Trace(3, "Loop 1: Finish K");
      }
      Trace(3, "Loop 1: Finish M0");
    }
    Trace(3, "Loop 1: Finish N1");
    Trace(2, "Done.");
  }
};
  
class Loop2 : public Node {
 
 Coordinate n2_;
 Coordinate m1_;
 Coordinate n0_;

 Value* a_;
 Value* b_;
 Value* z_;
 Value* y_;
 
 public:
 
  __CUDA_DEVICE__
  Loop2 (
    Coordinate n2,
    Coordinate m1,
    Coordinate n0,
    Value* a,
    Value* b,
    Value* z,
    Value* y,
    Name instance_name = "Loop2"
  ) 
  : Node(instance_name),
    n2_(n2),
    m1_(m1),
    n0_(n0),
    a_(a),
    b_(b),
    z_(z),
    y_(y) {
    
    //assert(M__ % M1 == 0);
    //assert(N__ % N2*nN0) == 0);
  }
  
  ~Loop2() = default;

  __CUDA_DEVICE__
  virtual void Run() {
  
    Trace(2, "Begin Run.");
    Coordinate n0 = n0_; // Thread ID
    Coordinate m1 = m1_; // Warp ID
    Coordinate n2 = n2_; // Block ID
    
    Coordinate M1 = opt::device_->M1;
    Coordinate N2 = opt::device_->N2;
    Coordinate N0 = opt::device_->N0;

    Coordinate M0(M__/M1);
    Coordinate N1(N__/(N2*N0));

    // Second Loop: Low compute intensity
    for (Coordinate n1 = 0; n1 < N1; n1++) {
      Coordinate n = n2 * N1 * N0 + n1 * N0 + n0;
      for (Coordinate m0 = 0; m0 < M0; m0++) {
        Coordinate m = m1 * M0 + m0;
        y_[m * N__ + n] = z_[m * N__ + n] * 17;
        Trace(4, "Loop 2: Iteration %d, %d", m, n);
      }
      Trace(3, "Loop 2: Finish M0");
    }
    Trace(3, "Loop 2: Finish N1");
    Trace(2, "Done.");
  }
};


__CUDA_DEVICE__
void
BaselineTest(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* z,
  Value* y
  ) {
  

  Coordinate M1 = opt::device_->M1;
  Coordinate N2 = opt::device_->N2;
  Coordinate N0 = opt::device_->N0;

  for (Coordinate n2 = 0; n2 < N2; n2++) {
    for (Coordinate m1 = 0; m1 < M1; m1++) {
      for (Coordinate n0 = 0; n0 < N0; n0++) {
        printf("Build %d, %d, %d\n", n2, m1, n0);
        auto l1 = new Loop1(n2, m1, n0, a, b, z);
        l1->Bind(n2, m1, n0);
        auto l2 = new Loop2(n2, m1, n0, a, b, z, y);
        l2->Bind(n2, m1, n0);
      }
    }
  }
}


__CUDA_GLOBAL__
void 
BTBuildKernel(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* z,
  Value* y) {
   
  BuilderFunction build_func = 
    [M, K, N, a, b, z, y] () {
      BaselineTest(M, K, N, a, b, z, y);
    };

  queueda::Build(build_func);
}


__CUDA_GLOBAL__
void 
BTKernel(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* z,
  Value* y) {
   
  M__ = M;
  K__ = K;
  N__ = N;

  queueda::Run();
}

inline
int
RunBT(Coordinate M, Coordinate K, Coordinate N) {

  queueda::Init();
  
  SimpleTensor* af = new SimpleTensor({M, K}, "A");
  SimpleTensor* bf = new SimpleTensor({K, N}, "B");
  SimpleTensor* zf = new SimpleTensor({M, N}, "Z");
  SimpleTensor* yf = new SimpleTensor({M, N}, "Y");
  
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      af->v_[m * K + k] = rand() % 255;
    }
  }
  //af->Print();
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      bf->v_[k * N + n] = rand() % 255;
    }
  }
  //bf->Print();
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      zf->v_[m * N + n] = 0;
      yf->v_[m * N + n] = 0;
    }
  }

#ifdef __CUDACC__

  auto ad = AllocArrayOnDevice<Value>(M * K);
  auto bd = AllocArrayOnDevice<Value>(K * N);
  auto zd = AllocArrayOnDevice<Value>(M * N);
  auto yd = AllocArrayOnDevice<Value>(M * N);

  SetDeviceArray<Value>(ad, af->v_, M * K);
  SetDeviceArray<Value>(bd, bf->v_, K * N);
  SetDeviceArray<Value>(zd, zf->v_, M * N);
  SetDeviceArray<Value>(yd, yf->v_, M * N);
  
  Coordinate M1 = opt::host_->M1;
  Coordinate N2 = opt::host_->N2;

  BTBuildKernel<<<1, 1>>>(M, K, N, ad, bd, zd, yd);

  BTKernel<<<N2, M1*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  /*BTKernel<<<N2, M1*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  BTKernel<<<N2, M1*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  BTKernel<<<N2, M1*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
*/
  cudaDeviceSynchronize();
  SetHostArray<Value>(yf->v_, yd, M * N);

#else

  
  BTBuildKernel(M, K, N, af->v_, bf->v_, zf->v_, yf->v_);
  BTKernel(M, K, N, af->v_, bf->v_, zf->v_, yf->v_);

#endif


  yf->Print();
  
  SimpleTensor* y_ref = new SimpleTensor({M, N}, "Y_REF");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      y_ref->v_[m * N + n] = 0;
      for (int k = 0; k < K; k++) {
        y_ref->v_[m * N + n] += af->v_[m * K + k] * bf->v_[k * N + n];
      }
      y_ref->v_[m * N + n] = y_ref->v_[m * N + n] * 17;
    }   
  }
  
  return yf->CheckMismatches(y_ref);

}



int 
main(int argc, char** argv) {
  
  opt::host_ = new opt::DynamicOptions();
  
  if (argc > 1)
  {
    opt::host_->N2 = std::atoi(argv[1]);
  }
  if (argc > 2)
  {
    opt::host_->M1 = std::atoi(argv[2]);
  }
  if (argc > 3)
  {
    opt::host_->N0 = std::atoi(argv[3]);
  }
  
  Coordinate NUM_PASSES = 4;
  
  if (argc > 4)
  {
    NUM_PASSES = std::atoi(argv[4]);
  }
#ifdef __CUDACC__

  auto dev_opt = AllocOnDevice<opt::DynamicOptions>();
  SetDeviceValue(dev_opt, opt::host_);
  gpuErrchk(cudaMemcpyToSymbol(opt::device_, &dev_opt, sizeof(opt::DynamicOptions*)));

#else
  opt::device_ = opt::host_;
#endif
  
  Coordinate M = opt::host_->M1 * NUM_PASSES;
  Coordinate K = 8;
  Coordinate N = opt::host_->N2 * NUM_PASSES * opt::host_->N0;
  
  Coordinate a_size = M * K * sizeof(Value);
  Coordinate b_size = K * N * sizeof(Value);
  Coordinate z_size = M * N * sizeof(Value);

  printf("M1: %'d, M0: %'d, N2: %'d, N1: %'d, N0: %'d\n", opt::host_->M1, NUM_PASSES, opt::host_->N2, NUM_PASSES, opt::host_->N0);
  printf("M: %'d, K: %'d, N: %'d\n", M, K, N);
  printf("Size of A in bytes: %'d\n", a_size);
  printf("Size of B in bytes: %'d\n", b_size);
  printf("Size of Z in bytes: %'d\n", z_size);
  printf("Total GPU memory footprint in bytes: %'d\n", a_size + b_size + z_size);
  printf("Total Muls: %'d\n", M * K * N);
  return RunBT(M, K, N);
}

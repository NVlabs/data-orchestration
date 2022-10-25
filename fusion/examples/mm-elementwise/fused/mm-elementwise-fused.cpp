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


static const Coordinate N0 = 32; // THREADS_PER_WARP
static const Coordinate M1 = 8; // WARPS_PER_BLOCK
static const Coordinate N2 = 1; // BLOCKS_PER_GPU


// Globals
__CUDA_DEVICE__
Coordinate M__;
__CUDA_DEVICE__
Coordinate K__;
__CUDA_DEVICE__
Coordinate N__;


class Loop1FusedLoop2 : public Node {
 
 Coordinate n2_;
 Coordinate m1_;
 Coordinate n0_;

 Value* a_;
 Value* b_;
 Value* y_;
 
 public:
 
  __CUDA_DEVICE__
  Loop1FusedLoop2 (
    Coordinate n2,
    Coordinate m1,
    Coordinate n0,
    Value* a,
    Value* b,
    Value* y,
    Name instance_name = "Fused"
  ) 
  : Node(instance_name),
    n2_(n2),
    m1_(m1),
    n0_(n0),
    a_(a),
    b_(b),
    y_(y) {
    
    //assert(M__ % (M2*M0) == 0);
    //assert(N__ % (N3*N1*N0) == 0);
  }
  
  ~Loop1FusedLoop2() = default;

  __CUDA_DEVICE__
  virtual void Run() {
    Trace(2, "Begin Run.");
    Coordinate n0 = n0_; // Thread ID
    Coordinate m1 = m1_; // Warp ID
    Coordinate n2 = n2_; // Block ID
    
    Coordinate M0(M__/M1);
    Coordinate N1(N__/(N2*N0));
    Coordinate K(K__);
    
    for (Coordinate n1 = 0; n1 < N1; n1++) {
      Coordinate n = n2 * N1 * N0 + n1 * N0 + n0;
      for (Coordinate m0 = 0; m0 < M0; m0++) {
        Coordinate m = m1 * M0 + m0;
        Value tmp = 0;
        // First Loop: High compute intensity
        for (Coordinate k = 0; k < K; k++) {
          Trace(4, "Loop 1: Iteration %d, %d, %d: %d += %d * %d", m, n, k, tmp, a_[m * K__ + k], b_[k * N__ + n]);
          tmp += a_[m * K__ + k] * b_[k * N__ + n];
        }
        Trace(3, "Loop 1: Finish K");
        // Second Loop: Low compute intensity
        y_[m * N__ + n] = tmp * 17;
        Trace(4, "Loop 2: Iteration %d, %d", m, n);
      }
      Trace(3, "Loop 1+2: Finish M0");
    }
    Trace(3, "Loop 1+2: Finish N1");
    Trace(2, "Done.");
  }
};


__CUDA_DEVICE__
void
FusionTest(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* y
  ) {
  
  M__ = M;
  K__ = K;
  N__ = N;

  for (Coordinate n2 = 0; n2 < N2; n2++) {
    for (Coordinate m1 = 0; m1 < M1; m1++) {
      for (Coordinate n0 = 0; n0 < N0; n0++) {
        auto fused = new Loop1FusedLoop2(n2, m1, n0, a, b, y);
        printf("Build %d, %d, %d\n", n2, m1, n0);
        fused->Bind(n2, m1, n0);
      }
    }
  }
}


__CUDA_GLOBAL__
void 
FTKernel(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* y) {
   
  BuilderFunction build_func = 
    [M, K, N, a, b, y] () {
      FusionTest(M, K, N, a, b, y);
    };

  queueda::Build(build_func);
  queueda::Run();
}


inline
int
RunFT(Coordinate M, Coordinate K, Coordinate N) {

  SimpleTensor* af = new SimpleTensor({M, K}, "A");
  SimpleTensor* bf = new SimpleTensor({K, N}, "B");
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
      yf->v_[m * N + n] = 0;
    }
  }

#ifdef __CUDACC__

  auto ad = AllocArrayOnDevice<Value>(M * K);
  auto bd = AllocArrayOnDevice<Value>(K * N);
  auto yd = AllocArrayOnDevice<Value>(M * N);
  SetDeviceArray<Value>(ad, af->v_, M * K);
  SetDeviceArray<Value>(bd, bf->v_, K * N);
  SetDeviceArray<Value>(yd, yf->v_, M * N);
  
  FTKernel<<<N2, M1*N0>>>(M, K, N, ad, bd, yd);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  SetHostArray<Value>(yf->v_, yd, M * N);

#else

  
  FTKernel(M, K, N, af->v_, bf->v_, yf->v_);

#endif


  yf->Print();
  
  SimpleTensor* y_ref = new SimpleTensor({M, N}, "Y_REF");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
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
  
  static const Coordinate NUM_PASSES = 4;
  return RunFT(M1 * NUM_PASSES, 8, N2 * NUM_PASSES * N0);
}

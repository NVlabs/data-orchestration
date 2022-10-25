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
static const Coordinate M2 = 8; // WARPS_PER_BLOCK
static const Coordinate N3 = 1; // BLOCKS_PER_GPU
static const Coordinate N1 = 4; // Buffer N
static const Coordinate M0 = 4; // Buffer M


// Globals
__CUDA_DEVICE__
Coordinate M__;
__CUDA_DEVICE__
Coordinate K__;
__CUDA_DEVICE__
Coordinate N__;

 
template<
  int M0,
  int N1
>
class Loop1FusedLoop2 : public Node {
 
 Coordinate n3_;
 Coordinate m2_;
 Coordinate n0_;

 Value* a_;
 Value* b_;
 Value* z_;
 
 Value buffer_[N1][M0]; // TODO: Shared

 public:
 
  __CUDA_DEVICE__
  Loop1FusedLoop2 (
    Coordinate n3,
    Coordinate m2,
    Coordinate n0,
    Value* a,
    Value* b,
    Value* z,
    Name instance_name = "Fused"
  ) 
  : Node(instance_name),
    n3_(n3),
    m2_(m2),
    n0_(n0),
    a_(a),
    b_(b),
    z_(z) {
    
    //assert(M__ % (M2*M0) == 0);
    //assert(N__ % (N3*N1*N0) == 0);
  }
  
  ~Loop1FusedLoop2() = default;

  __CUDA_DEVICE__
  virtual void Run() {
    Trace(2, "Begin Run.");
    Coordinate n0 = n0_; // Thread ID
    Coordinate m2 = m2_; // Warp ID
    Coordinate n3 = n3_; // Block ID
    
    Coordinate M1(M__/(M2*M0));
    Coordinate N2(N__/(N3*N1*N0));
    Coordinate K(K__);
    
    for (Coordinate n2 = 0; n2 < N2; n2++) {
      for (Coordinate m1 = 0; m1 < M1; m1++) {
        // First Loop: High compute intensity
        for (Coordinate k = 0; k < K; k++) {
          for (Coordinate n1 = 0; n1 < N1; n1++) {
            Coordinate n = n3 * N2 * N1 * N0 + n2 * N1 * N0 + n1 * N0 + n0;
            for (Coordinate m0 = 0; m0 < M0; m0++) {
              Coordinate m = m2 * M1 * M0 + m1 * M0 + m0;
              Trace(4, "Loop 1: Iteration %d, %d, %d: %d += %d * %d", m, n, k, buffer_[n1][m0], a_[m * K__ + k], b_[k * N__ + n]);
              buffer_[n1][m0] += a_[m * K__ + k] * b_[k * N__ + n];
            }
            Trace(3, "Loop 1: Finish M0");
          }
          Trace(3, "Loop 1: Finish N1");
        }
        Trace(3, "Loop 1: Finish K");
        // Second Loop: Low compute intensity
        for (Coordinate n1 = 0; n1 < N1; n1++) {
          Coordinate n = n3 * N2 * N1 * N0 + n2 * N1 * N0 + n1 * N0 + n0;
          for (Coordinate m0 = 0; m0 < M0; m0++) {
            Coordinate m = m2 * M1 * M0 + m1 * M0 + m0;
            z_[m * N__ + n] = buffer_[n1][m0] * 17;
            buffer_[n1][m0] = 0;
            Trace(4, "Loop 2: Iteration %d, %d", m, n);
          }
          Trace(3, "Loop 2: Finish M0");
        }
        Trace(3, "Loop 2: Finish N1");
      }
      Trace(3, "Loop: Finish M1");
    }
    Trace(3, "Loop: Finish N2");
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
  Value* z
  ) {
  
  M__ = M;
  K__ = K;
  N__ = N;

  for (Coordinate n3 = 0; n3 < N3; n3++) {
    for (Coordinate m2 = 0; m2 < M2; m2++) {
      for (Coordinate n0 = 0; n0 < N0; n0++) {
        auto fused = new Loop1FusedLoop2<N1, M0>(n3, m2, n0, a, b, z);
        printf("Build %d, %d, %d\n", n3, m2, n0);
        fused->Bind(n3, m2, n0);
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
  Value* z) {
   
  BuilderFunction build_func = 
    [M, K, N, a, b, z] () {
      FusionTest(M, K, N, a, b, z);
    };

  queueda::Build(build_func);
  queueda::Run();
}


inline
int
RunFT(Coordinate M, Coordinate K, Coordinate N) {

  SimpleTensor* af = new SimpleTensor({M, K}, "A");
  SimpleTensor* bf = new SimpleTensor({K, N}, "B");
  SimpleTensor* zf = new SimpleTensor({M, N}, "Z");
  
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
    }
  }

#ifdef __CUDACC__

  auto ad = AllocArrayOnDevice<Value>(M * K);
  auto bd = AllocArrayOnDevice<Value>(K * N);
  auto zd = AllocArrayOnDevice<Value>(M * N);
  SetDeviceArray<Value>(ad, af->v_, M * K);
  SetDeviceArray<Value>(bd, bf->v_, K * N);
  SetDeviceArray<Value>(zd, zf->v_, M * N);
  
  FTKernel<<<N3, M2*N0>>>(M, K, N, ad, bd, zd);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  SetHostArray<Value>(zf->v_, zd, M * N);

#else

  
  FTKernel(M, K, N, af->v_, bf->v_, zf->v_);

#endif


  zf->Print();
  
  SimpleTensor* z_ref = new SimpleTensor({M, N}, "Z_REF");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        z_ref->v_[m * N + n] += af->v_[m * K + k] * bf->v_[k * N + n];
      }
      z_ref->v_[m * N + n] = z_ref->v_[m * N + n] * 17;
    }   
  }
  
  return zf->CheckMismatches(z_ref);

}



int 
main(int argc, char** argv) {
  
  static const Coordinate NUM_PASSES = 4;
  RunFT(M2 * NUM_PASSES * M0, 8, N3 * NUM_PASSES * N1 * N0);
}

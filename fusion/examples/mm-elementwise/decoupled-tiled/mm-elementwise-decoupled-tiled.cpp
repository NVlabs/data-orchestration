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
static const Coordinate M0 = 4; // Buffer M, and UNROLLING_FACTOR


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
class Loop1 : public Node {
 
  Coordinate n3_;
  Coordinate m2_;
  Coordinate n0_;

  Value* a_;
  Value* b_;
 
  Value buffer_[N1][M0]; // TODO: Shared
 
 public:
 
  QO<Value, 4> z_q_{this}; // TODO: combine with buffer_ to save memory and a copy().
 
 
  __CUDA_DEVICE__
  Loop1 (
    Coordinate n3,
    Coordinate m2,
    Coordinate n0,
    Value* a,
    Value* b,
    Name instance_name = "Loop1"
  ) 
  : Node(instance_name),
    n3_(n3),
    m2_(m2),
    n0_(n0),
    a_(a),
    b_(b) {
    
    //assert(M__ % (M2*M0) == 0);
    //assert(N__ % (N3*N1*N0) == 0);
  }
  
  ~Loop1() = default;

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
            #pragma unroll
            for (Coordinate m0 = 0; m0 < M0; m0++) {
              Coordinate m = m2 * M1 * M0 + m1 * M0 + m0;
              Trace(4, "Loop 1: Iteration %d, %d, %d: %d += %d * %d", m, n, k, buffer_[n1][m0], a_[m * K__ + k], b_[k * N__ + n]);
              buffer_[n1][m0] += a_[m * K__ + k] * b_[k * N__ + n];
            }
            Trace(3, "Loop 1: Finish M0");
          }
          Trace(3, "Loop 1: Finish N1");
        }
        // Copy final sums to queue:
        for (Coordinate n1 = 0; n1 < N1; n1++) {
          #pragma unroll
          for (Coordinate m0 = 0; m0 < M0; m0++) {
             z_q_.Push(buffer_[n1][m0]);
             buffer_[n1][m0] = 0;
          }
          Trace(3, "Loop 1: Finish M0 (2)");
          z_q_.Finish();
        }
        z_q_.Finish();
        Trace(3, "Loop 1: Finish N1 (2)");
      }
      z_q_.Finish();
      Trace(3, "Loop 1: Finish M1");
    }
    z_q_.Finish();
    Trace(3, "Loop 1: Finish N2");
    Trace(2, "Done.");
  }
};
  
template<
  int M0,
  int N1
>
class Loop2 : public Node {
 
  Coordinate n3_;
  Coordinate m2_;
  Coordinate n0_;

  Value* a_;
  Value* b_;
  Value* y_;

  QI<Value, 4> z_q_;
 
 public:
 
  __CUDA_DEVICE__
  Loop2 (
    Coordinate n3,
    Coordinate m2,
    Coordinate n0,
    Value* a,
    Value* b,
    Value* y,
    QO<Value, 4>* z_q,
    Name instance_name = "Loop2"
  ) 
  : Node(instance_name),
    n3_(n3),
    m2_(m2),
    n0_(n0),
    a_(a),
    b_(b),
    y_(y),
    z_q_(z_q) {
    
    //assert(M__ % M1 == 0);
    //assert(N__ % N2*nN0) == 0);
  }
  
  ~Loop2() = default;

  __CUDA_DEVICE__
  virtual void Run() {
    Trace(2, "Begin Run.");
    Coordinate n0 = n0_; // Thread ID
    Coordinate m2 = m2_; // Warp ID
    Coordinate n3 = n3_; // Block ID
    
    Coordinate M1(M__/(M2*M0));
    Coordinate N2(N__/(N3*N1*N0));

    // Second Loop: Low compute intensity
    Coordinate m0 = 0;
    Coordinate n1 = 0;
    Coordinate m1 = 0;
    Coordinate n2 = 0;
    
    while (!z_q_.IsDone()) {
      while (!z_q_.IsDone()) {
        while (!z_q_.IsDone()) {
          Coordinate n = n3 * N2 * N1 * N0 + n2 * N1 * N0 + n1 * N0 + n0;
          while (!z_q_.IsDone()) {
            Coordinate m = m2 * M1 * M0 + m1 * M0 + m0;
            Trace(4, "Loop 2: Iteration %d, %d", m, n);
            y_[m * N__ + n] = z_q_.Pop() * 17;
            m0++;
          }
          Trace(3, "Loop 2: Finish M0");
          z_q_.Resume();
          m0 = 0;
          n1++;
        }
        Trace(3, "Loop 2: Finish N1");
        z_q_.Resume();
        n1 = 0;
        m1++;
      }
      Trace(3, "Loop 2: Finish M1");
      z_q_.Resume();
      m1 = 0;
      n2++;
    }
    Trace(3, "Loop 2: Finish N2");
    z_q_.Resume();
    Trace(2, "Done.");
  }
};

 
__CUDA_DEVICE__
void
DecoupledTest(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* z,
  Value* y
  ) {
  
  M__ = M;
  K__ = K;
  N__ = N;

  for (Coordinate n3 = 0; n3 < N3; n3++) {
    Loop1<M0, N1>* l1[M2][N0];
    for (Coordinate m2 = 0; m2 < M2; m2++) {
      for (Coordinate n0 = 0; n0 < N0; n0++) {
        printf("Build1 %d, %d, %d\n", n3, m2, n0);
        l1[m2][n0] = new Loop1<M0, N1>(n3, m2, n0, a, b);
        l1[m2][n0]->Bind(n3, m2, n0);
      }
    }
    for (Coordinate m2 = 0; m2 < M2; m2++) {
      for (Coordinate n0 = 0; n0 < N0; n0++) {
        printf("Build2 %d, %d, %d\n", n3, m2, n0);
        auto l2 = new Loop2<M0, N1>(n3, m2, n0, a, b, y, &l1[m2][n0]->z_q_);
        l2->Bind(n3, m2+M2, n0);
      }
    }
  }
}


__CUDA_GLOBAL__
void 
DTKernel(
  Coordinate M,
  Coordinate K,
  Coordinate N,
  Value* a,
  Value* b,
  Value* z,
  Value* y) {
   
  BuilderFunction build_func = 
    [M, K, N, a, b, z, y] () {
      DecoupledTest(M, K, N, a, b, z, y);
    };

  queueda::Build(build_func);
  queueda::Run();
}


inline
int
RunDT(Coordinate M, Coordinate K, Coordinate N) {

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
  
  DTKernel<<<N3, M2*2*N0>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  SetHostArray<Value>(yf->v_, yd, M * N);

#else

  
  DTKernel(M, K, N, af->v_, bf->v_, zf->v_, yf->v_);

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
  
  static const Coordinate NUM_PASSES = 4;
  RunDT(M2 * NUM_PASSES * M0, 8, N3 * NUM_PASSES * N1 * N0);
}

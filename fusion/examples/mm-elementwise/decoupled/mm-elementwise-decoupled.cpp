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

__CUDA_DEVICE__
Value* a_;
__CUDA_DEVICE__
Value* b_;
__CUDA_DEVICE__
Value* z_;
__CUDA_DEVICE__
Value* y_;

__CUDA_DEVICE__
void Loop1(QO<Value, 2>* z_q) {

  Trace(2, "Begin Loop1.");
  Coordinate n0 = GetThread();
  Coordinate m1 = GetWarp();
  Coordinate n2 = GetBlock();

  Coordinate M1 = opt::device_->M1;
  Coordinate N2 = opt::device_->N2;
  Coordinate N0 = opt::device_->N0;

  Coordinate M0(M__/M1);
  Coordinate N1(N__/(N2*N0));
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
      z_q->Push(tmp);
      Trace(3, "Loop 1: Finish K");
    }
    z_q->Finish();
    Trace(3, "Loop 1: Finish M0");
  }
  z_q->Finish();
  Trace(3, "Loop 1: Finish N1");
  Trace(2, "Done.");
}

__CUDA_DEVICE__
void Loop2(QI<Value, 2>* z_q) {

  Trace(2, "Begin Loop2.");

  Coordinate M1 = opt::device_->M1;
  Coordinate N2 = opt::device_->N2;
  Coordinate N0 = opt::device_->N0;

  Coordinate n0 = GetThread();
  Coordinate m1 = GetWarp() - M1;
  Coordinate n2 = GetBlock();

  Coordinate M0(M__/M1);
  Coordinate N1(N__/(N2*N0));

  // Second Loop: Low compute intensity
  Coordinate m0 = 0;
  Coordinate n1 = 0;

  while (!z_q->IsDone()) {
    Coordinate n = n2 * N1 * N0 + n1 * N0 + n0;
    while (!z_q->IsDone()) {
      Coordinate m = m1 * M0 + m0;
      Trace(4, "Loop 2: Iteration %d, %d", m, n);
      y_[m * N__ + n] = z_q->Pop() * 17;
      m0++;
    }
    Trace(3, "Loop 2: Finish M0");
    z_q->Resume();
    m0 = 0;
    n1++;
  }
  Trace(3, "Loop 2: Finish N1");
  z_q->Resume();
  Trace(2, "Done.");

}
 
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

  Coordinate M1 = opt::device_->M1;
  Coordinate N0 = opt::device_->N0;
  
  auto qo = static_cast<QO<Value, 2>**>(std::malloc(M1 * N0 * sizeof(QO<Value, 2>*)));
  auto qi = static_cast<QI<Value, 2>**>(std::malloc(M1 * N0 * sizeof(QI<Value, 2>*)));
  
  for (Coordinate m1 = 0; m1 < M1; m1++) {
    for (Coordinate n0 = 0; n0 < N0; n0++) {
      qo[m1 * N0 + n0] = new QO<Value, 2>();
      auto wrapped_loop1 = [=]() { return Loop1(qo[m1 * N0 + n0]); };
      queueda::Bind(GetBlock(), m1, n0, wrapped_loop1);
    }
  }
  for (Coordinate m1 = 0; m1 < M1; m1++) {
    for (Coordinate n0 = 0; n0 < N0; n0++) {
      qi[m1 * N0 + n0] = new QI<Value, 2>(qo[m1 * N0 + n0]);
      auto wrapped_loop2 = [=]() { return Loop2(qi[m1 * N0 + n0]); };
      queueda::Bind(GetBlock(), m1 + M1, n0, wrapped_loop2);
    }
  }
}


__CUDA_GLOBAL__
void 
DTBuildKernel(
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
   
  M__ = M;
  K__ = K;
  N__ = N;
  a_ = a;
  b_ = b;
  z_ = z;
  y_ = y;

  queueda::Run();
}

inline
int
RunDT(Coordinate M, Coordinate K, Coordinate N) {

  Coordinate M1 = opt::host_->M1;
  Coordinate N2 = opt::host_->N2;
  Coordinate N0 = opt::host_->N0;

  queueda::Init(N2, M1, N0);
  
  SimpleTensor* af = new SimpleTensor({M, K}, "A", {0, 255});
  SimpleTensor* bf = new SimpleTensor({K, N}, "B", {0, 255});
  SimpleTensor* zf = new SimpleTensor({M, N}, "Z");
  SimpleTensor* yf = new SimpleTensor({M, N}, "Y");
  
  auto ad = af->CopyArrayToDevice();
  auto bd = bf->CopyArrayToDevice();
  auto zd = zf->CopyArrayToDevice();
  auto yd = yf->CopyArrayToDevice();
  
#ifdef __CUDACC__

  DTBuildKernel<<<N2, 1>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  DTKernel<<<N2, M1*2*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  DTKernel<<<N2, M1*2*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  DTKernel<<<N2, M1*2*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, zd, yd);
  gpuErrchk(cudaPeekAtLastError());

  cudaDeviceSynchronize();
#else

  DTBuildKernel(M, K, N, ad, bd, zd, yd);
  DTKernel(M, K, N, ad, bd, zd, yd);

#endif


  yf->CopyArrayFromDevice(yd);
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
  
  Coordinate NUM_PASSES_M = 4;
  
  if (argc > 4)
  {
    NUM_PASSES_M = std::atoi(argv[4]);
  }

  Coordinate NUM_PASSES_N = 5;
  
  if (argc > 5)
  {
    NUM_PASSES_N = std::atoi(argv[5]);
  }

  Coordinate K = 8;

  if (argc > 6)
  {
    K = std::atoi(argv[6]);
  }
  
  SET_DEVICE_OPTIONS(opt::DynamicOptions, opt::device_, opt::host_);

  Coordinate M = opt::host_->M1 * NUM_PASSES_M;
  Coordinate N = opt::host_->N2 * NUM_PASSES_N * opt::host_->N0;
  
  Coordinate a_size = M * K * sizeof(Value);
  Coordinate b_size = K * N * sizeof(Value);
  Coordinate z_size = M * N * sizeof(Value);
  Coordinate num_muls = M * K * N;

  printf("M1: %'d, M0: %'d, N2: %'d, N1: %'d, N0: %'d\n", opt::host_->M1, NUM_PASSES_M, opt::host_->N2, NUM_PASSES_N, opt::host_->N0);
  printf("M: %'d, K: %'d, N: %'d\n", M, K, N);
  printf("Size of A in bytes: %'d\n", a_size);
  printf("Size of B in bytes: %'d\n", b_size);
  printf("Size of Z in bytes: %'d\n", z_size);
  printf("Total GPU memory footprint in bytes: %'d\n", a_size + b_size + z_size);
  printf("Total Muls: %'d\n", num_muls);

  return RunDT(M, K, N);
}

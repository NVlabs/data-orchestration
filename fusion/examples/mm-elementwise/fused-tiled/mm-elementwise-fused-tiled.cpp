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


// Dynamic options, shared between host and device,
// with default values.
namespace opt {

struct DynamicOptions {
  Coordinate N3 = 1; // BLOCKS_PER_GPU
  Coordinate M2 = 8; // WARPS_PER_BLOCK
  Coordinate N0 = 32;// THREADS_PER_WARP
  Coordinate N1 = 4; // Buffer N
  Coordinate M0 = 4; // Buffer M, and UNROLLING_FACTOR
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
Value* y_;

 
__CUDA_DEVICE__
void Loop1Loop2FusedTiled() {
                             
  Coordinate n0 = GetThread();
  Coordinate m2 = GetWarp();
  Coordinate n3 = GetBlock();

  Coordinate M0 = opt::device_->M0;
  Coordinate M2 = opt::device_->M2;
  Coordinate N0 = opt::device_->N0;
  Coordinate N1 = opt::device_->N1;
  Coordinate N3 = opt::device_->N3;

  Coordinate M1(M__/(M2*M0));
  Coordinate N2(N__/(N3*N1*N0));
  Coordinate K(K__);

  Trace(2, "Begin Run: %d, %d, %d, %d, %d, %d, %d, %d", N3, N2, N1, N0, M2, M1, M0, K);



  Value* buffer = static_cast<Value*>(std::malloc(N1 * M0 * sizeof(Value)));
  
  // Initialize buffer
  for (Coordinate n1 = 0; n1 < N1; n1++) {
    #pragma unroll
    for (Coordinate m0 = 0; m0 < M0; m0++) {
      buffer[n1 * M0 + m0] = 0;
    }
  }

  for (Coordinate n2 = 0; n2 < N2; n2++) {
    for (Coordinate m1 = 0; m1 < M1; m1++) {
      // First Loop: High compute intensity
      for (Coordinate k = 0; k < K; k++) {
        for (Coordinate n1 = 0; n1 < N1; n1++) {
          Coordinate n = n3 * N2 * N1 * N0 + n2 * N1 * N0 + n1 * N0 + n0;
          #pragma unroll
          for (Coordinate m0 = 0; m0 < M0; m0++) {
            Coordinate m = m2 * M1 * M0 + m1 * M0 + m0;
            Trace(4, "Loop 1: Iteration %d, %d, %d: %d += %d * %d", m, n, k, buffer[n1 * M0 + m0], a_[m * K__ + k], b_[k * N__ + n]);
            buffer[n1 * M0 + m0] += a_[m * K__ + k] * b_[k * N__ + n];
          }
          Trace(3, "Loop 1: Finish M0");
        }
        Trace(3, "Loop 1: Finish N1");
      }
      Trace(3, "Loop 1: Finish K");
      // Second Loop: Low compute intensity
      for (Coordinate n1 = 0; n1 < N1; n1++) {
        Coordinate n = n3 * N2 * N1 * N0 + n2 * N1 * N0 + n1 * N0 + n0;
        #pragma unroll
        for (Coordinate m0 = 0; m0 < M0; m0++) {
          Coordinate m = m2 * M1 * M0 + m1 * M0 + m0;
          y_[m * N__ + n] = buffer[n1 * M0 + m0] * 17;
          buffer[n1 * M0 + m0] = 0;
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
  

  Coordinate M2 = opt::device_->M2;
  Coordinate N0 = opt::device_->N0;

  for (Coordinate m2 = 0; m2 < M2; m2++) {
    for (Coordinate n0 = 0; n0 < N0; n0++) {
      Trace(2, "Build: %d, %d", m2, n0);
      queueda::Bind(GetBlock(), m2, n0, Loop1Loop2FusedTiled);
    }
  }
}

__CUDA_GLOBAL__
void 
FTBuildKernel(
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

  M__ = M;
  K__ = K;
  N__ = N;
  a_ = a;
  b_ = b;
  y_ = y;

  queueda::Run();
}


inline
int
RunFT(Coordinate M, Coordinate K, Coordinate N) {

  Coordinate N0 = opt::host_->N0;
  Coordinate M2 = opt::host_->M2;
  Coordinate N3 = opt::host_->N3;

  queueda::Init(N3, M2, N0);

  SimpleTensor* af = new SimpleTensor({M, K}, "A", {0, 255});
  SimpleTensor* bf = new SimpleTensor({K, N}, "B", {0, 255});
  SimpleTensor* yf = new SimpleTensor({M, N}, "Y");
    
  auto ad = af->CopyArrayToDevice();
  auto bd = bf->CopyArrayToDevice();
  auto yd = yf->CopyArrayToDevice();

#ifdef __CUDACC__

  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 100000000*sizeof(Value));
  gpuErrchk(cudaPeekAtLastError());

  FTBuildKernel<<<N3, 1>>>(M, K, N, ad, bd, yd);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  FTKernel<<<N3, M2*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, yd);
  gpuErrchk(cudaPeekAtLastError());
  FTKernel<<<N3, M2*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, yd);
  gpuErrchk(cudaPeekAtLastError());
  FTKernel<<<N3, M2*options::kMaxThreadsPerWarp>>>(M, K, N, ad, bd, yd);
  gpuErrchk(cudaPeekAtLastError());

  cudaDeviceSynchronize();

#else

  
  FTBuildKernel(M, K, N, ad, bd, yd);
  FTKernel(M, K, N, ad, bd, yd);

#endif

  yf->CopyArrayFromDevice(yd);
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
  
  opt::host_ = new opt::DynamicOptions();
  
  if (argc > 1)
  {
    opt::host_->N3 = std::atoi(argv[1]);
  }
  if (argc > 2)
  {
    opt::host_->M2 = std::atoi(argv[2]);
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
  
  if (argc > 7)
  {
    opt::host_->M0 = std::atoi(argv[7]);
  }
  
  if (argc > 8)
  {
    opt::host_->N1 = std::atoi(argv[8]);
  }
  
  SET_DEVICE_OPTIONS(opt::DynamicOptions, opt::device_, opt::host_);

  Coordinate M = opt::host_->M2 * NUM_PASSES_M * opt::host_->M0;
  Coordinate N = opt::host_->N3 * NUM_PASSES_N * opt::host_->N1 * opt::host_->N0;
  
  Coordinate a_size = M * K * sizeof(Value);
  Coordinate b_size = K * N * sizeof(Value);
  Coordinate z_size = M * N * sizeof(Value);
  Coordinate num_muls = M * K * N;

  printf("M2: %'d, M1: %'d, M0: %'d, N3: %'d, N2: %'d, N1: %'d, N0: %'d\n", opt::host_->M2, NUM_PASSES_M, opt::host_->M0, opt::host_->N3, NUM_PASSES_N, opt::host_->N1, opt::host_->N0);
  printf("M: %'d, K: %'d, N: %'d\n", M, K, N);
  printf("Size of A in bytes: %'d\n", a_size);
  printf("Size of B in bytes: %'d\n", b_size);
  printf("Size of Z in bytes: %'d\n", z_size);
  printf("Total GPU memory footprint in bytes: %'d\n", a_size + b_size + z_size);
  printf("Total Muls: %'d\n", num_muls);

  return RunFT(M, K, N);
}

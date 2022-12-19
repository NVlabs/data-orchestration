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
  Coordinate M1 = 1;  // BLOCKS_PER_GPU
  Coordinate N2 = 8;  // WARPS_PER_BLOCK
  Coordinate N0 = 32; // THREADS_PER_WARP
};

DynamicOptions* host_;
__CUDA_DEVICE__ DynamicOptions* device_;

}

// Globals
__CUDA_DEVICE__
Coordinate M__;
__CUDA_DEVICE__
Coordinate N__;

__CUDA_DEVICE__
Value* z_;
__CUDA_DEVICE__
Value* y_;


__CUDA_DEVICE__
void Elementwise() {

  Trace(2, "Begin Loop2.");
  Coordinate n0 = GetThread();
  Coordinate m1 = GetBlock();
  Coordinate n2 = GetWarp();

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


__CUDA_DEVICE__
void
BaselineTest(
  Coordinate M,
  Coordinate N,
  Value* z,
  Value* y
  ) {
  

  Coordinate N2 = opt::device_->N2;
  Coordinate N0 = opt::device_->N0;

  for (Coordinate n2 = 0; n2 < N2; n2++) {
    for (Coordinate n0 = 0; n0 < N0; n0++) {
      queueda::Bind(GetBlock(), n2, n0, Elementwise);
    }
  }
}


__CUDA_GLOBAL__
void 
BTBuildKernel(
  Coordinate M,
  Coordinate N,
  Value* z,
  Value* y) {
   
  BuilderFunction build_func = 
    [M, N, z, y] () {
      BaselineTest(M, N, z, y);
    };

  queueda::Build(build_func);
}


__CUDA_GLOBAL__
void 
BTKernel(
  Coordinate M,
  Coordinate N,
  Value* z,
  Value* y) {
   
  M__ = M;
  N__ = N;
  z_ = z;
  y_ = y;

  queueda::Run();
}

inline
int
RunBT(Coordinate M, Coordinate N) {

  Coordinate M1 = opt::host_->M1;
  Coordinate N2 = opt::host_->N2;
  Coordinate N0 = opt::host_->N0;

  queueda::Init(M1, N2, N0);

  SimpleTensor* zf = new SimpleTensor({M, N}, "Z", {0, 255});
  SimpleTensor* yf = new SimpleTensor({M, N}, "Y");
  
  auto zd = zf->CopyArrayToDevice();
  auto yd = yf->CopyArrayToDevice();
  
  //queueda::Build(build_func)
  //queueda::Run(B, W, T, run_func)
  
#ifdef __CUDACC__


  BTBuildKernel<<<M1, 1>>>(M, N, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  BTKernel<<<M1, N2*options::kMaxThreadsPerWarp>>>(M, N, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  BTKernel<<<M1, N2*options::kMaxThreadsPerWarp>>>(M, N, zd, yd);
  gpuErrchk(cudaPeekAtLastError());
  BTKernel<<<M1, N2*options::kMaxThreadsPerWarp>>>(M, N, zd, yd);
  gpuErrchk(cudaPeekAtLastError());

  cudaDeviceSynchronize();

#else

  
  BTBuildKernel(M, N,  zd, yd);
  BTKernel(M, N, zd, yd);

#endif

  yf->CopyArrayFromDevice(yd);
  yf->Print();
  
  SimpleTensor* y_ref = new SimpleTensor({M, N}, "Y_REF");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      y_ref->v_[m * N + n] = zf->v_[m * N + n] * 17;
    }   
  }
  
  return yf->CheckMismatches(y_ref);

}



int 
main(int argc, char** argv) {
  
  opt::host_ = new opt::DynamicOptions();
  
  if (argc > 1)
  {
    opt::host_->M1 = std::atoi(argv[1]);
  }
  if (argc > 2)
  {
    opt::host_->N2 = std::atoi(argv[2]);
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
  
  SET_DEVICE_OPTIONS(opt::DynamicOptions, opt::device_, opt::host_);

  Coordinate M = opt::host_->M1 * NUM_PASSES_M;
  Coordinate N = opt::host_->N2 * NUM_PASSES_N * opt::host_->N0;
  
  Coordinate z_size = M * N * sizeof(Value);
  Coordinate num_muls = M * N;

  printf("M1: %'d, M0: %'d, N2: %'d, N1: %'d, N0: %'d\n", opt::host_->M1, NUM_PASSES_M, opt::host_->N2, NUM_PASSES_N, opt::host_->N0);
  printf("M: %'d, N: %'d\n", M, N);
  printf("Size of Z in bytes: %'d\n", z_size);
  printf("Total GPU memory footprint in bytes: %'d\n", z_size + z_size);
  printf("Total Muls: %'d\n", num_muls);

  return RunBT(M, N);
}

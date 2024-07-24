/*
This file is part of the ECL Suite version 1.0.

Race-free ECL-SCC: This code computes the Strongly Connected Components of a directed graph.

Copyright (c) 2023-2024, Martin Burtscher and Yiqian Liu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The original version of this code is available at https://cs.txstate.edu/~burtscher/research/ECL-SCC/ and at https://github.com/burtscher/ECL-SCC.

Original publication: This work is described in detail in the following paper.
Ghadeer Alabandi, William Sands, George Biros, and Martin Burtscher. "A GPU Algorithm for Detecting Strongly Connected Components." Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2023.

URL: The latest version of this race-free version is available at https://github.com/burtscher/ECL-Suite.

Publication: The ECL Suite is described in detail in the following paper.
Yiqian Liu, Avery VanAusdal, and Martin Burtscher. Performance Impact of Removing Data Races from GPU Programs. Proceedings of the IEEE International Symposium on Workload Characterization. September 2024.
*/


#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <sys/time.h>
#include <cuda.h>
#include "ECLgraph.h"
#include "ECLatomic.h"


static const int Device = 0;
static const int ThreadsPerBlock = 512;

using ull = unsigned long long;
using ibool = int;  // atomics not supported on bools


static inline __device__ void writeLongLong(ull* const addr, const unsigned int first, const unsigned int second)  // non-atomic
{
  ull val = second;
  val = (val << 32) | first;
  *addr = val;  // non-atomic okay for all uses in this code
}


static inline __device__ void writeFirstA(ull* const addr, const int first)  // atomic
{
  int* const iaddr = (int*)addr;
  atomicWrite(&iaddr[0], first);
}


static inline __device__ void writeSecondA(ull* const addr, const int second)  // atomic
{
  int* const iaddr = (int*)addr;
  atomicWrite(&iaddr[1], second);
}


static inline __host__ __device__ int readFirst(const ull value)
{
  return static_cast<int>(value & 0xFFFFFFFF);
}


static inline __device__ int readSecond(const ull value)
{
  return static_cast<int>(value >> 32);
}


static inline __device__ int readFirstA(ull* addr)
{
  int* iaddr = (int*)addr;
  return atomicRead(&iaddr[0]);
}


static inline __device__ int readSecondA(ull* addr)
{
  int* iaddr = (int*)addr;
  return atomicRead(&iaddr[1]);
}


static __global__
void globalInit(const ECLgraph g, ull* const __restrict__ wl1, int* const __restrict__ wl1size, ull* const __restrict__ iomax)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;
  for (int v = thread; v < g.nodes; v += threads) {
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    int y = v;
    for (int j = beg; j < end; j++) {
      const int w = g.nlist[j];
      writeLongLong(&wl1[j], v, w);
      y = max(y, w);
    }
    writeLongLong(&iomax[v], v, y);
  }
  if (thread == 0) *wl1size = g.edges;
}


static __global__
void propagateMax(const ull* const __restrict__ wl1, const int wl1size, ull* const __restrict__ iomax, ibool* const __restrict__ goagain)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;
  ibool updated, again = false;
  do {  // iterate locally
    updated = false;
    for (int i = thread; i < wl1size; i += threads) {
      const ull el = wl1[i];
      const int v = readFirst(el);
      const int w = readSecond(el);

      const ull iov = atomicRead(&iomax[v]);
      const ull iow = atomicRead(&iomax[w]);
      const int iov1 = readFirst(iov);
      const int iov2 = readSecond(iov);
      const int iow1 = readFirst(iow);
      const int iow2 = readSecond(iow);
      int im = iov1;
      int om = iow2;
      if (im > v) im = readFirstA(&iomax[im]);  // 'path compress'
      if (om > w) om = readSecondA(&iomax[om]);  // 'path compress'

      // propagate
      if (iov1 < im) {writeFirstA(&iomax[v], im); updated = true;}
      if (iov2 < om) {writeSecondA(&iomax[v], om); updated = true;}
      if (iow1 < im) {writeFirstA(&iomax[w], im); updated = true;}
      if (iow2 < om) {writeSecondA(&iomax[w], om); updated = true;}

      // update other vertices on path
      if ((iov1 < om) && (readSecondA(&iomax[iov1]) < om)) {writeSecondA(&iomax[iov1], om); updated = true;}
      if ((iov1 != iow1) && (iow1 < om) && (readSecondA(&iomax[iow1]) < om)) {writeSecondA(&iomax[iow1], om); updated = true;}
      if ((iov2 < im) && (readFirstA(&iomax[iov2]) < im)) {writeFirstA(&iomax[iov2], im); updated = true;}
      if ((iov2 != iow2) && (iow2 < im) && (readFirstA(&iomax[iow2]) < im)) {writeFirstA(&iomax[iow2], im); updated = true;}
    }
    again |= updated;
  } while (__syncthreads_or(updated));
  again = __syncthreads_or(again);
  if ((threadIdx.x == 0) && again) atomicWrite(goagain, (ibool)true);
}


static __global__
void removeEdges(const ull* const __restrict__ wl1, const int wl1size, ull* const __restrict__ wl2, int* const __restrict__ wl2size, const ull* const __restrict__ iomax)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;
  for (int i = thread; i < wl1size; i += threads) {
    const ull el = wl1[i];

    const int v = readFirst(el);
    const int w = readSecond(el);
    const ull iov = iomax[v];
    const ull iow = iomax[w];

    if (readFirst(iov) != readSecond(iov)) {
      if ((readFirst(iow) == readFirst(iov)) && (readSecond(iow) == readSecond(iov))) {
        const int k = atomicAdd(wl2size, 1);
        wl2[k] = el;
      }
    }
  }
}


static __global__
void localInit(const int nodes, ull* const __restrict__ iomax, ibool* const __restrict__ goagain)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;
  ibool again = false;
  for (int v = thread; v < nodes; v += threads) {
    const ull iov = iomax[v];

    if (readFirst(iov) != readSecond(iov)) {
      writeLongLong(&iomax[v], v, v);
      again = true;
    }
  }
  again = __syncthreads_or(again);
  if ((threadIdx.x == 0) && again) atomicWrite(goagain, (ibool)true);
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  float stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001f * ms;}
};


int main(int argc, char* argv [])
{
  printf("ECL-SCC v1.0\n\n");  fflush(stdout);

  // process command line
  if (argc != 2) {printf("USAGE: %s input_file_name\n\n", argv[0]);  exit(-1);}

  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n\n", 1.0 * g.edges / g.nodes);

  // get GPU info
  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);  fflush(stdout);
  const int blocks = SMs * (mTpSM / ThreadsPerBlock);

  // allocate GPU memory
  ull *d_wl1, *d_wl2, *d_iomax;
  int *d_wl2size;
  ibool *d_goagain;
  cudaMalloc((void **)&d_iomax, g.nodes * sizeof(ull));
  cudaMalloc((void **)&d_wl1, g.edges * sizeof(ull));
  cudaMalloc((void **)&d_wl2, g.edges * sizeof(ull));
  cudaMalloc((void **)&d_wl2size, sizeof(int));
  cudaMalloc((void **)&d_goagain, sizeof(ibool));

  // copy graph to GPU
  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, (g.nodes + 1) * sizeof(int));
  cudaMemcpy(d_g.nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_g.nlist, g.edges * sizeof(int));
  cudaMemcpy(d_g.nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

/*************************************************************************/

  // start time
  printf("running algorithm\n");  fflush(stdout);
  GPUTimer timer;
  timer.start();

  // global initialization
  int wl1size;
  globalInit<<<blocks, ThreadsPerBlock>>>(d_g, d_wl1, d_wl2size, d_iomax);
  cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);

  ibool goagain;
  do {
    // propagate max values
    do {
      cudaMemsetAsync(d_goagain, 0, sizeof(ibool));
      propagateMax<<<blocks, ThreadsPerBlock>>>(d_wl1, wl1size, d_iomax, d_goagain);
      cudaMemcpy(&goagain, d_goagain, sizeof(ibool), cudaMemcpyDeviceToHost);
    } while (goagain);

    // remove edges that cannot be part of an SCC
    cudaMemsetAsync(d_wl2size, 0, sizeof(int));
    removeEdges<<<blocks, ThreadsPerBlock>>>(d_wl1, wl1size, d_wl2, d_wl2size, d_iomax);
    std::swap(d_wl1, d_wl2);
    cudaMemcpyAsync(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);

    // local re-initialization
    cudaMemsetAsync(d_goagain, 0, sizeof(ibool));
    localInit<<<blocks, ThreadsPerBlock>>>(g.nodes, d_iomax, d_goagain);
    cudaMemcpy(&goagain, d_goagain, sizeof(ibool), cudaMemcpyDeviceToHost);
  } while (goagain);

  // end time
  const double runtime = timer.stop();
  printf("compute time: %.9f s\n\n", runtime);
  CheckCuda(__LINE__);

/*************************************************************************/

  // copy result to host
  const int n = g.nodes;
  ull* iomax = new ull [n];
  cudaMemcpy(iomax, d_iomax, n * sizeof(ull), cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);

  // output SCC sizes and frequency
  printf("result:\n");
  int* count = new int [n];
  for (int v = 0; v < n; v++) count[v] = 0;
  for (int v = 0; v < n; v++) count[readFirst(iomax[v])]++;
  std::sort(count, count + n, std::greater<int>());
  int cnt = 1;
  int SCC_cnt = 0;
  for (int v = 1; v < n; v++) {
    if (count[v] != count[v - 1]) {
      printf("%d SCCs of size %d\n", cnt, count[v - 1]);
      SCC_cnt += cnt;
      cnt = 1;
    } else {
      cnt++;
    }
  }
  if (count[n - 1] > 0) {
    printf("%d SCCs of size %d\n", cnt, count[n - 1]);
    SCC_cnt += cnt;
  }
  printf("number of SCCs: %d\n", SCC_cnt);

  // clean up
  cudaFree(d_g.nindex);  cudaFree(d_g.nlist);  cudaFree(d_iomax);  cudaFree(d_wl1);  cudaFree(d_wl2);  cudaFree(d_wl2size);  cudaFree(d_goagain);
  delete [] iomax;
  delete [] count;
  freeECLgraph(g);
  return 0;
}

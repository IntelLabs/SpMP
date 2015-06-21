/**
Copyright (c) 2015, Intel Corporation. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>

#include "CSR.hpp"

using namespace std;

namespace SpMP
{

template<class T, int BASE = 0>
static void SpMV_(
  int m,
  T *w,
  T alpha,
  const int *rowptr, const int *colidx, const T* values,
  const T *x,
  T beta,
  const T *y,
  T gamma)
{
//#define MEASURE_LOAD_BALANCE
#ifdef MEASURE_LOAD_BALANCE
  double barrierTimes[omp_get_max_threads()];
  double tBegin = omp_get_wtime();
#endif

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int nnz = rowptr[m] - BASE;
    int nnzPerThread = (nnz + nthreads - 1)/nthreads;
    int iBegin = lower_bound(rowptr, rowptr + m, nnzPerThread*tid + BASE) - rowptr;
    int iEnd = lower_bound(rowptr, rowptr + m, nnzPerThread*(tid + 1) + BASE) - rowptr;
    assert(iBegin <= iEnd);
    assert(iBegin >= 0 && iBegin <= m);
    assert(iEnd >= 0 && iEnd <= m);

    if (1 == alpha && 0 == beta && 0 == gamma) {
      for (int i = iBegin; i < iEnd; ++i) {
        T sum = 0;
        for (int j = rowptr[i] - BASE; j < rowptr[i + 1] - BASE; ++j) {
          sum += values[j]*x[colidx[j] - BASE];
        }
        w[i] = sum;
      }
    }
    else {
      for (int i = iBegin; i < iEnd; ++i) {
        T sum = 0;
        for (int j = rowptr[i] - BASE; j < rowptr[i + 1] - BASE; ++j) {
          sum += values[j]*x[colidx[j] - BASE];
        }
        w[i] = alpha*sum + beta*y[i] + gamma;
      }
    }
#ifdef MEASURE_LOAD_BALANCE
    double t = omp_get_wtime();
#pragma omp barrier
    barrierTimes[tid] = omp_get_wtime() - t;

#pragma omp barrier
#pragma omp master
    {
      double tEnd = omp_get_wtime();
      double barrierTimeSum = 0;
      for (int i = 0; i < nthreads; ++i) {
        barrierTimeSum += barrierTimes[i];
      }
      printf("%f load imbalance = %f\n", tEnd - tBegin, barrierTimeSum/(tEnd - tBegin)/nthreads);
    }
#undef MEASURE_LOAD_BALANCE
#endif // MEASURE_LOAD_BALANCE
  } // omp parallel
}

void CSR::multiplyWithVector(
  double *w,
  double alpha, const double *x, double beta, const double *y, double gamma)
  const
{
  if (0 == base) {
    SpMV_<double, 0>(m, w, alpha, rowptr, colidx, values, x, beta, y, gamma);
  }
  else {
    assert(1 == base);
    SpMV_<double, 1>(m, w, alpha, rowptr, colidx, values, x, beta, y, gamma);
  }
}

void CSR::multiplyWithVector(double *w, const double *x) const
{
  return multiplyWithVector(w, 1, x, 0, w, 0);
}

} // namespace SpMP

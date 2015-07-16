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

#include <cstring>
#include <algorithm>

#include "CSR.hpp"

using namespace std;

namespace SpMP
{

void CSR::permuteRowptr(CSR *ret, const int *reversePerm) const
{
  ret->rowptr[0] = 0;

  int rowPtrSum[omp_get_max_threads() + 1];
  rowPtrSum[0] = 0;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int iPerThread = (m + nthreads - 1)/nthreads;
    int iBegin = min(iPerThread*tid, m);
    int iEnd = min(iBegin + iPerThread, m);

    ret->rowptr[iBegin] = 0;
    int i;
    for (i = iBegin; i < iEnd - 1; ++i) {
      int row = reversePerm ? reversePerm[i] : i;
      int begin = rowptr[row], end = rowptr[row + 1];
      ret->rowptr[i + 1] = ret->rowptr[i] + end - begin;
      if (extptr) {
        ret->extptr[i] = ret->rowptr[i] + extptr[row] - rowptr[row];
      }
    }
    if (i < iEnd) {
      int row = reversePerm ? reversePerm[i] : i;
      int begin = rowptr[row], end = rowptr[row + 1];
      rowPtrSum[tid + 1] = ret->rowptr[i] + end - begin;
      if (extptr) {
        ret->extptr[i] = ret->rowptr[i] + extptr[row] - rowptr[row];
      }
    }
    else {
      rowPtrSum[tid + 1] = 0;
    }

#pragma omp barrier
#pragma omp master
    {
      for (int tid = 1; tid < nthreads; ++tid) {
        rowPtrSum[tid + 1] += rowPtrSum[tid];
      }
      ret->rowptr[m] = rowPtrSum[nthreads];
    }
#pragma omp barrier

    for (i = iBegin; i < iEnd; ++i) {
      ret->rowptr[i] += rowPtrSum[tid];
      if (ret->extptr) ret->extptr[i] += rowPtrSum[tid];
    }
  } // omp parallel
}

CSR *CSR::permuteRowptr(const int *reversePerm) const
{
  CSR *ret = new CSR();

  ret->m = m;
  ret->n = n;
  ret->rowptr = MALLOC(int, m + 1);
  assert(ret->rowptr);
  int nnz = rowptr[m];
  ret->colidx = MALLOC(int, nnz);
  assert(ret->colidx);
  ret->values = MALLOC(double, nnz);
  assert(ret->values);
  ret->idiag = MALLOC(double, m);
  assert(ret->idiag);
  if (diagptr) {
    ret->diagptr = MALLOC(int, m);
    assert(ret->diagptr);
  }
  if (extptr) {
    ret->extptr = MALLOC(int, m);
    assert(ret->extptr);
  }
  
  permuteRowptr(ret, reversePerm);

  return ret;
}

template<class T, int BASE = 0>
void permuteColsInPlace_(CSR *A, const int *perm)
{
  assert(perm);

  const int *rowptr = A->rowptr;
  const int *extptr = A->extptr ? A->extptr : rowptr + 1;
  int *diagptr = A->diagptr;

  int *colidx = A->colidx;
  T *values = A->values;

  int m = A->m;

#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    int diagCol = -1;
    for (int j = rowptr[i] - BASE; j < extptr[i] - BASE; ++j) {
      int c = colidx[j] - BASE;
      assert(c >= 0 && c < m);
      colidx[j] = perm[c] + BASE;
      assert(colidx[j] - BASE >= 0 && colidx[j] - BASE < m);
      if (c == i) diagCol = perm[c] + BASE;
    }

    for (int j = rowptr[i] + 1 - BASE; j < extptr[i] - BASE; ++j) {
      int c = colidx[j];
      double v = values[j];

      int k = j - 1;
      while (k >= rowptr[i] - BASE & colidx[k] > c) {
        colidx[k + 1] = colidx[k];
        values[k + 1] = values[k];
        --k;
      }

      colidx[k + 1] = c;
      values[k + 1] = v;
    }

    if (diagptr) {
      for (int j = rowptr[i] - BASE; j < extptr[i] - BASE; ++j) {
        if (colidx[j] == diagCol) {
          diagptr[i] = j + BASE;
          break;
        }
      }
    }
  } // for each row
}

void CSR::permuteColsInPlace(const int *perm)
{
  if (0 == base) {
    permuteColsInPlace_<double, 0>(this, perm);
  }
  else {
    assert(1 == base);
    permuteColsInPlace_<double, 1>(this, perm);
  }
}

template<class T, int BASE = 0, bool SORT = false>
static void permuteMain_(
  CSR *out, const CSR *in,
  const int *columnPerm, const int *rowInversePerm)
{
  const int *rowptr = in->rowptr;
  const int *colidx = in->colidx;
  const int *diagptr = in->diagptr;
  const T *values = in->values;
  const T *idiag = in->idiag;

  int m = in->m;
  int nnz = rowptr[m] - BASE;

  const int *extptr = in->extptr ? in->extptr : rowptr + 1;
  const int *newExtptr = out->extptr ? out->extptr : out->rowptr + 1;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int nnzPerThread = (nnz + nthreads - 1)/nthreads;
    int iBegin = lower_bound(out->rowptr, out->rowptr + m, nnzPerThread*tid + BASE) - out->rowptr;
    if (tid == 0) iBegin = 0;
    int iEnd = lower_bound(out->rowptr, out->rowptr + m, nnzPerThread*(tid + 1) + BASE) - out->rowptr;
    if (tid == nthreads - 1) iEnd = m;
    assert(iBegin <= iEnd);
    assert(iBegin >= 0 && iBegin <= m);
    assert(iEnd >= 0 && iEnd <= m);

    for (int i = iBegin; i < iEnd; ++i) {
      int row = rowInversePerm ? rowInversePerm[i] : i;
      int begin = rowptr[row] - BASE, end = extptr[row] - BASE;
      int newBegin = out->rowptr[i] - BASE;

      int diagCol = -1;
      int k = newBegin;
      for (int j = begin; j < end; ++j, ++k) {
        int colIdx = colidx[j] - BASE;
        int newColIdx = columnPerm[colIdx];

        out->colidx[k] = newColIdx + BASE;
        out->values[k] = values[j];

        if (diagptr && colidx[j] == row) {
          diagCol = newColIdx;
        }
      }
      assert(!diagptr || diagCol != -1);

      if (SORT) {
        // insertion sort
        for (int j = newBegin + 1; j < newExtptr[i]; ++j) {
          int c = out->colidx[j];
          double v = out->values[j];

          int k = j - 1;
          while (k >= newBegin && out->colidx[k] > c) {
            out->colidx[k + 1] = out->colidx[k];
            out->values[k + 1] = out->values[k];
            --k;
          }

          out->colidx[k + 1] = c;
          out->values[k + 1] = v;
        }
      }

      if (idiag) out->idiag[i] = idiag[row];

      if (diagptr) {
        for (int j = newBegin; j < newExtptr[i]; ++j) {
          if (out->colidx[j] == diagCol) {
            out->diagptr[i] = j;
            break;
          }
        }
      } // if (diagptr)
    } // for each row
  } // omp parallel
}

void CSR::permuteMain(
  CSR *out, const int *columnPerm, const int *rowInversePerm,
  bool sort /*=false*/) const
{
  if (0 == base) {
    if (sort) {
      permuteMain_<double, 0, true>(out, this, columnPerm, rowInversePerm);
    }
    else {
      permuteMain_<double, 0, false>(out, this, columnPerm, rowInversePerm);
    }
  }
  else {
    assert(1 == base);
    if (sort) {
      permuteMain_<double, 1, true>(out, this, columnPerm, rowInversePerm);
    }
    else {
      permuteMain_<double, 1, false>(out, this, columnPerm, rowInversePerm);
    }
  }
}

template<class T, int BASE = 0>
static void permuteRowsMain_(
  CSR *out, const CSR *in, const int *rowInversePerm)
{
  const int *rowptr = in->rowptr;
  const int *colidx = in->colidx;
  const int *diagptr = in->diagptr;
  const T *values = in->values;
  const T *idiag = in->idiag;

  int m = in->m;
  int nnz = rowptr[m] - BASE;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int nnzPerThread = (nnz + nthreads - 1)/nthreads;
    int iBegin = lower_bound(out->rowptr, out->rowptr + m, nnzPerThread*tid + BASE) - out->rowptr;
    if (tid == 0) iBegin = 0;
    int iEnd = lower_bound(out->rowptr, out->rowptr + m, nnzPerThread*(tid + 1) + BASE) - out->rowptr;
    if (tid == nthreads - 1) iEnd = m;
    assert(iBegin <= iEnd);
    assert(iBegin >= 0 && iBegin <= m);
    assert(iEnd >= 0 && iEnd <= m);

    for (int i = iBegin; i < iEnd; ++i) {
      int row = rowInversePerm ? rowInversePerm[i] : i;
      int begin = rowptr[row] - BASE, end = rowptr[row + 1] - BASE;
      int newBegin = out->rowptr[i] - BASE;

      memcpy(out->values + newBegin, values + begin, (end - begin)*sizeof(double));
      memcpy(out->colidx + newBegin, colidx + begin, (end - begin)*sizeof(int));

      if (diagptr)
        out->diagptr[i] =
          out->rowptr[i] + (diagptr[row] - rowptr[row]);
      if (idiag) out->idiag[i] = idiag[row];
    }
  }
}

void CSR::permuteRowsMain(CSR *out, const int *rowInversePerm) const
{
  if (0 == base) {
    permuteRowsMain_<double, 0>(out, this, rowInversePerm);
  }
  else {
    assert(1 == base);
    permuteRowsMain_<double, 1>(out, this, rowInversePerm);
  }
}

CSR *CSR::permute(const int *columnPerm, const int *rowInversePerm, bool sort /*=false*/) const
{
  CSR *ret = permuteRowptr(rowInversePerm);
  permuteMain(ret, columnPerm, rowInversePerm, sort);
  return ret;
}

CSR *CSR::permuteRows(const int *reversePerm) const
{
  CSR *ret = permuteRowptr(reversePerm);
  permuteRowsMain(ret, reversePerm);
  return ret;
}

} // namespace SpMP

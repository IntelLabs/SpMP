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

#include <cstdio>
#include <cstring>
#include <climits>

#include <omp.h>

#include "CSR.hpp"
#include "COO.hpp"
#include "mm_io.h"
#include "Utils.hpp"
#include "MemoryPool.hpp"

#ifdef LOADIMBA
#include "synk/loadimba.hpp"
#else
#include "synk/barrier.hpp"
#endif

using namespace std;

#ifdef LOADIMBA
extern synk::LoadImba *bar;
#else
extern synk::Barrier *bar;
#endif

namespace SpMP
{

bool CSR::useMemoryPool_() const
{
  return MemoryPool::getSingleton()->contains(rowptr);
}

CSR::CSR() : rowptr(NULL), colidx(NULL), values(NULL), idiag(NULL), diag(NULL), diagptr(NULL), extptr(NULL), base(0), ownData_(false)
{
}

void CSR::alloc(int m, int nnz, bool createSeparateDiagData /*= true*/)
{
  this->m = m;

  rowptr = MALLOC(int, m + 1);
  colidx = MALLOC(int, nnz);
  values = MALLOC(double, nnz);
  diagptr = MALLOC(int, m);

  assert(rowptr != NULL);
  assert(colidx != NULL);
  assert(values != NULL);
  assert(diagptr != NULL);

  if (createSeparateDiagData) {
    idiag = MALLOC(double, m);
    diag = MALLOC(double, m);
    assert(idiag != NULL);
    assert(diag != NULL);
  }

  ownData_ = true;
}

CSR::CSR(int m, int n, int nnz, int base /*=0*/)
 : base(base)
{
  this->m=m;
  this->n=n;
  alloc(n, nnz);
}

CSR::CSR(const char *file, int pad)
 : base(0), rowptr(NULL), colidx(NULL), values(NULL), idiag(NULL), diag(NULL), diagptr(NULL), extptr(NULL)
{
  int m = atoi(file);
  char buf[1024];
  sprintf(buf, "%d", m);

  int l = strlen(file);

  if (!strcmp(buf, file)) {
    generate3D27PtLaplacian(this, m);
  }
  else {
    COO Acoo;
    load_matrix_market((char *)file, Acoo, pad);

    alloc(Acoo.m, Acoo.nnz);

    dcoo2crs(&Acoo, this);
  }
}

CSR::CSR(int m, int n, int *rowptr, int *colidx, double *values, int base /*=0*/) :
 m(m), n(n), rowptr(rowptr), colidx(colidx), values(values), ownData_(false), idiag(NULL), diag(NULL), extptr(NULL)
{
  diagptr = MALLOC(int, m);
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i] - base; j < rowptr[i + 1] - base; ++j) {
      if (colidx[j] - base == i) {
        diagptr[i] = j + base;
      }
    }
  }
}

void CSR::dealloc()
{
  if (useMemoryPool_()) {
    // a large single contiguous chunk is allocated to
    // buffers except rowptr and colidx.
    rowptr = NULL;
    extptr = NULL;
    colidx = NULL;
    values = NULL;
    idiag = NULL;
    diag = NULL;
    diagptr = NULL;
  }
  else {
    if (ownData_) {
      FREE(rowptr);
      FREE(extptr);
      FREE(colidx);
      FREE(values);
    }

    FREE(idiag);
    FREE(diag);
    FREE(diagptr);
  }
}

CSR::~CSR()
{
  dealloc();
}

bool CSR::isSymmetric(bool checkValues /*=true*/, bool printFirstNonSymmetry /* = false*/) const
{
  const int *extptr = this->extptr ? this->extptr : rowptr + 1;
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i] - base; j < extptr[i] - base; ++j) {
      int c = colidx[j] - base;
      if (c > i) {
        bool hasPair = false;
        for (int k = rowptr[c] - base; k < extptr[c] - base; ++k) {
          if (colidx[k] - base == i) {
            hasPair = true;
            if (checkValues && values[j] != values[k]) {
              if (printFirstNonSymmetry) {
                printf(
                  "assymmetric (%d, %d) = %g, (%d, %d) = %g\n", 
                  i + 1, c + 1, values[j], c + 1, i + 1, values[k]);
              }
              return false;
            }
            break;
          }
        }
        if (!hasPair) {
          if (printFirstNonSymmetry) {
            printf(
              "assymmetric (%d, %d) exists but (%d, %d) doesn't\n",
              i + 1, c + 1, c + 1, i + 1);
          }
          return false;
        }
      }
    }
  } // for each row

  return true;
}

void CSR::store_matrix_market(const char *file_name) const
{
  FILE *fp = fopen(file_name, "w");
  assert(fp);

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  // print banner followed by typecode.
  fprintf(fp, "%s ", MatrixMarketBanner);
  fprintf(fp, "%s\n", mm_typecode_to_str(matcode));

  // print matrix size and nonzeros.
  fprintf(fp, "%d %d %d\n", m, n, rowptr[m]);
  
  // print values
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      fprintf(fp, "%d %d %20.16g\n", i + 1, colidx[j] + 1, values[j]);
    }
  }

  fclose(fp);
}

void CSR::make0BasedIndexing()
{
  if (0 == base) return;

#pragma omp parallel for
  for(int i=0; i <= m; i++)
    rowptr[i]--;

  int nnz = rowptr[m];
#pragma omp parallel for
  for(int i=0; i < nnz; i++)
    colidx[i]--;

  base = 0;
}

void CSR::make1BasedIndexing()
{
  if (1 == base) return;

#pragma omp parallel for
  for(int i=0; i <= m; i++)
    rowptr[i]++;

  int nnz = rowptr[m];
#pragma omp parallel for
  for(int i=0; i < nnz; i++)
    colidx[i]++;

  base = 1;
}

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline int transpose_idx(int idx, int dim1, int dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}

// TODO: only supports local matrix for now
/**
 * Transposition using parallel counting sort
 */
CSR *CSR::transpose() const
{
   const double *A_data = values;
   const int *A_i = rowptr;
   const int *A_j = colidx;
   int num_rowsA = m;
   int num_colsA = n;
   int nnz = rowptr[m];
   int num_nonzerosA = nnz;

   CSR *AT = new CSR();

   AT->m = m;
   AT->n = n;
   AT->colidx = MALLOC(int, nnz);
   assert(AT->colidx);
   if (A_data) {
     AT->values = MALLOC(double, nnz);
     assert(AT->values);
   }
   if (diagptr) {
     AT->diagptr = MALLOC(int, m);
     assert(AT->diagptr);
   }

   if (0 == num_colsA) {
     return AT;
   }

   int *AT_j = AT->colidx;
   double *AT_data = AT->values;

   double t = omp_get_wtime();

   int *bucket = MALLOC(
    int, (num_colsA + 1)*omp_get_max_threads());

#ifndef NDEBUG
  int i;
  for (i = 0; i < num_rowsA; ++i) {
    assert(A_i[i + 1] >= A_i[i]);
  }
#endif

#define MIN(a, b) (((a) <= (b)) ? (a) : (b))

#pragma omp parallel
   {
   int nthreads = omp_get_num_threads();
   int tid = omp_get_thread_num();

   int nnzPerThread = (num_nonzerosA + nthreads - 1)/nthreads;
   int iBegin = lower_bound(A_i, A_i + num_rowsA, nnzPerThread*tid) - A_i;
   int iEnd = lower_bound(A_i, A_i + num_rowsA, nnzPerThread*(tid + 1)) - A_i;

   int i, j;
   memset(bucket + tid*num_colsA, 0, sizeof(int)*num_colsA);

   // count the number of keys that will go into each bucket
   for (j = A_i[iBegin]; j < A_i[iEnd]; ++j) {
     int idx = A_j[j];
#ifndef NDEBUG
     if (idx < 0 || idx >= num_colsA) {
       printf("tid = %d num_rowsA = %d num_colsA = %d num_nonzerosA = %d iBegin = %d iEnd = %d A_i[iBegin] = %d A_i[iEnd] = %d j = %d idx = %d\n", tid, num_rowsA, num_colsA, num_nonzerosA, iBegin, iEnd, A_i[iBegin], A_i[iEnd], j, idx);
     }
#endif
     assert(idx >= 0 && idx < num_colsA);
     bucket[tid*num_colsA + idx]++;
   }
   // up to here, bucket is used as int[nthreads][num_colsA] 2D array

   // prefix sum
#pragma omp barrier

   for (i = tid*num_colsA + 1; i < (tid + 1)*num_colsA; ++i) {
     int transpose_i = transpose_idx(i, nthreads, num_colsA);
     int transpose_i_minus_1 = transpose_idx(i - 1, nthreads, num_colsA);

     bucket[transpose_i] += bucket[transpose_i_minus_1];
   }

#pragma omp barrier
#pragma omp master
   {
     for (i = 1; i < nthreads; ++i) {
       int j0 = num_colsA*i - 1, j1 = num_colsA*(i + 1) - 1;
       int transpose_j0 = transpose_idx(j0, nthreads, num_colsA);
       int transpose_j1 = transpose_idx(j1, nthreads, num_colsA);

       bucket[transpose_j1] += bucket[transpose_j0];
     }
     bucket[num_colsA] = num_nonzerosA;
   }
#pragma omp barrier

   if (tid > 0) {
     int transpose_i0 = transpose_idx(num_colsA*tid - 1, nthreads, num_colsA);

     for (i = tid*num_colsA; i < (tid + 1)*num_colsA - 1; ++i) {
       int transpose_i = transpose_idx(i, nthreads, num_colsA);

       bucket[transpose_i] += bucket[transpose_i0];
     }
   }

#pragma omp barrier

   if (A_data) {
      for (i = iEnd - 1; i >= iBegin; --i) {
        for (j = A_i[i + 1] - 1; j >= A_i[i]; --j) {
          int idx = A_j[j];
          --bucket[tid*num_colsA + idx];

          int offset = bucket[tid*num_colsA + idx];

          assert(offset >= 0 && offset < num_nonzerosA);
          AT_data[offset] = A_data[j];
          AT_j[offset] = i;
        }
      }
   }
   else {
      for (i = iEnd - 1; i >= iBegin; --i) {
        for (j = A_i[i + 1] - 1; j >= A_i[i]; --j) {
          int idx = A_j[j];
          --bucket[tid*num_colsA + idx];

          int offset = bucket[tid*num_colsA + idx];

          AT_j[offset] = i;
        }
      }
   }

   if (diagptr) {
#pragma omp barrier
     for (i = iBegin; i < iEnd; ++i) {
       for (int j = bucket[i]; j < bucket[i + 1]; ++j) {
         int c = AT_j[j];
         if (c == i) AT->diagptr[i] = j;
       }
     }
   }

   } // omp parallel

   AT->rowptr = bucket; 

   return AT;
 }

template<int BASE = 0>
int getBandwidth_(const CSR *A)
{
  int bw = INT_MIN;
#pragma omp parallel for reduction(max:bw)
  for (int i = 0; i < A->m; ++i) {
    for (int j = A->rowptr[i] - BASE; j < A->rowptr[i + 1] - BASE; ++j) {
      int c = A->colidx[j] - BASE;
      int temp = c - i;
      if (temp < 0) temp = -temp;
      bw = max(temp, bw);
    }
  }
  return bw;
}

int CSR::getBandwidth() const
{
  if (0 == base) {
    return getBandwidth_<0>(this);
  }
  else {
    assert(1 == base);
    return getBandwidth_<1>(this);
  }
}

} // namespace SpMP

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

#include "COO.hpp"
#include "mm_io.h"

using namespace std;

namespace SpMP
{

COO::COO() : rowidx(NULL), colidx(NULL), values(NULL), isSymmetric(false)
{
}

COO::~COO()
{
  dealloc();
}

void COO::dealloc()
{
  FREE(rowidx);
  FREE(colidx);
  FREE(values);
}

void COO::storeMatrixMarket(const char *fileName) const
{
  FILE *fp = fopen(fileName, "w");
  assert(fp);

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  int err = mm_write_mtx_crd(
    (char *)fileName, m, n, nnz, rowidx, colidx, values, matcode);
  if (err) {
    fprintf(
      stderr,
      "Fail to write matrix to %s (error code = %d)\n", fileName, err);
    exit(-1);
  }
}

template<class T>
static void qsort(int *idx, T *w, int left, int right)
{
  if (left >= right) return;

  swap(idx[left], idx[left + (right - left)/2]);
  swap(w[left], w[left + (right - left)/2]);

  int last = left;
  for (int i = left+1; i <= right; i++) {
    if (idx[i] < idx[left]) {
       ++last;
       swap(idx[last], idx[i]);
       swap(w[last], w[i]);
    }
  }

  swap(idx[left], idx[last]);
  swap(w[left], w[last]);

  qsort(idx, w, left, last-1);
  qsort(idx, w, last+1, right);
}

/* converts COO format to CSR format, not in-place,
   if SORT_IN_ROW is defined, each row is sorted in column index.
assume COO is one-based index */

template<class T>
void coo2csr(int n, int nz, const T *a, const int *i_idx, const int *j_idx,
       T *csr_a, int *col_idx, int *row_start, bool sort)
{
  int i, l;

#pragma omp parallel for
  for (i=0; i<=n; i++) row_start[i] = 0;

  /* determine row lengths */
  for (i=0; i<nz; i++) row_start[i_idx[i]]++;


  for (i=0; i<n; i++) row_start[i+1] += row_start[i];


  /* go through the structure  once more. Fill in output matrix. */
  for (l=0; l<nz; l++){
    i = row_start[i_idx[l] - 1];
    csr_a[i] = a[l];
    col_idx[i] = j_idx[l] - 1;
    row_start[i_idx[l] - 1]++;
  }

  /* shift back row_start */
  for (i=n; i>0; i--) row_start[i] = row_start[i-1];

  row_start[0] = 0;

  if (sort) {
#pragma omp parallel for
    for (i=0; i<n; i++){
      qsort (col_idx, csr_a, row_start[i], row_start[i+1] - 1);
      assert(is_sorted(col_idx + row_start[i], col_idx + row_start[i+1]));
    }
  }
}

void dcoo2csr(int n, int nz, const double *a, const int *i_idx, const int *j_idx,
       double *csr_a, int *col_idx, int *row_start, bool sort/*=true*/)
{
  coo2csr(n, nz, a, i_idx, j_idx, csr_a, col_idx, row_start, sort);
}

void dcoo2csr(const COO *Acoo, CSR *Acrs, bool createSeparateDiagData /*= true*/)
{
  Acrs->n=Acoo->n;
  Acrs->m=Acoo->m;

  dcoo2csr(
    Acrs->n, Acoo->nnz,
    Acoo->values, Acoo->rowidx, Acoo->colidx,
    Acrs->values, Acrs->colidx, Acrs->rowptr);

  if (Acrs->diagptr) {
    if (!Acrs->idiag || !Acrs->diag) {
      createSeparateDiagData = false;
    }
#pragma omp parallel for
    for (int i = 0; i < Acrs->m; ++i) {
      for (int j = Acrs->rowptr[i]; j < Acrs->rowptr[i + 1]; ++j) {
        if (Acrs->colidx[j] == i) {
          Acrs->diagptr[i] = j;

          if (createSeparateDiagData) {
            Acrs->idiag[i] = 1/Acrs->values[j];
            Acrs->diag[i] = Acrs->values[j];
          }
        }
      }
    }
  }
}

void loadMatrixMarket (const char *file, COO &coo, bool force_symmetric /*=false*/, int pad /*=1*/)
{
    FILE *fp=fopen(file, "r");
    if (NULL == fp) {
      fprintf(stderr, "Failed to open file %s\n", file);
      exit(-1);
    }
    MM_typecode matcode;
    int m;
    int n;
    int nnz;
    int x;
    int y;
    double value;
    size_t count;
    int pattern;
    int i;
    int *colidx;
    int *rowidx;
    double *values;
    int lines;

    if (mm_read_banner (fp, &matcode) != 0)
    {
        printf ("Error: could not process Matrix Market banner.\n");
        exit(1);
    }

    if ( !mm_is_valid (matcode) &&
         (mm_is_array (matcode) ||
          mm_is_dense (matcode)) )
    {
        printf ("Error: only support sparse and real matrices.\n");
        exit(1);
    }

    if (mm_read_mtx_crd_size(fp, &m, &n, &nnz) !=0)
    {
        printf ("Error: could not read matrix size.\n");
        exit(1);

    }
    int origM = m, origN = n;
    m = (m + pad - 1)/pad*pad;
    n = (n + pad - 1)/pad*pad;
    assert(m==n);

    if (force_symmetric || mm_is_symmetric (matcode) == 1)
    {
        coo.isSymmetric = true;
        count = 2L*nnz;
    }
    else
    {
        count = nnz;
    }

    coo.m = m;
    coo.n = n;
    size_t extraCount = min(m, n) - min(origM, origN);
    values = MALLOC(double, count + extraCount);
    colidx = MALLOC(int, count + extraCount);
    rowidx = MALLOC(int, count + extraCount);
    assert (values != NULL);
    assert (colidx != NULL);
    assert (rowidx != NULL);

    int *colidx_temp, *rowcnt;
    if (coo.isSymmetric) {
      colidx_temp = MALLOC(int, count);
      rowcnt = MALLOC(int, m);
      memset(rowcnt, 0, sizeof(int)*m);
      assert(colidx_temp != NULL);
      assert(rowcnt != NULL);
    }

    count = 0;
    lines = 0;
    pattern = mm_is_pattern (matcode);
    int x_o=1, y_o;
    double imag;
    while (mm_read_mtx_crd_entry (fp, &x, &y, &value, &imag, matcode) == 0)
    {
        rowidx[count] = x;
        colidx[count] = y;

        if (x > origM || y > origN)
        {
            printf ("Error: (%d %d) coordinate is out of range.\n", x, y);
            exit(1);
        }
        if (pattern == 1)
        {
            values[count] = 1;//RAND01();
        }
        else
        {
            values[count] = (double)value;
        }

        count++;
        lines++;
        if (coo.isSymmetric) rowcnt[x]++;
    }
    for (int i = min(origM, origN); i < min(m, n); ++i) {
      rowidx[count] = i + 1;
      colidx[count] = i + 1;
      values[count] = 1;
      ++count;
    }

    assert (lines == nnz);

  if (coo.isSymmetric) {
    // add transposed elements only if it doesn't exist
    size_t real_count = count;
    // preix-sum
    for (int i = 0; i < m; ++i) {
      rowcnt[i + 1] += rowcnt[i];
    }
    for (int i = 0; i < count; ++i) {
      int j = rowcnt[rowidx[i] - 1];
      colidx_temp[j] = colidx[i];
      rowcnt[rowidx[i] - 1]++;
    }
    for (int i = m; i > 0; --i) {
      rowcnt[i] = rowcnt[i - 1];
    }
    rowcnt[0] = 0;

#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
      sort(colidx_temp + rowcnt[i], colidx_temp + rowcnt[i + 1]);
    }

    for (int i = 0; i < count; ++i) {
      int x = rowidx[i], y = colidx[i];
      if (x != y) {
        if (!binary_search(
            colidx_temp + rowcnt[y - 1], colidx_temp + rowcnt[y], x)) {
          rowidx[real_count] = y;
          colidx[real_count] = x;
          values[real_count] = values[i];
          ++real_count;
        }
      }
    }
    count = real_count;
  }
  nnz = count;

    //printf("count=%d trc=%d\n", count, trc+m);
    nnz = count;

    if (coo.isSymmetric) {
      FREE(rowcnt);
      FREE(colidx_temp);
    }

    coo.nnz = nnz;
    coo.dealloc();
    coo.values = values;
    coo.colidx = colidx;
    coo.rowidx = rowidx;

    fclose(fp);
}

} // namespace SpMP

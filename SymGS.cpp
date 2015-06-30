#include "CSR.hpp"

using namespace std;
using namespace SpMP;

namespace SpMP
{

void splitLU(const CSR& A, CSR *L, CSR *U)
{
  L->dealloc();
  U->dealloc();

  L->m = U->m = A.m;
  L->n = U->n = A.n;

  const int *extptr = A.extptr ? A.extptr : A.rowptr + 1;
  if (A.extptr) {
    U->extptr = MALLOC(int, U->m);
  }

  L->rowptr = MALLOC(int, L->m + 1);
  U->rowptr = MALLOC(int, U->m + 1);
  L->idiag = MALLOC(double, L->m);
  U->idiag = MALLOC(double, U->m);

  // Count # of nnz per row
  int rowPtrPartialSum[2][omp_get_max_threads() + 1];
  rowPtrPartialSum[0][0] = rowPtrPartialSum[1][0] = 0;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int iPerThread = (A.m + nthreads - 1)/nthreads;
    int iBegin = min(tid*iPerThread, A.m);
    int iEnd = min(iBegin + iPerThread, A.m);

    // count # of nnz per row
    int sum0 = 0, sum1 = 0;
    for (int i = iBegin; i < iEnd; ++i) {
      L->rowptr[i] = sum0;
      U->rowptr[i] = sum1;

      for (int j = A.rowptr[i]; j < extptr[i]; ++j) {
        if (A.colidx[j] < i) sum0++;
        if (A.colidx[j] > i) sum1++;
      } // for each element
      sum1 += A.rowptr[i + 1] - extptr[i];
    } // for each row

    rowPtrPartialSum[0][tid + 1] = sum0;
    rowPtrPartialSum[1][tid + 1] = sum1;

#pragma omp barrier
#pragma omp single
    {
      for (int i = 1; i < nthreads; ++i) {
        rowPtrPartialSum[0][i + 1] += rowPtrPartialSum[0][i];
        rowPtrPartialSum[1][i + 1] += rowPtrPartialSum[1][i];
      }
      L->rowptr[L->m] = rowPtrPartialSum[0][nthreads];
      U->rowptr[U->m] = rowPtrPartialSum[1][nthreads];

      int nnzL = L->rowptr[L->m];
      int nnzU = U->rowptr[U->m];

      L->colidx = MALLOC(int, nnzL);
      L->values = MALLOC(double, nnzL);

      U->colidx = MALLOC(int, nnzU);
      U->values = MALLOC(double, nnzU);
    }

    for (int i = iBegin; i < iEnd; ++i) {
      L->rowptr[i] += rowPtrPartialSum[0][tid];
      U->rowptr[i] += rowPtrPartialSum[1][tid];

      if (A.extptr && i > iBegin) {
        U->extptr[i - 1] = U->rowptr[i] - (A.rowptr[i] - extptr[i - 1]);
      }

      int idx0 = L->rowptr[i], idx1 = U->rowptr[i];
      for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
        if (A.colidx[j] < i) {
          L->colidx[idx0] = A.colidx[j];
          L->values[idx0] = A.values[j];
          ++idx0;
        }
        if (A.colidx[j] > i) {
          U->colidx[idx1] = A.colidx[j];
          U->values[idx1] = A.values[j];
          ++idx1;
        }
      }

      L->idiag[i] = A.idiag[i];
      U->idiag[i] = A.idiag[i];
    } // for each row

    if (A.extptr && iEnd > iBegin) {
      U->extptr[iEnd - 1] =
        rowPtrPartialSum[1][tid + 1] - (A.rowptr[iEnd] - extptr[iEnd - 1]);
    }
  } // omp parallel
}

} // namespace SpMP

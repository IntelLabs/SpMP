#include <cmath>
#include <cstring>

#include "../Vector.hpp"
#include "../LevelSchedule.hpp"
#include "../synk/barrier.hpp"

#include "test.hpp"

using namespace SpMP;

/**
 * Reference sequential sparse triangular solver
 */
void forwardSolveRef(const CSR& A, double y[], const double b[])
{
  ADJUST_FOR_BASE;

  for (int i = base; i < A.m + base; ++i) {
    double sum = b[i];
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      sum -= values[j]*y[colidx[j]];
    }
    y[i] = sum*idiag[i];
  } // for each row
}

void backwardSolveRef(const CSR& A, double y[], const double b[])
{
  ADJUST_FOR_BASE;

  for (int i = A.m - 1 + base; i >= base; --i) {
    double sum = b[i];
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      sum -= values[j]*y[colidx[j]];
    }
    y[i] = sum*idiag[i];
  } // for each row
}

/**
 * Forward sparse triangular solver parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardSolveWithReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsForward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    int nPerThread = (ntasks + nthreads - 1)/nthreads;
    int nBegin = min(nPerThread*tid, ntasks);
    int nEnd = min(nBegin + nPerThread, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsForward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task] + base; i < taskBoundaries[task + 1] + base; ++i) {
        double sum = b[i];
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[i] = sum*idiag[i];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward sparse triangular solver parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardSolveWithReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsBackward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    int nPerThread = (ntasks + nthreads - 1)/nthreads;
    int nBegin = min(nPerThread*tid, ntasks);
    int nEnd = min(nBegin + nPerThread, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsBackward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task + 1] - 1 + base; i >= taskBoundaries[task] + base; --i) {
        double sum = b[i];
        for (int j = rowptr[i + 1] - 1; j >= rowptr[i]; --j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[i] = sum*idiag[i];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

// serial ilu0
void ilu0_ref(double *lu, const CSR& A)
{
  int base = A.getBase();

  const int *rowptr = A.rowptr - base;
  const int *colidx = A.colidx - base;
  const int *diagptr = A.diagptr - base;
  const double *values = A.values - base;

  lu -= base;

#pragma omp for
  for (int i = base; i < A.getNnz() + base; i++) {
    lu[i] = values[i];
  }

  for (int i = 0; i < A.m; ++i) {
    for (int j = rowptr[i]; j < diagptr[i]; ++j) {
      int c = colidx[j];
      double tmp = lu[j] /= lu[diagptr[c]];

      int k1 = j + 1, k2 = diagptr[c] + 1;

      while (k1 < rowptr[i + 1] && k2 < rowptr[c + 1]) {
        if (colidx[k1] < colidx[k2]) ++k1;
        else if (colidx[k1] > colidx[k2]) ++k2;
        else {
          lu[k1] -= tmp*lu[k2];
          ++k1; ++k2;
        }
      }
    }
  } // for each row
}

// parallel ilu0
void ilu0(double *lu, const CSR& A, const LevelSchedule& schedule)
{
  int base = A.getBase();

  const int *rowptr = A.rowptr - base;
  const int *colidx = A.colidx - base;
  const int *diagptr = A.diagptr - base;
  const double *values = A.values - base;

  lu -= base;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

#pragma omp for
    for (int i = base; i < A.getNnz() + base; i++) {
      lu[i] = values[i];
    }

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsForward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    const int *perm = schedule.threadContToOrigPerm;

    int nBegin, nEnd;
    getSimpleThreadPartition(&nBegin, &nEnd, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsForward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
        int row = perm[i] + base;

        for (int j = rowptr[row]; j < diagptr[row]; ++j) {
          int c = colidx[j];
          double tmp = lu[j] /= lu[diagptr[c]];

          int k1 = j + 1, k2 = diagptr[c] + 1;

          while (k1 < rowptr[row + 1] && k2 < rowptr[c + 1]) {
            if (colidx[k1] < colidx[k2]) ++k1;
            else if (colidx[k1] > colidx[k2]) ++k2;
            else {
              lu[k1] -= tmp*lu[k2];
              ++k1; ++k2;
            }
          }
        }
      } // for each row

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each level
  } // omp parallel
}

double dot(const double x[], const double y[], int len)
{
  double sum = 0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < len; ++i) {
    sum += x[i]*y[i];
  }
  return sum;
}

double norm(const double x[], int len)
{
  return sqrt(dot(x, x, len));
}

// w = a*x + b*y
void waxpby(int n, double w[], double alpha, const double x[], double beta, const double y[])
{
  if (1 == alpha) {
    if (1 == beta) {
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        w[i] = x[i] + y[i];
      }
    }
    else if (-1 == beta) {
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        w[i] = x[i] - y[i];
      }
    }
    else {
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        w[i] = x[i] + beta*y[i];
      }
    }
  }
  else if (1 == beta) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      w[i] = alpha*x[i] + y[i];
    }
  }
  else if (-1 == alpha) {
    if (0 == beta) {
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        w[i] = -x[i];
      }
    }
    else {
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        w[i] = beta*y[i] - x[i];
      }
    }
  }
  else if (0 == beta) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      w[i] = alpha*x[i];
    }
  }
  else {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      w[i] = alpha*x[i] + beta*y[i];
    }
  }
}

void pcg_symgs(CSR *A, double *x, const double *b, double tol, int maxiter)
{
  double spmv_time = 0;
  double trsv_time = 0;

  // Construct SymGS pre-conditioner
  CSR *L = new CSR, *U = new CSR;
  splitLU(*A, L, U);
#pragma omp parallel for
  for (int i = 0; i < U->m; ++i) {
    for (int j = U->rowptr[i]; j < U->rowptr[i + 1]; ++j) {
      U->values[j] *= U->idiag[i];
    }
    U->idiag[i] = 1;
  }

  double *r = MALLOC(double, A->m);
  spmv_time -= omp_get_wtime();
  A->multiplyWithVector(r, -1, x, 1, b, 0); // r = b - A*x
  spmv_time += omp_get_wtime();
  double normr0 = norm(r, A->m);
  double rel_err = 1.;

  double *z = MALLOC(double, A->m);
  double *y = MALLOC(double, A->m);
  // z = M\r, where M is pre-conditioner
  trsv_time -= omp_get_wtime();
  forwardSolveRef(*L, y, r);
  backwardSolveRef(*U, z, y);
  trsv_time += omp_get_wtime();

  double *p = MALLOC(double, A->m);
  copyVector(p, z, A->m);
  double rz = dot(r, z, A->m);
  int k = 1;

  double *Ap = MALLOC(double, A->m);

  while (k <= maxiter) {
    double old_rz = rz;

    spmv_time -= omp_get_wtime();
    A->multiplyWithVector(Ap, p); // Ap = A*p
    spmv_time += omp_get_wtime();

    double alpha = old_rz/dot(p, Ap, A->m);
    waxpby(A->m, x, 1, x, alpha, p); // x += alpha*p
    waxpby(A->m, r, 1, r, -alpha, Ap); // r -= alpha*Ap
    rel_err = norm(r, A->m)/normr0;
    if (rel_err < tol) break;

    trsv_time -= omp_get_wtime();
    forwardSolveRef(*L, y, r);
    backwardSolveRef(*U, z, y);
    trsv_time += omp_get_wtime();

    rz = dot(r, z, A->m);
    double beta = rz/old_rz;
    waxpby(A->m, p, 1, z, beta, p); // p = z + beta*p
    ++k;
  }

  printf("iter = %d rel_err = %g\n", k, rel_err);

  double spmv_bytes = (k + 1)*(12.*A->getNnz() + (4. + 2*8)*A->m);
  double trsv_bytes = k*(12.*L->getNnz() + 12.*U->getNnz() + (8. + 2*4 + 2*2*8)*L->m);
  printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes/spmv_time/1e9, trsv_bytes/trsv_time/1e9);

  delete L;
  delete U;

  FREE(r);
  FREE(z);
  FREE(y);
  FREE(p);
  FREE(Ap);
}

void pcg_symgs_opt(CSR *A, double *x, const double *b, double tol, int maxiter)
{
  double spmv_time = 0;
  double trsv_time = 0;

  // Construct SymGS pre-conditioner
  CSR *L = new CSR, *U = new CSR;
  splitLU(*A, L, U);
#pragma omp parallel for
  for (int i = 0; i < U->m; ++i) {
    for (int j = U->rowptr[i]; j < U->rowptr[i + 1]; ++j) {
      U->values[j] *= U->idiag[i];
    }
    U->idiag[i] = 1;
  }

  LevelSchedule schedule;
  schedule.constructTaskGraph(*A);
  printf("parallelism = %g\n", (double)A->m/(schedule.levIndices.size() - 1));

  const int *perm = schedule.origToThreadContPerm;
  const int *invPerm = schedule.threadContToOrigPerm;

  CSR *APerm = A->permute(perm, invPerm);
  CSR *LPerm = L->permute(perm, invPerm);
  CSR *UPerm = U->permute(perm, invPerm);

  delete L;
  delete U;

  double *bPerm = getReorderVectorWithInversePerm(b, invPerm, A->m);
  double *xPerm = getReorderVectorWithInversePerm(x, invPerm, A->m);

  double *r = MALLOC(double, A->m);
  spmv_time -= omp_get_wtime();
  APerm->multiplyWithVector(r, -1, xPerm, 1, bPerm, 0); // r = b - A*x
  spmv_time += omp_get_wtime();
  double normr0 = norm(r, A->m);
  double rel_err = 1.;

  double *z = MALLOC(double, A->m);
  double *y = MALLOC(double, A->m);
  // z = M\r, where M is pre-conditioner
  trsv_time -= omp_get_wtime();
  forwardSolveWithReorderedMatrix(*LPerm, y, r, schedule);
  backwardSolveWithReorderedMatrix(*UPerm, z, y, schedule);
  trsv_time += omp_get_wtime();

  double *p = MALLOC(double, A->m);
  copyVector(p, z, A->m);
  double rz = dot(r, z, A->m);
  int k = 1;

  double *Ap = MALLOC(double, A->m);

  while (k <= maxiter) {
    double old_rz = rz;

    spmv_time -= omp_get_wtime();
    APerm->multiplyWithVector(Ap, p); // Ap = A*p
    spmv_time += omp_get_wtime();

    double alpha = old_rz/dot(p, Ap, A->m);
    waxpby(A->m, xPerm, 1, xPerm, alpha, p); // x += alpha*p
    waxpby(A->m, r, 1, r, -alpha, Ap); // r -= alpha*Ap
    rel_err = norm(r, A->m)/normr0;
    if (rel_err < tol) break;

    trsv_time -= omp_get_wtime();
    forwardSolveWithReorderedMatrix(*LPerm, y, r, schedule);
    backwardSolveWithReorderedMatrix(*UPerm, z, y, schedule);
    trsv_time += omp_get_wtime();

    rz = dot(r, z, A->m);
    double beta = rz/old_rz;
    waxpby(A->m, p, 1, z, beta, p); // p = z + beta*p
    ++k;
  }

  reorderVectorOutOfPlaceWithInversePerm(x, xPerm, perm, A->m);

  printf("iter = %d, rel_err = %g\n", k, rel_err);

  double spmv_bytes = (k + 1)*(12.*A->getNnz() + (4. + 2*8)*A->m);
  double trsv_bytes = k*(12.*LPerm->getNnz() + 12.*UPerm->getNnz() + (8. + 2*4 + 2*2*8)*LPerm->m);
  printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes/spmv_time/1e9, trsv_bytes/trsv_time/1e9);

  delete APerm;
  delete LPerm;
  delete UPerm;

  FREE(r);
  FREE(z);
  FREE(y);
  FREE(p);
  FREE(Ap);

  FREE(bPerm);
  FREE(xPerm);
}

void pcg_ilu0(CSR *A, double *x, const double *b, double tol, int maxiter)
{
  double spmv_time = 0;
  double trsv_time = 0;

  // Construct ILU0 pre-conditioner
  double *lu = MALLOC(double, A->getNnz());
  ilu0_ref(lu, *A);
  CSR LU(A->m, A->n, A->rowptr, A->colidx, lu);
  LU.computeInverseDiag();

  CSR *L = new CSR, *U = new CSR;
  splitLU(LU, L, U);
#pragma omp parallel for
  for (int i = 0; i < L->m; ++i) {
    L->idiag[i] = 1;
  }

  double *r = MALLOC(double, A->m);
  spmv_time -= omp_get_wtime();
  A->multiplyWithVector(r, -1, x, 1, b, 0); // r = b - A*x
  spmv_time += omp_get_wtime();
  double normr0 = norm(r, A->m);
  double rel_err = 1.;

  double *z = MALLOC(double, A->m);
  double *y = MALLOC(double, A->m);
  // z = M\r, where M is pre-conditioner
  trsv_time -= omp_get_wtime();
  forwardSolveRef(*L, y, r);
  backwardSolveRef(*U, z, y);
  trsv_time += omp_get_wtime();

  double *p = MALLOC(double, A->m);
  copyVector(p, z, A->m);
  double rz = dot(r, z, A->m);
  int k = 1;

  double *Ap = MALLOC(double, A->m);

  while (k <= maxiter) {
    double old_rz = rz;

    spmv_time -= omp_get_wtime();
    A->multiplyWithVector(Ap, p); // Ap = A*p
    spmv_time += omp_get_wtime();

    double alpha = old_rz/dot(p, Ap, A->m);
    waxpby(A->m, x, 1, x, alpha, p); // x += alpha*p
    waxpby(A->m, r, 1, r, -alpha, Ap); // r -= alpha*Ap
    rel_err = norm(r, A->m)/normr0;
    if (rel_err < tol) break;

    trsv_time -= omp_get_wtime();
    forwardSolveRef(*L, y, r);
    backwardSolveRef(*U, z, y);
    trsv_time += omp_get_wtime();

    rz = dot(r, z, A->m);
    double beta = rz/old_rz;
    waxpby(A->m, p, 1, z, beta, p); // p = z + beta*p
    ++k;
  }

  printf("iter = %d rel_err = %g\n", k, rel_err);

  double spmv_bytes = (k + 1)*(12.*A->getNnz() + (4. + 2*8)*A->m);
  double trsv_bytes = k*(12.*L->getNnz() + 12.*U->getNnz() + (8. + 2*4 + 2*2*8)*L->m);
  printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes/spmv_time/1e9, trsv_bytes/trsv_time/1e9);

  delete L;
  delete U;

  FREE(r);
  FREE(z);
  FREE(y);
  FREE(p);
  FREE(Ap);
}

void pcg_ilu0_opt(CSR *A, double *x, const double *b, double tol, int maxiter)
{
  double spmv_time = 0;
  double trsv_time = 0;

  LevelSchedule schedule;
  schedule.constructTaskGraph(*A);
  printf("parallelism = %g\n", (double)A->m/(schedule.levIndices.size() - 1));

  // Construct ILU0 pre-conditioner
  double *lu = MALLOC(double, A->getNnz());
  ilu0(lu, *A, schedule);
  CSR LU(A->m, A->n, A->rowptr, A->colidx, lu);
  LU.computeInverseDiag();

  CSR *L = new CSR, *U = new CSR;
  splitLU(LU, L, U);
#pragma omp parallel for
  for (int i = 0; i < L->m; ++i) {
    L->idiag[i] = 1;
  }

  const int *perm = schedule.origToThreadContPerm;
  const int *invPerm = schedule.threadContToOrigPerm;

  CSR *APerm = A->permute(perm, invPerm);
  CSR *LPerm = L->permute(perm, invPerm);
  CSR *UPerm = U->permute(perm, invPerm);

  delete L;
  delete U;

  double *bPerm = getReorderVectorWithInversePerm(b, invPerm, A->m);
  double *xPerm = getReorderVectorWithInversePerm(x, invPerm, A->m);

  double *r = MALLOC(double, A->m);
  spmv_time -= omp_get_wtime();
  APerm->multiplyWithVector(r, -1, xPerm, 1, bPerm, 0); // r = b - A*x
  spmv_time += omp_get_wtime();
  double normr0 = norm(r, A->m);
  double rel_err = 1.;

  double *z = MALLOC(double, A->m);
  double *y = MALLOC(double, A->m);
  // z = M\r, where M is pre-conditioner
  trsv_time -= omp_get_wtime();
  forwardSolveWithReorderedMatrix(*LPerm, y, r, schedule);
  backwardSolveWithReorderedMatrix(*UPerm, z, y, schedule);
  trsv_time += omp_get_wtime();

  double *p = MALLOC(double, A->m);
  copyVector(p, z, A->m);
  double rz = dot(r, z, A->m);
  int k = 1;

  double *Ap = MALLOC(double, A->m);

  while (k <= maxiter) {
    double old_rz = rz;

    spmv_time -= omp_get_wtime();
    APerm->multiplyWithVector(Ap, p); // Ap = A*p
    spmv_time += omp_get_wtime();

    double alpha = old_rz/dot(p, Ap, A->m);
    waxpby(A->m, xPerm, 1, xPerm, alpha, p); // x += alpha*p
    waxpby(A->m, r, 1, r, -alpha, Ap); // r -= alpha*Ap
    rel_err = norm(r, A->m)/normr0;
    if (rel_err < tol) break;

    trsv_time -= omp_get_wtime();
    forwardSolveWithReorderedMatrix(*LPerm, y, r, schedule);
    backwardSolveWithReorderedMatrix(*UPerm, z, y, schedule);
    trsv_time += omp_get_wtime();

    rz = dot(r, z, A->m);
    double beta = rz/old_rz;
    waxpby(A->m, p, 1, z, beta, p); // p = z + beta*p
    ++k;
  }

  reorderVectorOutOfPlaceWithInversePerm(x, xPerm, perm, A->m);

  printf("iter = %d, rel_err = %g\n", k, rel_err);

  double spmv_bytes = (k + 1)*(12.*A->getNnz() + (4. + 2*8)*A->m);
  double trsv_bytes = k*(12.*LPerm->getNnz() + 12.*UPerm->getNnz() + (8. + 2*4 + 2*2*8)*LPerm->m);
  printf("spmv_perf = %g gbps trsv_perf = %g gbps\n", spmv_bytes/spmv_time/1e9, trsv_bytes/trsv_time/1e9);

  delete APerm;
  delete LPerm;
  delete UPerm;

  FREE(r);
  FREE(z);
  FREE(y);
  FREE(p);
  FREE(Ap);

  FREE(bPerm);
  FREE(xPerm);
}

int main(int argc, char *argv[])
{
  int m = argc > 1 ? atoi(argv[1]) : 64; // default input is 64^3 27-pt 3D Lap.
  if (argc < 2) {
    fprintf(
      stderr,
      "Using default 64^3 27-pt 3D Laplacian matrix\n"
      "-- Usage examples --\n"
      "  %s 128 : 128^3 27-pt 3D Laplacian matrix\n"
      "  %s inline_1: run with inline_1.mtx matrix in matrix market format\n\n",
      argv[0], argv[0]);
  }

  char buf[1024];
  sprintf(buf, "%d", m);
  printf("input=%s\n", argc > 1 ? argv[1] : buf);

  CSR A(argc > 1 ? argv[1] : buf);
  double *x = MALLOC(double, A.m);
  double *b = NULL;
  if (argc > 1 && strlen(argv[1]) >= 4) {
    // load rhs file if it exists
    char buf2[1024];
    strncpy(buf2, argv[1], strlen(argv[1]) - 4);
    int m, n;
    loadVectorMatrixMarket((string(buf2) + "_b.mtx").c_str(), &b, &m, &n);
  }
  if (!b) {
    b = MALLOC(double, A.m);
    for (int i = 0; i < A.m; ++i) b[i] = 1;
  }

  double tol = 1e-7;
  int maxiter = 20000;

  for (int i = 0; i < A.m; ++i) x[i] = 0;
  pcg_symgs(&A, x, b, tol, maxiter);

  for (int i = 0; i < A.m; ++i) x[i] = 0;
  pcg_symgs_opt(&A, x, b, tol, maxiter);

  for (int i = 0; i < A.m; ++i) x[i] = 0;
  pcg_ilu0(&A, x, b, tol, maxiter);

  for (int i = 0; i < A.m; ++i) x[i] = 0;
  pcg_ilu0_opt(&A, x, b, tol, maxiter);

  FREE(x);
  FREE(b);

  return 0;
}

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

/*!
 * \brief Example of parallelizing GS-like loops with data-dependent
 *        loop carried dependencies.
 *        This example runs symmetric GS smoothing, but SpMP also can
 *        be used for sparse triangular solver, ILU factorization, and so on.
 *
 * \ref "Sparsifying Synchronizations for High-Performance Shared-Memory Sparse
 *      Triangular Solver", Park et al., ISC 2014
 *
 * Expected performance
   (inline_1.mtx can be downloaded from U of Florida matrix collection)
  
 In a 18-core Xeon E5-2699 v3 @ 2.3GHz, 56 gbps STREAM BW

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/gs_test 192
input=192
parallelism 5289.901345
ref             1.71 1.43 gflops 10.37  8.67 gbps
barrier         3.07 2.96 gflops 18.66 18.00 gbps
p2p             5.09 4.07 gflops 30.91 24.75 gbps
p2p_tr_red      4.77 3.95 gflops 28.97 23.98 gbps
barrier_perm    2.71 2.14 gflops 16.46 13.01 gbps
p2p_perm        9.06 9.35 gflops 55.02 56.78 gbps
p2p_tr_red_perm 9.08 9.35 gflops 55.15 56.83 gbps

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/gs_test inline_1.mtx
input=/home/jpark103/matrices/inline_1.mtx
/home/jpark103/matrices/inline_1.mtx:::symmetric m=503712 n=503712 nnz=36816342
parallelism 287.506849
ref             1.80 1.27 gflops 10.86  7.68 gbps
barrier         2.80 1.99 gflops 16.89 11.98 gbps
p2p             4.40 4.01 gflops 26.52 24.17 gbps
p2p_tr_red      4.52 4.15 gflops 27.26 25.00 gbps
barrier_perm    1.81 1.96 gflops 10.89 11.81 gbps
p2p_perm        8.02 7.53 gflops 48.33 45.37 gbps
p2p_tr_red_perm 8.95 8.65 gflops 53.96 52.13 gbps

 */

#include <cassert>
#include <cstring>
#include <climits>
#include <cfloat>

#include <omp.h>

#include "../CSR.hpp"
#include "../LevelSchedule.hpp"
#include "../synk/barrier.hpp"

using namespace std;
using namespace SpMP;

synk::Barrier *bar;

typedef enum
{
  REFERENCE = 0,
  BARRIER,
  P2P,
  P2P_WITH_TRANSITIVE_REDUCTION,
} Option;

static void printEfficiency(
  double timeForward, double timeBackward, double flop, double byte)
{
  printf(
    "%7.2f %7.2f gflops %7.2f %7.2f gbps\n",
    flop/timeForward/1e9, flop/timeBackward/1e9,
    byte/timeForward/1e9, byte/timeBackward/1e9);
}

bool correctnessCheck(CSR *A, double *y, double *b)
{
  static double *yt = NULL;
  if (NULL == yt) {
    yt = new double[A->m];

    std::copy(y, y + A->m, yt);
  }
  else {
    for (int i = 0; i < A->m; i++) {
      if (fabs(y[i] - yt[i])/fabs(yt[i]) >= 1e-8) {
        printf("y error at %d expected %e actual %e\n", i, yt[i], y[i]);
        return false;
      }
    }
  }

  return true;
}

static const size_t LLC_CAPACITY = 32*1024*1024;
static const double *bufToFlushLlc = NULL;

void flushLlc()
{
  double sum = 0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < LLC_CAPACITY/sizeof(bufToFlushLlc[0]); ++i) {
    sum += bufToFlushLlc[i];
  }
  FILE *fp = fopen("/dev/null", "w");
  fprintf(fp, "%f\n", sum);
  fclose(fp);
}

void initializeX(double *x, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) x[i] = n - i;
}

//#define DBG_GS
#ifdef DBG_GS
const int ROW_TO_DEBUG = 3;
#endif

/**
 * Reference sequential Gauss-Seidel smoother
 */
template<bool IS_FORWARD>
void gsRef(const CSR& A, double y[], const double b[])
{
  int iBegin = IS_FORWARD ? 0 : A.m - 1;
  int iEnd = IS_FORWARD ? A.m : -1;
  int iDelta = IS_FORWARD ? 1 : -1;

  for (int i = iBegin; i != iEnd; i += iDelta) {
    double sum = b[i];
    for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
      sum -= A.values[j]*y[A.colidx[j]];
    }
    y[i] += sum*A.idiag[i];
  } // for each row
}

void forwardGSRef(const CSR& A, double y[], const double b[])
{
  gsRef<true>(A, y, b);
}

void backwardGSRef(const CSR& A, double y[], const double b[])
{
  gsRef<false>(A, y, b);
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization
 */
void forwardGSWithBarrier(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
#pragma omp parallel
  {
    for (int l = 0; l < schedule.levIndices.size() - 1; ++l) {
#pragma omp for nowait
      for (int i = schedule.levIndices[l]; i < schedule.levIndices[l + 1]; ++i) {
        int row = perm[i];
        double sum = b[row];
        for (int j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[row] += sum*A.idiag[row];
      } // for each row
      bar->wait(omp_get_thread_num());
    } // for each level
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization
 */
void backwardGSWithBarrier(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
#pragma omp parallel
  {
    for (int l = schedule.levIndices.size() - 2; l >= 0; --l) {
#pragma omp for nowait
      for (int i = schedule.levIndices[l + 1] - 1; i >= schedule.levIndices[l]; --i) {
        int row = perm[i];
        double sum = b[row];
        for (int j = A.rowptr[row + 1] - 1; j >= A.rowptr[row]; --j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[row] += sum*A.idiag[row];
      } // for each row
      bar->wait(omp_get_thread_num());
    } // for each level
  } // omp parallel
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardGS(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
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

    bar->wait(tid);

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
        int row = perm[i];
        double sum = b[row];
        for (int j = A.rowptr[row]; j < A.rowptr[row + 1]; ++j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[row] += sum*A.idiag[row];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardGS(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
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

    bar->wait(tid);

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task + 1] - 1; i >= taskBoundaries[task]; --i) {
        int row = perm[i];
        double sum = b[row];
        for (int j = A.rowptr[row + 1] - 1; j >= A.rowptr[row]; --j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[row] += sum*A.idiag[row];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization. Matrix is reordered.
 */
void forwardGSWithBarrierAndReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
#pragma omp parallel
  {
    for (int l = 0; l < schedule.levIndices.size() - 1; ++l) {
#pragma omp for nowait
      for (int i = schedule.levIndices[l]; i < schedule.levIndices[l + 1]; ++i) {
        double sum = b[i];
        for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[i] += sum*A.idiag[i];
      } // for each row
      bar->wait(omp_get_thread_num());
    } // for each level
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization. Matrix is reordered.
 */
void backwardGSWithBarrierAndReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
#pragma omp parallel
  {
    for (int l = schedule.levIndices.size() - 2; l >= 0; --l) {
#pragma omp for nowait
      for (int i = schedule.levIndices[l + 1] - 1; i >= schedule.levIndices[l]; --i) {
        double sum = b[i];
        for (int j = A.rowptr[i + 1] - 1; j >= A.rowptr[i]; --j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[i] += sum*A.idiag[i];
      } // for each row
      bar->wait(omp_get_thread_num());
    } // for each level
  } // omp parallel
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardGSWithReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
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

    bar->wait(tid);

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
        double sum = b[i];
        for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[i] += sum*A.idiag[i];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardGSWithReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
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

    bar->wait(tid);

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task + 1] - 1; i >= taskBoundaries[task]; --i) {
        double sum = b[i];
        for (int j = A.rowptr[i + 1] - 1; j >= A.rowptr[i]; --j) {
          sum -= A.values[j]*y[A.colidx[j]];
        }
        y[i] += sum*A.idiag[i];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

int main(int argc, char **argv)
{
  double tBegin = omp_get_wtime();

  /////////////////////////////////////////////////////////////////////////////
  // Initialize barrier
  /////////////////////////////////////////////////////////////////////////////

  int nthreads = omp_get_max_threads();

#ifdef __MIC__
  bar = new synk::Barrier(omp_get_max_threads()/4, 4);
#else
  bar = new synk::Barrier(omp_get_max_threads(), 1);
#endif

#pragma omp parallel
  {
    bar->init(omp_get_thread_num());
  }

  /////////////////////////////////////////////////////////////////////////////
  // Load input
  /////////////////////////////////////////////////////////////////////////////

  int m = argc > 1 ? atoi(argv[1]) : 64; // default input is 64^3 27-pt 3D Lap.
  if (argc < 2) {
    fprintf(
      stderr,
      "Using default 64^3 27-pt 3D Laplacian matrix\n"
      "-- Usage examples --\n"
      "  %s 128 : 128^3 27-pt 3D Laplacian matrix\n"
      "  %s inline_1.mtx: run with inline_1 matrix in matrix market format\n\n",
      argv[0], argv[0]);
  }
  char buf[1024];
  sprintf(buf, "%d", m);

  bool readFromFile = argc > 1 ? strcmp(buf, argv[1]) && !strstr(argv[1], ".mtx"): false;
  printf("input=%s\n", argc > 1 ? argv[1] : buf);

  CSR *A = new CSR(argc > 1 ? argv[1] : buf);

  /////////////////////////////////////////////////////////////////////////////
  // Construct schedules
  /////////////////////////////////////////////////////////////////////////////

  LevelSchedule *barrierSchedule = new LevelSchedule;
  barrierSchedule->useBarrier = true;
  barrierSchedule->findLevels(*A, true /* load balancing for smoothing*/);
  barrierSchedule->constructTaskGraph(
    *A,
    false); // !transitiveReduction

  LevelSchedule *p2pSchedule = new LevelSchedule;
  p2pSchedule->findLevels(*A, true /* load balancing for smoothing*/);
  p2pSchedule->constructTaskGraph(
    *A,
    false); // !transitiveReduction

  LevelSchedule *p2pScheduleWithTransitiveReduction = new LevelSchedule;
  p2pScheduleWithTransitiveReduction->findLevels(*A, true /* load balancing for smoothing*/);
  p2pScheduleWithTransitiveReduction->constructTaskGraph(
    *A,
    true); // transitiveReduction

  printf("parallelism %f\n", (double)A->m/(barrierSchedule->levIndices.size() - 1));
  assert(barrierSchedule->levIndices.size() == p2pSchedule->levIndices.size());
  assert(barrierSchedule->levIndices.size() == p2pScheduleWithTransitiveReduction->levIndices.size());

  /////////////////////////////////////////////////////////////////////////////
  // Reorder matrix
  /////////////////////////////////////////////////////////////////////////////

  const int *permBarrier = barrierSchedule->origToLevContPerm;
  const int *invPermBarrier = barrierSchedule->levContToOrigPerm;

  const int *permP2P = p2pScheduleWithTransitiveReduction->origToThreadContPerm;
  const int *invPermP2P = p2pScheduleWithTransitiveReduction->threadContToOrigPerm;
  assert(isPerm(permP2P, A->m));
  assert(isPerm(invPermP2P, A->m));

  CSR *APermBarrier = A->permute(permBarrier, invPermBarrier, true);
  CSR *APermP2P = A->permute(permP2P, invPermP2P, true);

  /////////////////////////////////////////////////////////////////////////////
  // Allocate vectors
  /////////////////////////////////////////////////////////////////////////////

  double *b = MALLOC(double, A->m);
#pragma omp parallel for
  for(int i=0; i < A->m; i++) b[i] = i;

  double *y = MALLOC(double, A->m);
  double *x = MALLOC(double, A->m);

  double flop[2], byte[2];
  for (int i = 0; i < 2; ++i) {
    int nnz = A->rowptr[A->m];
    flop[i] = 2*nnz;
    byte[i] = nnz*(sizeof(double) + sizeof(int)) + A->m*sizeof(int);
  }

  // allocate a large buffer to flush out cache
  bufToFlushLlc = (double *)_mm_malloc(LLC_CAPACITY, 64);

  /////////////////////////////////////////////////////////////////////////////
  // GS smoother w/o reordering
  /////////////////////////////////////////////////////////////////////////////

  int REPEAT = 128;
  for (int o = REFERENCE; o <= P2P_WITH_TRANSITIVE_REDUCTION; ++o) {
    Option option = (Option)o;
    double minTimeForward = DBL_MAX, minTimeBackward = DBL_MAX;

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      initializeX(x, A->m);
      initializeX(y, A->m);

      double t = omp_get_wtime();

      switch (option) {
      case REFERENCE :
        forwardGSRef(*A, y, b); break;
      case BARRIER :
        forwardGSWithBarrier(*A, y, b, *barrierSchedule, invPermBarrier); break;
      case P2P :
        forwardGS(*A, y, b, *p2pSchedule, invPermP2P); break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        forwardGS(*A, y, b, *p2pScheduleWithTransitiveReduction, invPermP2P); break;
      default: assert(false); break;
      }

      minTimeForward = min(omp_get_wtime() - t, minTimeForward);
      t = omp_get_wtime();

      switch (option) {
      case REFERENCE :
        backwardGSRef(*A, x, y); break;
      case BARRIER :
        backwardGSWithBarrier(*A, x, y, *barrierSchedule, invPermBarrier); break;
      case P2P :
        backwardGS(*A, x, y, *p2pSchedule, invPermP2P); break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        backwardGS(*A, x, y, *p2pScheduleWithTransitiveReduction, invPermP2P); break;
      default: assert(false); break;
      }

      minTimeBackward = min(omp_get_wtime() - t, minTimeBackward);

      if (i == REPEAT - 1) {
        switch (option) {
        case REFERENCE : printf("ref\t\t"); break;
        case BARRIER: printf("barrier\t\t"); break;
        case P2P: printf("p2p\t\t"); break;
        case P2P_WITH_TRANSITIVE_REDUCTION: printf("p2p_tr_red\t"); break;
        default: assert(false); break;
        }
        printEfficiency(minTimeForward, minTimeBackward, flop[0], byte[0]);

        correctnessCheck(A, x, b);
      }
    } // for each iteration
  } // for each option

  /////////////////////////////////////////////////////////////////////////////
  // GS smoother w/ reordering
  /////////////////////////////////////////////////////////////////////////////

  double *bPermBarrier = getReorderVector(b, permBarrier, A->m);
  double *bPermP2P = getReorderVector(b, permP2P, A->m);
  double *tempVector = MALLOC(double, A->m);

  for (int o = BARRIER; o <= P2P_WITH_TRANSITIVE_REDUCTION; ++o) {
    Option option = (Option)o;
    double minTimeForward = DBL_MAX, minTimeBackward = DBL_MAX;

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      initializeX(x, A->m);
      initializeX(y, A->m);
      reorderVector(
        x, tempVector, BARRIER == option ? permBarrier : permP2P, A->m);
      reorderVector(
        y, tempVector, BARRIER == option ? permBarrier : permP2P, A->m);

      double t = omp_get_wtime();

      switch (option) {
      case BARRIER :
        forwardGSWithBarrierAndReorderedMatrix(
          *APermBarrier, y, bPermBarrier, *barrierSchedule);
        break;
      case P2P :
        forwardGSWithReorderedMatrix(
          *APermP2P, y, bPermP2P, *p2pSchedule);
        break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        forwardGSWithReorderedMatrix(
          *APermP2P, y, bPermP2P, *p2pScheduleWithTransitiveReduction);
        break;
      default: assert(false); break;
      }

      minTimeForward = min(omp_get_wtime() - t, minTimeForward);
      t = omp_get_wtime();

      switch (option) {
      case BARRIER :
        backwardGSWithBarrierAndReorderedMatrix(
          *APermBarrier, x, y, *barrierSchedule);
        break;
      case P2P :
        backwardGSWithReorderedMatrix(
          *APermP2P, x, y, *p2pSchedule);
        break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        backwardGSWithReorderedMatrix(
          *APermP2P, x, y, *p2pScheduleWithTransitiveReduction);
        break;
      default: assert(false); break;
      }

      minTimeBackward = min(omp_get_wtime() - t, minTimeBackward);

      if (i == REPEAT - 1) {
        switch (option) {
        case BARRIER: printf("barrier_perm\t"); break;
        case P2P: printf("p2p_perm\t"); break;
        case P2P_WITH_TRANSITIVE_REDUCTION: printf("p2p_tr_red_perm\t"); break;
        default: assert(false); break;
        }
        printEfficiency(minTimeForward, minTimeBackward, flop[0], byte[0]);

        reorderVector(
          x, tempVector, BARRIER == option ? invPermBarrier : invPermP2P, A->m);
        correctnessCheck(A, x, b);
      }
    }
  }

  delete barrierSchedule;
  delete p2pSchedule;
  delete p2pScheduleWithTransitiveReduction;

  delete A;
  delete APermBarrier;
  delete APermP2P;

  FREE(b);
  FREE(y);
  FREE(bPermBarrier);
  FREE(bPermP2P);
  FREE(tempVector);

  return 0;
}

#include "../reordering/BFSBipartite.hpp"
#include "test.hpp"

using namespace SpMP;

int main(int argc, char **argv)
{
  CSR *A = new CSR(argv[1], 1);
  CSR *AT = A->transpose();

  int nnz = A->getNnz();
  double flops = 2*nnz;
  double bytes = (sizeof(double) + sizeof(int))*nnz + sizeof(double)*(A->m + A->n);

  printf("original bandwidth %d\n", A->getBandwidth());

  double *x = MALLOC(double, A->n);
  double *y = MALLOC(double, A->m);

  // allocate a large buffer to flush out cache
  bufToFlushLlc = (double *)_mm_malloc(LLC_CAPACITY, 64);

  const int REPEAT = 128;
  double times[REPEAT];

  for (int i = 0; i < REPEAT; ++i) {
    flushLlc();

    double t = omp_get_wtime();
    A->multiplyWithVector(y, x);
    times[i] = omp_get_wtime() - t;
  }

  printf("SpMV BW");
  printEfficiency(times, REPEAT, flops, bytes);

  for (int i = 0; i < REPEAT; ++i) {
    flushLlc();

    double t = omp_get_wtime();
    AT->multiplyWithVector(x, y);
    times[i] = omp_get_wtime() - t;
  }

  printf("SpMVT BW");
  printEfficiency(times, REPEAT, flops, bytes);

  int *rowPerm = new int[A->m];
  int *rowInversePerm = new int[A->m];
  int *colPerm = new int[A->n];
  int *colInversePerm = new int[A->n];

  bfsBipartite(*A, *AT, rowPerm, rowInversePerm, colPerm, colInversePerm);

  //for (int i = 0; i < A->n; ++i) {
    //colPerm[i] = i;
  //}

  CSR *APerm = A->permute(colPerm, rowInversePerm);
  CSR *ATPerm = A->transpose();

  printf("Permuted bandwidth %d\n", APerm->getBandwidth());

  for (int i = 0; i < REPEAT; ++i) {
    flushLlc();

    double t = omp_get_wtime();
    APerm->multiplyWithVector(y, x);
    times[i] = omp_get_wtime() - t;
  }

  printf("SpMV BW");
  printEfficiency(times, REPEAT, flops, bytes);

  for (int i = 0; i < REPEAT; ++i) {
    flushLlc();

    double t = omp_get_wtime();
    ATPerm->multiplyWithVector(x, y);
    times[i] = omp_get_wtime() - t;
  }

  printf("SpMV BW");
  printEfficiency(times, REPEAT, flops, bytes);

  return 0;
}

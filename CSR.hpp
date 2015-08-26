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

#pragma once

#include <cstdlib>
#include <vector>
#include <string>

#include "MemoryPool.hpp"
#include "Utils.hpp"

namespace SpMP
{

class CSR {
public:
  int m;
  int n;
  int *rowptr;
  int *colidx;
  double *values;

  int base; // 0: 0-based, 0: 1-based

  double *idiag; /**< inverse of diagonal elements */
  double *diag;
  int *diagptr;
  int *extptr; // points to the beginning of non-local columns (when MPI is used)

  //std::vector<int> levvec;

  CSR();

  // Following two constructors will make CSR own the data
  CSR(const char *file, bool forceSymmetric = false, int pad = 1);
  CSR(int m, int n, int nnz, int base = 0);

  // Following constructor will make CSR does not own the data
  CSR(int m, int n, int *rowptr, int *colidx, double *values, int base = 0);
 
  ~CSR();

  void storeMatrixMarket(const char *fileName) const;
  /**
   * Load PETSc bin format
   */
  void loadBin(const char *fileName);
  /**
   * Load PETSc bin format
   */
  void storeBin(const char *fileName) const;

  /**
   * permutes current matrix based on permutation vector "perm"
   * and return new permuted matrix
   *
   * @param sort true if we want to sort non-zeros of each row based on colidx
   */
  CSR *permute(const int *columnPerm, const int *rowInversePerm, bool sort = false) const;
  /**
   * permutes rows but not columns
   */
  CSR *permuteRows(const int *inversePerm) const;
  /**
   * just permute rowptr
   */
  CSR *permuteRowptr(const int *inversePerm) const;
  void permuteRowptr(CSR *out, const int *inversePerm) const;

  void permuteColsInPlace(const int *perm);
  void permuteInPlaceIgnoringExtptr(const int *perm);

  /**
   * assuming rowptr is permuted, permute the remaining (colidx and values)
   *
   * @param sort true if we want to sort non-zeros of each row based on colidx
   */
  void permuteMain(
    CSR *out, const int *columnPerm, const int *rowInversePerm,
    bool sort = false) const;
  void permuteRowsMain(
    CSR *out, const int *inversePerm) const;

  /**
   * Compute w = alpha*A*x + beta*y + gamma
   * where A is this matrix
   */
  void multiplyWithVector(
    double *w,
    double alpha, const double *x, double beta, const double *y, double gamma)
    const;
  /**
   * Compute w = A*x
   */
  void multiplyWithVector(double *w, const double *x) const;

  /**
   * get reverse Cuthill Mckee permutation that tends to reduce the bandwidth
   *
   * @note only works for a symmetric matrix
   *
   * @param pseudoDiameterSourceSelection true to use heurstic of using a source
   *                                      in a pseudo diameter.
   *                                      Further reduce diameter at the expense
   *                                      of more time on finding permutation.
   */
  void getRCMPermutation(int *perm, int *inversePerm, bool pseudoDiameterSourceSelection = true);
  void getBFSPermutation(int *perm, int *inversePerm);

  CSR *transpose() const;

  bool isSymmetric(bool checkValues = true, bool printFirstNonSymmetry = false) const;

  void make0BasedIndexing();
  void make1BasedIndexing();

  /**
   * Precompute idiag to speedup triangular solver or GS
   */
  void computeInverseDiag();

  void alloc(int m, int nnz, bool createSeparateDiagData = true);
  void dealloc();

  bool useMemoryPool_() const;

  int getBandwidth() const;
  bool equals(const CSR& A, bool print = false) const;
  int getNnz() const { return rowptr[m] - base; }

  void print() const;

  template<class T> T *allocate_(size_t n) const
  {
    if (useMemoryPool_()) {
      return MemoryPool::getSingleton()->allocate<T>(n);
    }
    else {
      return MALLOC(T, n);
    }
  }

private:
  bool ownData_;
}; // CSR

void generate3D27PtLaplacian(CSR *A, int nx, int ny, int nz);
void generate3D27PtLaplacian(CSR *A, int n);

void splitLU(const CSR& A, CSR *L, CSR *U);
bool getSymmetricNnzPattern(
  const CSR *A, int **symRowPtr, int **symDiagPtr, int **symExtPtr, int **symColIdx);

} // namespace SpMP

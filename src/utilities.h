
// #include <math.h>
// #include <string.h>
// #include <R.h>
// #include <Rinternals.h>
// #include <Rdefines.h>
// #include <Rmath.h>
// #include <iostream>
// #include <vector>
// #include <algorithm>

#include <RcppArmadillo.h>
#include "bigmemory/BigMatrix.h"
#include <time.h>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"

#include "biglasso_omp.h"
//#include "defines.h"

#ifndef UTILITIES_H
#define UTILITIES_H

using namespace Rcpp;
using namespace std;

double sign(double x);

double sum(double *x, int n);

// template<typename T>
// T sum(T *x, int n);

// Sum of squares of jth column of X
double sqsum(double *X, int n, int j);
 
double crossprod(double *X, double *y, int n, int j);

int sum(int *vec, int p);

double lasso(double z, double l1, double l2, double v);

double gLoss(double *r, int n);

// get X[i, j]: i-th row, j-th column element
double get_elem_bm(XPtr<BigMatrix> xpMat, double center_, double scale_, int i, int j);

//crossprod - given specific rows of X
double crossprod_bm(XPtr<BigMatrix> xpMat, double *y_, int *row_idx_, double center_, 
                    double scale_, int n_row, int j);

double crossprod_bmC(SEXP xP, double *y_, int *row_idx_, double center_, 
                     double scale_, int n_row, int j);

//crossprod_resid - given specific rows of X: separate computation
double crossprod_resid(XPtr<BigMatrix> xpMat, double *y_, double sumY_, int *row_idx_, 
                       double center_, double scale_, int n_row, int j);

// update residul vector if variable j enters eligible set
void update_resid(XPtr<BigMatrix> xpMat, double *r, double shift, int *row_idx_, 
                  double center_, double scale_, int n_row, int j);

// Sum of squares of jth column of X
double sqsum_bm(SEXP xP, int n_row, int j, int useCores);

// Weighted sum of residuals
double wsum(double *r, double *w, int n_row);

// Weighted cross product of y with jth column of x
double wcrossprod_resid(XPtr<BigMatrix> xpMat, double *y, double sumYW_, int *row_idx_, 
                        double center_, double scale_, double *w, int n_row, int j);

// Weighted sum of squares of jth column of X
// sum w_i * x_i ^2 = sum w_i * ((x_i - c) / s) ^ 2
// = 1/s^2 * (sum w_i * x_i^2 - 2 * c * sum w_i x_i + c^2 sum w_i)
double wsqsum_bm(XPtr<BigMatrix> xpMat, double *w, int *row_idx_, double center_, 
                 double scale_, int n_row, int j);

void Free_memo_hsr(double *a, double *r, int *e1, int *e2);

// standardize
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx,
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
                                  int *row_idx, double lambda_min, double alpha, int n, int p);


#endif

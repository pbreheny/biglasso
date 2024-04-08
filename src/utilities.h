
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
// sqsum for file-backed X
double sqsum_bm(XPtr<BigMatrix> xpMat, int n_row, int j);
 
double crossprod(double *X, double *y, int n, int j);

int sum(int *vec, int p);

double lasso(double z, double l1, double l2, double v);

double MCP(double z, double l1, double l2, double gamma, double v);

double SCAD(double z, double l1, double l2, double gamma, double v);

double gLoss(double *r, int n);

// get X[i, j]: i-th row, j-th column element
double get_elem_bm(XPtr<BigMatrix> xpMat, double center_, double scale_, int i, int j);

// //crossprod for big.matrix, no standardization (raw)
// double crossprod_bm_raw(XPtr<BigMatrix> xpMat, double *y, int *row_idx, int n, int j);

//crossprod - given specific rows of X
double crossprod_bm(XPtr<BigMatrix> xpMat, double *y_, int *row_idx_, double center_, 
                    double scale_, int n_row, int j);

//crossprod - for gaussian_simple()
double crossprod_bm_no_std(XPtr<BigMatrix> xpMat, double *y_, int n_row, int j);

// crossprod of columns X_j and X_k
double crossprod_bm_Xj_Xk(XPtr<BigMatrix> xMat, int *row_idx,
                          NumericVector &center, NumericVector &scale,
                          int n, int j, int k);

// double crossprod_bmC(SEXP xP, double *y_, int *row_idx_, double center_,
//                      double scale_, int n_row, int j);

//crossprod_resid - given specific rows of X: separate computation
double crossprod_resid(XPtr<BigMatrix> xpMat, double *y_, double sumY_, int *row_idx_, 
                       double center_, double scale_, int n_row, int j);

//crossprod_resid - given specific rows of *standardized* X: separate computation
// double crossprod_resid_no_std(XPtr<BigMatrix> xpMat, double *y_, double sumY_,
//                               int n_row, int j);

// update residul vector if variable j enters eligible set
void update_resid(XPtr<BigMatrix> xpMat, double *r, double shift, int *row_idx_, 
                  double center_, double scale_, int n_row, int j);


// update residual vector if variable j enters eligible set; NO standardization
void update_resid_no_std(XPtr<BigMatrix> xpMat, double *r, double shift,
                   int n_row, int j);

// update residul vector and eta vector
void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j);

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

// standardize
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx,
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
                                  int *row_idx, double alpha, int n, int p);

// get residuals 
// void get_residual(vector<double> &z, double lambda,
//                   int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
//                   double alpha, int n, int p);

// check KKT conditions over features in the inactive set
int check_inactive_set(int *e1, vector<double> &z, XPtr<BigMatrix> xpMat, int *row_idx, 
                       vector<int> &col_idx, NumericVector &center, NumericVector &scale, double *a,
                       double lambda, double sumResid, double alpha, double *r, double *m, int n, int p);

// check KKT conditions over features in the safe set
int check_safe_set(int *ever_active, int *discard_beta, vector<double> &z, 
                   XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, 
                   double *r, double *m, int n, int p);

// check KKT conditions over features in (the safe set - the strong set)
int check_rest_safe_set(int *ever_active, int *strong_set, int *discard_beta, vector<double> &z,
                        XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                        NumericVector &center, NumericVector &scale, double *a, double lambda,
                        double sumResid, double alpha, double *r, double *m, int n, int p);

// check KKT conditions over features in (the safe set - the strong set) for *standardized* X
int check_rest_safe_set_no_std(int *ever_active, int *strong_set, int *discard_beta, vector<double> &z,
                        XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                        double *a, double lambda,
                        double sumResid, double alpha, double *r, double *m, int n, int p);

// check KKT conditions over features in the strong set
int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx,
                     NumericVector &center, NumericVector &scale, double *a,
                     double lambda, double sumResid, double alpha, 
                     double *r, double *m, int n, int p);

// check KKT conditions over features in the strong set for *standardized* X
int check_strong_set_no_std(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx, double *a,
                     double lambda, double sumResid, double alpha, 
                     double *r, double *m, int n, int p);

// check KKT conditions over features in the rest set
int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, int *row_idx, 
                   vector<int> &col_idx, NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, double *r, double *m, int n, int p);

// update z[j] for features which are rejected at previous lambda but not rejected at current one.
void update_zj(vector<double> &z,
               int *bedpp_reject, int *bedpp_reject_old,
               XPtr<BigMatrix> xpMat, int *row_idx,vector<int> &col_idx,
               NumericVector &center, NumericVector &scale, 
               double sumResid, double *r, double *m, int n, int p);

#endif

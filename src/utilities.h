#ifndef UTILITIES_H
#define UTILITIES_H

#include <math.h>
#include <string.h>
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rmath.h>

//#include "defines.h"
using namespace Rcpp;

double sign(double x);

double sum(double *x, int n);

// Sum of squares of jth column of X
double sqsum(double *X, int n, int j);
 
double crossprod(double *X, double *y, int n, int j);

// count discarded features
int sum_discard(int *discards, int p);

int sum_int(int *vec, int p);

int checkConvergence(arma::sp_mat beta, double *beta_old, double eps, int l, int p);

double lasso(double z, double l1, double l2, double v);

double gLoss(double *r, int n);

// get_row
int get_row_bm(SEXP xP);

// get_col
int get_col_bm(SEXP xP);

// get X[i, j]: i-th row, j-th column element
double get_elem_bm(SEXP xP, double center_, double scale_, int i, int j);

//crossprod - given specific rows of X
double crossprod_bm(SEXP xP, double *y_, int *row_idx_, double center_, 
                    double scale_, int n_row, int j);

double crossprod_bmC(SEXP xP, double *y_, int *row_idx_, double center_, 
                     double scale_, int n_row, int j);

//crossprod_resid - given specific rows of X: separate computation
double crossprod_resid(XPtr<BigMatrix> xpMat, double *y_, double sumY_, int *row_idx_, 
                       double center_, double scale_, int n_row, int j);

// update residul vector if variable j enters eligible set
void update_resid(SEXP xP, double *r, double shift, int *row_idx_, 
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

void free_memo_hsr(double *a, double *r, double *z, int *e1, int *e2);

// standardize
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  double *z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
                                  int *row_idx, double lambda_min, double alpha, int n, int p);

// -----------------------------------------------------------------------------
// Following functions are used for logistic regression
// -----------------------------------------------------------------------------

void free_memo_bin_hsr(double *s, double *w, double *a, double *r, double *z, 
                       int *e1, int *e2, double *eta);


int check_strong_set_bin(int *e1, int *e2, double *z, XPtr<BigMatrix> xpMat, int *row_idx, NumericVector &center, NumericVector &scale,
                         double lambda, double sumResid, double alpha, double *r, double *m, int n, int p);


int check_rest_set_bin(int *e1, int *e2, double *z, XPtr<BigMatrix> xpMat, int *row_idx, NumericVector &center, NumericVector &scale,
                       double lambda, double sumResid, double alpha, double *r, double *m, int n, int p);

void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j);




// -----------------------------------------------------------------------------
// C++ functions used for EDPP rule
// -----------------------------------------------------------------------------
void update_theta(double *theta, SEXP xP, int *row_idx_, NumericVector &center, 
                  NumericVector &scale, double *y, arma::sp_mat beta, double lambda, 
                  int *nzero_beta, int n, int p, int l);

// V2 - <v1, v2> / ||v1||^2_2 * V1
void update_pv2(double *pv2, double *v1, double *v2, int n);

// apply EDPP 
void edpp_screen(int *discard_beta, SEXP xP, double *o, int *row_idx, 
                 NumericVector &center, NumericVector &scale, int n, int p, 
                 double rhs);

// apply EDPP - by chunk
void edpp_screen_by_chunk(int *discard_beta, const char *xf_bin, int nchunks, int chunk_cols, 
                          double *o, int *row_idx, NumericVector &center, 
                          NumericVector &scale, int n, int p, double rhs, int n_total);

// apply EDPP - by chunk, openmp
void edpp_screen_by_chunk_omp(int *discard_beta, const char *xf_bin, int nchunks, int chunk_cols, 
                              double *o, int *row_idx, NumericVector &center, 
                              NumericVector &scale, int n, int p, double rhs, int n_total);
// apply EDPP 
void edpp_screen2(int *discard_beta, SEXP xP, double *o, int *row_idx, 
                 NumericVector &center, NumericVector &scale, int n, int p, 
                 double rhs);

#endif

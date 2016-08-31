// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <iostream>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"
#include "bigmemory/isna.hpp"
//#include <omp.h>

//#include "defines.h"

using namespace Rcpp;
// -----------------------------------------------------------------------------
// Following functions are callled by C
// -----------------------------------------------------------------------------

// get_row
RcppExport int get_row_bm(SEXP xP) {
  XPtr<BigMatrix> xMat(xP);
  int nrow = xMat->nrow();
  return nrow;
}

// get_col
RcppExport int get_col_bm(SEXP xP) {
  XPtr<BigMatrix> xMat(xP);
  int ncol = xMat->ncol();
  return ncol;
}

// get X[i, j]: i-th row, j-th column element
RcppExport double get_elem_bm(SEXP xP, double center_, double scale_, int i, int j) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double res = (xAcc[j][i] - center_) / scale_;
  return res;
}

//crossprod - given specific rows of X
RcppExport double crossprod_bm(SEXP xP, double *y_, int *row_idx_, double center_, 
                               double scale_, int n_row, int j) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];

  double sum = 0.0;
  double sum_xy = 0.0;
  double sum_y = 0.0;
  for (int i=0; i < n_row; i++) {
    // row_idx only used by xP, not by y;
    sum_xy = sum_xy + xCol[row_idx_[i]] * y_[i];
    sum_y = sum_y + y_[i];
  }
  sum = (sum_xy - center_ * sum_y) / scale_;

  return sum;
}

//crossprod_resid - given specific rows of X: separate computation
RcppExport double crossprod_resid(SEXP xP, double *y_, double sumY_, int *row_idx_, 
                                  double center_, double scale_, int n_row, int j) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
  double sum = 0.0;
  for (int i=0; i < n_row; i++) {
    sum = sum + xCol[row_idx_[i]] * y_[i];
  }
  sum = (sum - center_ * sumY_) / scale_;
  return sum;
}

// update residul vector if variable j enters eligible set
RcppExport void update_resid(SEXP xP, double *r, double shift, int *row_idx_, 
                               double center_, double scale_, int n_row, int j) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
  for (int i=0; i < n_row; i++) {
    r[i] -= shift * (xCol[row_idx_[i]] - center_) / scale_;
  }
}

// Sum of squares of jth column of X
RcppExport double sqsum_bm(SEXP xP, int n_row, int j, int useCores) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
//   omp_set_dynamic(0);
//   omp_set_num_threads(useCores);
  //Rprintf("sqsum_bm: Number of used threads: %d\n", omp_get_num_threads());
  double val = 0.0;
  // #pragma omp parallel for reduction(+:val)
  for (int i=0; i < n_row; i++) {
    val += pow(xCol[i], 2);
//     if (i == 0) {
//       Rprintf("sqsum_bm: Number of used threads: %d\n", omp_get_num_threads());
//     }
  }
  return val;
}

// Weighted sum of residuals
RcppExport double wsum(double *r, double *w, int n_row) {
  double val = 0.0;
  for (int i = 0; i < n_row; i++) {
    val += r[i] * w[i];
  }
  return val;
}

// Weighted cross product of y with jth column of x
RcppExport double wcrossprod_resid(SEXP xP, double *y, double sumYW_, int *row_idx_, 
                                   double center_, double scale_, double *w, int n_row, int j) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
  double val = 0.0;
  for (int i = 0; i < n_row; i++) {
    val += xCol[row_idx_[i]] * y[i] * w[i];
  }
  val = (val - center_ * sumYW_) / scale_;
  
  return val;
}


// Weighted sum of squares of jth column of X
// sum w_i * x_i ^2 = sum w_i * ((x_i - c) / s) ^ 2
// = 1/s^2 * (sum w_i * x_i^2 - 2 * c * sum w_i x_i + c^2 sum w_i)
RcppExport double wsqsum_bm(SEXP xP, double *w, int *row_idx_, double center_, 
                            double scale_, int n_row, int j) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
  double val = 0.0;
  double sum_wx_sq = 0.0;
  double sum_wx = 0.0;
  double sum_w = 0.0;
  for (int i = 0; i < n_row; i++) {
    sum_wx_sq += w[i] * pow(xCol[row_idx_[i]], 2);
    sum_wx += w[i] * xCol[row_idx_[i]];
    sum_w += w[i];
  }
  val = (sum_wx_sq - 2 * center_ * sum_wx + pow(center_, 2) * sum_w) / pow(scale_, 2);
  return val;
}



// -----------------------------------------------------------------------------
// Following functions are used for EDPP rule
// -----------------------------------------------------------------------------

// theta = (y - X*beta) / lambda
//       = (y - x1*beta1 - x2*beta2 - .... - xp * betap) / lambda
RcppExport void update_theta(double *theta, SEXP xP, int *row_idx_, double *center, 
                  double *scale, double *y, double *beta, double lambda, 
                  int *nzero_beta, int n, int p, int l) {
  double temp[n];
  for (int i=0; i<n; i++) {
    temp[i] = 0;
  }
  
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  // first compute sum_xj*betaj: loop for variables with nonzero beta's.
  for (int j = 0; j < p; j++) {
    if (nzero_beta[j] != 0) {
      for (int i = 0; i < n; i++) {
        temp[i] += beta[l*p+j] * (xAcc[j][row_idx_[i]] - center[j]) / scale[j];
      }
    }
  }
  // then compute (y - sum_xj*betaj) / lambda
  for (int i = 0; i < n; i++) {
    theta[i] = (y[i] - temp[i]) / lambda;
  }
  
}

// V2 - <v1, v2> / ||v1||^2_2 * V1
RcppExport void update_pv2(double *pv2, double *v1, double *v2, int n) {

  double v1_dot_v2 = 0;
  double v1_norm = 0;
  for (int i = 0; i < n; i++) {
    v1_dot_v2 += v1[i] * v2[i];
    v1_norm += pow(v1[i], 2);
  }
  for (int i = 0; i < n; i++) {
    pv2[i] = v2[i] - v1[i] * (v1_dot_v2 / v1_norm);
  }
}

// apply EDPP 
RcppExport void edpp_screen(int *discard_beta, SEXP xP, double *o, int *row_idx, 
                            double *center, double *scale, int n, int p, 
                            double rhs) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  
  int j;
  double lhs;
  double sum_xy;
  double sum_y;
  double *xCol;
  #pragma omp parallel for private(j, lhs, sum_xy, sum_y) default(shared) schedule(static) 
  for (j = 0; j < p; j++) {
    sum_xy = 0.0;
    sum_y = 0.0;
    xCol = xAcc[j];
    for (int i=0; i < n; i++) {
      sum_xy = sum_xy + xCol[row_idx[i]] * o[i];
      sum_y = sum_y + o[i];
    }
    lhs = fabs((sum_xy - center[j] * sum_y) / scale[j]);
    if (lhs < rhs) {
      discard_beta[j] = 1;
    } else {
      discard_beta[j] = 0;
    }
  }
}

// apply EDPP 
RcppExport void edpp_screen2(int *discard_beta, SEXP xP, double *o, int *row_idx, 
                            double *center, double *scale, int n, int p, 
                            double rhs) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  
  int j;
  double lhs;
  double sum_xy;
  double sum_y;

  #pragma omp parallel for private(j, lhs, sum_xy, sum_y) default(shared) schedule(static) 
  for (j = 0; j < p; j++) {
    sum_xy = std::inner_product(xAcc[j], xAcc[j]+n, o, 0.0);
    sum_y = std::accumulate(o, o+n, 0.0);
    lhs = fabs((sum_xy - center[j] * sum_y) / scale[j]);
    if (lhs < rhs) {
      discard_beta[j] = 1;
    } else {
      discard_beta[j] = 0;
    }
  }
}


// -----------------------------------------------------------------------------
// Following functions are directly callled inside R
// -----------------------------------------------------------------------------

// standardize big matrix, return just 'center' and 'scale'
// [[Rcpp::export]]
RcppExport SEXP standardize_bm(SEXP xP, SEXP row_idx_) {
  BEGIN_RCPP
  SEXP __sexp_result;
  {
    Rcpp::RNGScope __rngScope;
    XPtr<BigMatrix> xMat(xP);
    MatrixAccessor<double> xAcc(*xMat);
    int ncol = xMat->ncol();

    NumericVector c(ncol);
    NumericVector s(ncol);
    IntegerVector row_idx(row_idx_);
    int nrow = row_idx.size();

    for (int j = 0; j < ncol; j++) {
      c[j] = 0; //center
      s[j] = 0; //scale
      for (int i = 0; i < nrow; i++) {
        c[j] += xAcc[j][row_idx[i]]; 
        s[j] += pow(xAcc[j][row_idx[i]], 2);
      }
      c[j] = c[j] / nrow;
      s[j] = sqrt(s[j] / nrow - pow(c[j], 2));
    }
    PROTECT(__sexp_result = Rcpp::List::create(c, s));
  }
  UNPROTECT(1);
  return __sexp_result;
  END_RCPP
}

// compute eta = X %*% beta. X: n-by-p; beta: p-by-l. l is length of lambda
// [[Rcpp::export]]
RcppExport SEXP get_eta(SEXP xP, SEXP row_idx_, SEXP beta, SEXP idx_p, SEXP idx_l) {
  BEGIN_RCPP
  SEXP __sexp_result;
  {
    Rcpp::RNGScope __rngScope;
    XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
    MatrixAccessor<double> xAcc(*xpMat);
   
    // sparse matrix for beta: only pass the non-zero entries and their indices;
    arma::sp_mat sp_beta = Rcpp::as<arma::sp_mat>(beta);

    IntegerVector row_idx(row_idx_);
    IntegerVector index_p(idx_p);
    IntegerVector index_l(idx_l);
   
    int n = row_idx.size();
    int l = sp_beta.n_cols;
    int nnz = index_p.size();
    
    // initialize result
    arma::sp_mat sp_res = arma::sp_mat(n, l);
  
    for (int k = 0; k < nnz; k++) {
      for (int i = 0; i < n; i++) {
        //double x = (xAcc[index_p[k]][row_idx[i]] - center[index_p[k]]) / scale[index_p[k]];
        // NOTE: beta here is unstandardized; so no need to standardize x
        double x = xAcc[index_p[k]][row_idx[i]];
        sp_res(i, index_l[k]) += x * sp_beta(index_p[k], index_l[k]);
      }
    }

    PROTECT(__sexp_result = Rcpp::wrap(sp_res));
  }
  UNPROTECT(1);
  return __sexp_result;
  END_RCPP
}

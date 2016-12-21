// #include <RcppArmadillo.h>
// #include <iostream>
// #include "bigmemory/BigMatrix.h"
// #include "bigmemory/MatrixAccessor.hpp"
// #include "bigmemory/bigmemoryDefines.h"
// #include "bigmemory/isna.hpp"
// #include <omp.h>
// #include <stdlib.h>

#include "utilities.h"

double sign(double x) {
  if(x > 0.00000000001) return 1.0;
  else if(x < -0.00000000001) return -1.0;
  else return 0.0;
}

// Cross product of y with jth column of X
double crossprod(double *X, double *y, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += X[nn+i]*y[i];
  return(val);
}

int sum(int *x, int n) {
  int sum = 0;
  for (int j = 0; j < n; j++) {
    sum += x[j];
  }
  return sum;
}

// TODO: fix bugs with template function
// template<typename T>
// T sum(T *x, int n) {
//   T result = 0;
//   for (int i = 0; i < n; i++) {
//     result = result + x[i];
//   }
//   return result;
// }

double sum(double *x, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return(sum);
}

// Sum of squares of jth column of X
double sqsum(double *X, int n, int j) {
  int nn = n*j;
  double val = 0;
  for (int i = 0; i < n; i++) val += pow(X[nn+i], 2);
  return(val);
}

double lasso(double z, double l1, double l2, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else return(s*(fabs(z)-l1)/(v*(1+l2)));
}

// Gaussian loss
double gLoss(double *r, int n) {
  double l = 0;
  for (int i=0;i<n;i++) l = l + pow(r[i],2);
  return(l);
}

// get X[i, j]: i-th row, j-th column element
double get_elem_bm(XPtr<BigMatrix> xpMat, double center_, double scale_, int i, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double res = (xAcc[j][i] - center_) / scale_;
  return res;
}

// //crossprod for big.matrix, no standardization (raw)
// double crossprod_bm_raw(XPtr<BigMatrix> xpMat, double *y, int *row_idx, int n, int j) {
//   double res = 0.0;
//   MatrixAccessor<double> xAcc(*xpMat);
//   double *xCol = xAcc[j];
//   for (int i = 0; i < n; i++) {
//     res += xCol[row_idx[i]] * y[i];
//   }
//   return res;
// }

//crossprod - given specific rows of X
double crossprod_bm(XPtr<BigMatrix> xpMat, double *y_, int *row_idx_, double center_, 
                               double scale_, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  double sum = 0.0;
  double sum_xy = 0.0;
  double sum_y = 0.0;
  for (int i=0; i < n_row; i++) {
    sum_xy = sum_xy + xCol[row_idx_[i]] * y_[i];
    sum_y = sum_y + y_[i];
  }
  sum = (sum_xy - center_ * sum_y) / scale_;

  return sum;
}

// crossprod of columns X_j and X_k
double crossprod_bm_Xj_Xk(XPtr<BigMatrix> xMat, int *row_idx,
                          NumericVector &center, NumericVector &scale,
                          int n, int j, int k) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol_j = xAcc[j];
  double *xCol_k = xAcc[k];
  double sum_xj_xk = 0.0;
  double res = 0.0;

  for (int i = 0; i < n; i++) {
    sum_xj_xk += xCol_j[row_idx[i]] * xCol_k[row_idx[i]];
  }
  res = (sum_xj_xk - n * center[j] * center[k]) / (scale[j] * scale[k]);

  return res;
}

//crossprod_resid - given specific rows of X: separate computation
double crossprod_resid(XPtr<BigMatrix> xpMat, double *y_, double sumY_, int *row_idx_, 
                                  double center_, double scale_, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
  double sum = 0.0;
  for (int i=0; i < n_row; i++) {
    sum = sum + xCol[row_idx_[i]] * y_[i];
  }
  sum = (sum - center_ * sumY_) / scale_;
  return sum;
}

// update residul vector
void update_resid(XPtr<BigMatrix> xpMat, double *r, double shift, int *row_idx_, 
                               double center_, double scale_, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  for (int i=0; i < n_row; i++) {
    r[i] -= shift * (xCol[row_idx_[i]] - center_) / scale_;
  }
}

// Sum of squares of jth column of X
double sqsum_bm(SEXP xP, int n_row, int j, int useCores) {
  XPtr<BigMatrix> xpMat(xP); //convert to big.matrix pointer;
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];

  double val = 0.0;
  // #pragma omp parallel for reduction(+:val)
  for (int i=0; i < n_row; i++) {
    val += pow(xCol[i], 2);
  }
  return val;
}

// Weighted sum of residuals
double wsum(double *r, double *w, int n_row) {
  double val = 0.0;
  for (int i = 0; i < n_row; i++) {
    val += r[i] * w[i];
  }
  return val;
}

// Weighted cross product of y with jth column of x
double wcrossprod_resid(XPtr<BigMatrix> xpMat, double *y, double sumYW_, int *row_idx_, 
                        double center_, double scale_, double *w, int n_row, int j) {
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
double wsqsum_bm(XPtr<BigMatrix> xpMat, double *w, int *row_idx_, double center_, 
                            double scale_, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  
  double val = 0.0;
  double sum_wx_sq = 0.0;
  double sum_wx = 0.0;
  double sum_w = 0.0;
  for (int i = 0; i < n_row; i++) {
    sum_wx_sq += w[i] * pow(xCol[row_idx_[i]], 2);
    sum_wx += w[i] * xCol[row_idx_[i]];
    sum_w += w[i]; // TODO: pre-compute SUM_W and
  }
  val = (sum_wx_sq - 2 * center_ * sum_wx + pow(center_, 2) * sum_w) / pow(scale_, 2);
  return val;
}

// standardize
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
                                  int *row_idx, double lambda_min, double alpha, int n, int p) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  double sum_xy, sum_y;
  double zmax = 0.0, zj = 0.0;
  int i, j;
  
  for (j = 0; j < p; j++) {
    xCol = xAcc[j];
    sum_xy = 0.0;
    sum_y = 0.0;
    
    for (i = 0; i < n; i++) {
      center[j] += xCol[row_idx[i]];
      scale[j] += pow(xCol[row_idx[i]], 2);
      
      sum_xy = sum_xy + xCol[row_idx[i]] * y[i];
      sum_y = sum_y + y[i];
    }
    
    center[j] = center[j] / n; //center
    scale[j] = sqrt(scale[j] / n - pow(center[j], 2)); //scale
    
    if (scale[j] > 1e-6) {
      col_idx.push_back(j);
      zj = (sum_xy - center[j] * sum_y) / (scale[j] * n); //residual
      if (fabs(zj) > zmax) {
        zmax = fabs(zj);
        *xmax_ptr = j; // xmax_ptr is the index in the raw xMat, not index in col_idx!
      }
      z.push_back(zj);
    }
  }
  *p_keep_ptr = col_idx.size();
  *lambda_max_ptr = zmax / alpha;
}

void Free_memo_hsr(double *a, double *r, int *e1, int *e2) {
  Free(a);
  Free(r);
  Free(e1);
  Free(e2);
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



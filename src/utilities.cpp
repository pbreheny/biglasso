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
  double s = 0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else return(s*(fabs(z)-l1)/(v*(1+l2)));
}

// MCP taken from ncvreg::ncvreg_init.c
double MCP(double z, double l1, double l2, double gamma, double v) {
  double s = 0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else if (fabs(z) <= gamma*l1*(1+l2)) return(s*(fabs(z)-l1)/(v*(1+l2-1/gamma)));
  else return(z/(v*(1+l2)));
}

// SCAD taken from ncvreg::ncvreg_init.c
double SCAD(double z, double l1, double l2, double gamma, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else if (fabs(z) <= (l1*(1+l2)+l1)) return(s*(fabs(z)-l1)/(v*(1+l2)));
  else if (fabs(z) <= gamma*l1*(1+l2)) return(s*(fabs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2)));
  else return(z/(v*(1+l2)));
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

// crossprod - cross product of y with jth column of a *standardized* X
double crossprod_bm_no_std(XPtr<BigMatrix> xpMat, double *y_, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat); // Initialize MatrixAccessor with the file-backed matrix
  double *xCol = xAcc[j];  
  double sum = 0.0;

  for (int i = 0; i < n_row; i++) {
    sum = sum + xCol[i] * y_[i];
  }
  
  return sum;
}



// cumulative difference (to use in residual calculation; see ncvreg::rawfit_gaussian
// lines 63 - 73 for reference)
// TODO: revisit this and make gaussian_simple setup easier to read...
// double cumdiff_bm(XPtr<BigMatrix> xpMat, NumericVector &a_, NumericVector &y_,
//                    int n, int p){
//   double *res = R_Calloc(n, double);
//   MatrixAccessor<double> xAcc(*xpMat);
//   
//   for (int j=0; j<p; j++) {
//     double *xCol = xAcc[j];
//     for (int i=0; i < n; i++){
//       res[i] -= xCol[i] * a_[i];
//       
//     }
//   }
//   
//   return res;
// }



// crossprod of columns X_j and X_k
double crossprod_bm_Xj_Xk(XPtr<BigMatrix> xMat, int *row_idx,
                          NumericVector &center, NumericVector &scale,
                          int n, int j, int k) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol_j = xAcc[j];
  double *xCol_k = xAcc[k];
  double sum_xj_xk = 0.0;
  // double res = 0.0;
  
  for (int i = 0; i < n; i++) {
    sum_xj_xk += xCol_j[row_idx[i]] * xCol_k[row_idx[i]];
  }
  double res = (sum_xj_xk - n * center[j] * center[k]) / (scale[j] * scale[k]);
  
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

//crossprod_resid - given specific rows of *standardized* X: separate computation
// double crossprod_resid_no_std(XPtr<BigMatrix> xpMat, double *y_, double sumY_,
//                         int n_row, int j) {
// 
//   MatrixAccessor<double> xAcc(*xpMat);
//   double *xCol = xAcc[j];
//   
//   double sum = 0.0;
//   for (int i=0; i < n_row; i++) {
//     sum = sum + xCol[i] * y_[i];
//   }
//   sum = (sum * sumY_);
//   return sum;
// }

// update residual vector
void update_resid(XPtr<BigMatrix> xpMat, double *r, double shift, int *row_idx_, 
                  double center_, double scale_, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  for (int i=0; i < n_row; i++) {
    r[i] -= shift * (xCol[row_idx_[i]] - center_) / scale_;
  }
}

// update residual vector -- no standardization
void update_resid_no_std(XPtr<BigMatrix> xpMat, double *r, double shift,
                  int n_row, int j) {

  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  for (int i=0; i < n_row; i++) {
    r[i] -= shift * xCol[i];
  }
}


// update residual vector and eta vector
void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  double si; 
  for (int i=0;i<n;i++) {
    si = shift * (xCol[row_idx_[i]] - center_) / scale_;
    r[i] -= si;
    eta[i] += si;
  }
}

// Sum of squares of jth column of X
double sqsum_bm(XPtr<BigMatrix> xpMat, int n_row, int j) {
  MatrixAccessor<double> xAcc(*xpMat); // Initialize MatrixAccessor with the file-backed matrix
  
  double *xCol = xAcc[j];
  double val = 0.0;
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

// standardize & get residual 
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, double *y, 
                                  int *row_idx, double alpha, int n, int p) {
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

// // get residual only -- need this in gaussian_simple 
// void get_residual(vector<double> &z, 
//                   double lambda, 
//                   int *xmax_ptr,
//                   XPtr<BigMatrix> xMat, 
//                   double *y, 
//                   double alpha,
//                   int n, 
//                   int p) {
//   MatrixAccessor<double> xAcc(*xMat);
//   double *xCol;
//   double sum;
//   double zmax = 0.0, zj = 0.0;
//   int i, j;
//   
//   for (j = 0; j < p; j++) {
//     xCol = xAcc[j];
//     sum = 0.0;
//     
//     for (i = 0; i < n; i++) {
//       // NB: this assumes X matrix has *already* been standardized 
//       sum = sum + xCol[i] * y[i];
//     }
//     
//     zj = (sum * sum) / (n); //residual
//     if (fabs(zj) > zmax) {
//       zmax = fabs(zj);
//       *xmax_ptr = j; // xmax_ptr is the index in the raw xMat, not index in col_idx!
//     }
//     z.push_back(zj);
//   }
//   lambda = zmax / alpha;
// }

// check KKT conditions over features in the inactive set
int check_inactive_set(int *e1, vector<double> &z, XPtr<BigMatrix> xpMat, int *row_idx, 
                       vector<int> &col_idx, NumericVector &center, NumericVector &scale, double *a,
                       double lambda, double sumResid, double alpha, double *r, double *m, int n, int p) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e1[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        e1[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check KKT conditions over features in the safe set
int check_safe_set(int *ever_active, int *discard_beta, vector<double> &z, 
                   XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, 
                   double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
  
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (ever_active[j] == 0 && discard_beta[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        ever_active[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check KKT conditions over features in (the safe set - the strong set)
int check_rest_safe_set(int *ever_active, int *strong_set, int *discard_beta, vector<double> &z,
                        XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                        NumericVector &center, NumericVector &scale, double *a, double lambda,
                        double sumResid, double alpha, double *r, double *m, int n, int p) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (strong_set[j] == 0 && discard_beta[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        ever_active[j] = strong_set[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check KKT conditions over features in (the safe set - the strong set)
int check_rest_safe_set_no_std(int *ever_active, int *strong_set, int *discard_beta, vector<double> &z,
                        XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                        double *a, double lambda,
                        double sumResid, double alpha, double *r, double *m, int n, int p) {
  if (!xpMat) return 0; // check 
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol;
  double sum = 0.0;
  double l1 = 0.0;
  double l2 = 0.0;
  int j, jj, violations = 0;
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (strong_set[j] == 0 && discard_beta[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum  * sumResid) / (n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        ever_active[j] = strong_set[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check KKT conditions over features in the strong set
int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx,
                     NumericVector &center, NumericVector &scale, double *a,
                     double lambda, double sumResid, double alpha, 
                     double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
  
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e1[j] == 0 && e2[j] == 1) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if(fabs(z[j] - a[j] * l2) > l1) {
        e1[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check KKT conditions over features in the strong set on *standardized* X 
int check_strong_set_no_std(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx, double *a,
                     double lambda, double sumResid, double alpha, 
                     double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol;
  double sum = 0.0;
  double l1 = 0.0;
  double l2 = 0.0;
  int j, jj, violations = 0;
  
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e1[j] == 0 && e2[j] == 1) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum * sumResid) / (n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if(fabs(z[j] - a[j] * l2) > l1) {
        e1[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}



// check KKT conditions over features in the rest set
int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, int *row_idx, 
                   vector<int> &col_idx, NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, double *r, double *m, int n, int p) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e2[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        e1[j] = e2[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// update z[j] for features which are rejected at previous lambda but not rejected at current one.
void update_zj(vector<double> &z,
               int *bedpp_reject, int *bedpp_reject_old,
               XPtr<BigMatrix> xpMat, int *row_idx,vector<int> &col_idx,
               NumericVector &center, NumericVector &scale, 
               double sumResid, double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum;
  int j, jj;
  
#pragma omp parallel for private(j, sum) schedule(static) 
  for (j = 0; j < p; j++) {
    if (bedpp_reject[j] == 0 && bedpp_reject_old[j] == 1) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
    }
  }
}

// compute eta = X %*% beta. X: n-by-p; beta: p-by-l. l is length of lambda
// [[Rcpp::export]]
RcppExport SEXP get_eta(SEXP xP, SEXP row_idx_, SEXP beta, SEXP idx_p, SEXP idx_l) {
  BEGIN_RCPP
    
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
  return Rcpp::wrap(sp_res);
  
  END_RCPP
}

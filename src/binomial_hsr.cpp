#include <RcppArmadillo.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"
#include <time.h>
#include <omp.h>

#include "utilities.h"

//#include "defines.h"
template<typename T>
List cdfit_binomial_hsr_cpp(XPtr<BigMatrix> xMat, SEXP y_, SEXP row_idx_, 
                            SEXP lambda_, SEXP nlambda_,
                            SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                            SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                            SEXP ncore_, SEXP warn_,
                            SEXP verbose_);

void free_memo_bin_hsr(double *s, double *w, double *a, double *r, 
                       int *e1, int *e2, double *eta) {
  free(s);
  free(w);
  free(a);
  free(r);
  free(e1);
  free(e2);
  free(eta);
}

template<typename T>
void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j) {
  
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol = xAcc[j];
  double si; 
  for (int i=0;i<n;i++) {
    si = shift * (xCol[row_idx_[i]] - center_) / scale_;
    r[i] -= si;
    eta[i] += si;
  }
}

template<typename T>
int check_strong_set_bin(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                         int *row_idx, vector<int> &col_idx,
                         NumericVector &center, NumericVector &scale,
                         double lambda, double sumResid, double alpha, 
                         double *r, double *m, int n, int p) {
  MatrixAccessor<T> xAcc(*xpMat);
  
  T *xCol;
  double sum, l1;
  int j, jj, violations = 0;
  
  #pragma omp parallel for private(j, sum, l1) reduction(+:violations) schedule(static) 
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
      if(fabs(z[j]) > l1) {
        e1[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

template<typename T>
int check_rest_set_bin(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                       int *row_idx, vector<int> &col_idx,
                       NumericVector &center, NumericVector &scale,
                       double lambda, double sumResid, double alpha, 
                       double *r, double *m, int n, int p) {
  
  MatrixAccessor<T> xAcc(*xpMat);
  T *xCol;
  double sum, l1;
  int j, jj, violations = 0;
  
  #pragma omp parallel for private(j, sum, l1) reduction(+:violations) schedule(static) 
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
      if(fabs(z[j]) > l1) {
        e1[j] = e2[j] = 1;
        violations++;
      }
    }
  }
  
  return violations;
}


// Coordinate descent for gaussian models
RcppExport SEXP cdfit_binomial_hsr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_,
                                   SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP warn_,
                                   SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  int xtype = xMat->matrix_type();
  
  switch(xtype)
  {
  case 2:
    return cdfit_binomial_hsr_cpp<short>(xMat, y_, row_idx_, lambda_,nlambda_, 
                                         lambda_min_,alpha_, user_, eps_, 
                                         max_iter_, multiplier_, dfmax_, ncore_,
                                         warn_, verbose_);
  case 4:
    return cdfit_binomial_hsr_cpp<int>(xMat, y_, row_idx_, lambda_,nlambda_, 
                                         lambda_min_,alpha_, user_, eps_, 
                                         max_iter_, multiplier_, dfmax_, ncore_,
                                         warn_, verbose_);
  case 6:
    return cdfit_binomial_hsr_cpp<float>(xMat, y_, row_idx_, lambda_,nlambda_, 
                                         lambda_min_,alpha_, user_, eps_, 
                                         max_iter_, multiplier_, dfmax_, ncore_,
                                         warn_, verbose_);
  case 8:
    return cdfit_binomial_hsr_cpp<double>(xMat, y_, row_idx_, lambda_,nlambda_, 
                                         lambda_min_,alpha_, user_, eps_, 
                                         max_iter_, multiplier_, dfmax_, ncore_,
                                         warn_, verbose_);
  default:
    throw Rcpp::exception("the type defined for big.matrix is not supported!");
  }
}

template<typename T>
List cdfit_binomial_hsr_cpp(XPtr<BigMatrix> xMat, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_,
                                   SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP warn_,
                                   SEXP verbose_) {
  // XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];

  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  NumericVector beta0(L);
  NumericVector center(p);
  NumericVector scale(p);
  
  int p_keep = 0; // keep columns whose scale > 1e-6
  int *p_keep_ptr = &p_keep;
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  // standardization
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual<T>(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6

  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }

  
  // Initialization
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration
  
  double *w = Calloc(n, double);
  double *s = Calloc(n, double); //y_i - pi_i
  double *eta = Calloc(n, double);
  int *e1 = Calloc(p, int); //ever-active set
  int *e2 = Calloc(p, int); //strong set
  int lstart = 0;
  int converged, violations;
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l; // temp index
  
  double ybar = sum(y, n)/n;
  a0 = beta0[0] = log(ybar/(1-ybar));
  double nullDev = 0;
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i]*log(ybar) - (1-y[i])*log(1-ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev;
 
  double sumS = sum(s, n); // temp result sum of s
  double sumWResid = 0.0; // temp result: sum of w * r

  if (!user) {
    // set up lambda, equally spaced on log scale
    double log_lambda_max = log(lambda_max);
    double log_lambda_min = log(lambda_min*lambda_max);
    double delta = (log_lambda_max - log_lambda_min) / (L-1);
    for (l = 0; l < L; l++) {
      lambda[l] = exp(log_lambda_max - l * delta);
    }
    Dev[0] = nullDev;
    //lstart = 1;
  } else {
    lambda = Rcpp::as<NumericVector>(lambda_);
  }

  // set up omp
  int useCores = INTEGER(ncore_)[0];
  int haveCores=omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);

  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Assign a, a0 by previous b, b0
      a0 = beta0[l-1];
      for (j = 0; j < p; j++) {
        a[j] = beta(j, l-1);
      }
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        free_memo_bin_hsr(s, w, a, r, e1, e2, eta);
        return List::create(beta0, beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
   
      // strong set
      cutoff = 2*lambda[l] - lambda[l-1];
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
      
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
   
    n_reject[l] = p - sum_int(e2, p);
    // path start
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("Solution Path start: Now time: %s\n", buff);
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              w[i] = pi * (1 - pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / w[i];
            if (y[i] == 1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1-pi);
            }
          }
          
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            free_memo_bin_hsr(s, w, a, r, e1, e2, eta);
            return List::create(beta0, beta, center, scale, lambda, Dev,
                                iter, n_reject, Rcpp::wrap(col_idx));
          }
          
          // Intercept
          xwr = crossprod(w, r, n, 0);
          xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          for (i = 0; i < n; i++) {
            si = beta0[l] - a0;
            r[i] -= si; //update r
            eta[i] += si; //update eta
          }
          // update temp result: sum of w * r, used for computing xwr;
          sumWResid = wsum(r, w, n);
         
          // Covariates
//           now = time (0);
//           strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//           Rprintf("Solve lasso start: Now time: %s\n", buff);

          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              // Calculate u, v
              xwr = wcrossprod_resid<T>(xMat, r, sumWResid, row_idx, center[jj], scale[jj], w, n, jj);
              v = wsqsum_bm<T>(xMat, w, row_idx, center[jj], scale[jj], n, jj) / n;
              u = xwr/n + v * a[j];
              
              // Update b_j
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              // Update r
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // update change of objective function
                update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + \
                  l1 * (std::abs(beta(j, l)) - std::abs(a[j]));
                if (update > max_update) max_update = update;
                
                update_resid_eta<T>(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj);
                // update temp result w * r, used for computing xwr;
                sumWResid = wsum(r, w, n);

              }
            }
          }

          // Check for convergence
          if (max_update < thresh) {
            converged = 1;
          } else {
            converged = 0;
          }
          // converged = checkConvergence(beta, a, eps, l, p);
          a0 = beta0[l];
          for (int j=0; j<p; j++) {
            a[j] = beta(j, l);
          }
          if (converged) break;
        }
//         now = time (0);
//         strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//         Rprintf("Solve lasso end: Now time: %s\n", buff);
        
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set_bin<T>(e1, e2, z, xMat, row_idx, col_idx, 
                                          center, scale, lambda[l], 
                                          sumS, alpha, s, m, n, p);
        if (violations==0) break;
        // Rprintf("\tNumber of violations in strong set: %d\n", violations);
      }
      
      // Scan for violations in rest
//       now = time (0);
//       strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//       Rprintf("Scan rest set start: %s;   ", buff);
      violations = check_rest_set_bin<T>(e1, e2, z, xMat, row_idx, col_idx,
                                      center, scale, lambda[l], 
                                      sumS, alpha, s, m, n, p);
    
//       now = time (0);
//       strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//       Rprintf("Scan rest set end: %s\n", buff);
      if (violations==0) break;
      // Rprintf("\tNumber of violations in rest set: %d\n", violations);
    }
  }

  free_memo_bin_hsr(s, w, a, r, e1, e2, eta);
  return List::create(beta0, beta, center, scale, lambda, Dev, 
                      iter, n_reject, Rcpp::wrap(col_idx));
  
}


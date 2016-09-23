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

int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                   int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale,
                   double lambda, double sumResid, double alpha, double *r, 
                   double *m, int n, int p);

int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx,
                     NumericVector &center, NumericVector &scale,
                     double lambda, double sumResid, double alpha, 
                     double *r, double *m, int n, int p);

// update z[j] for features which are rejected at previous lambda but accepted at current one.
void update_zj(vector<double> &z,
               int *bedpp_accept, int *bedpp_accept_old,
               XPtr<BigMatrix> xpMat, int *row_idx,vector<int> &col_idx,
               NumericVector &center, NumericVector &scale, 
               double sumResid, double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum;
  int j, jj;
  
  #pragma omp parallel for private(j, sum) schedule(static) 
  for (j = 0; j < p; j++) {
    if (bedpp_accept[j] && bedpp_accept_old[j] == 0) {
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

// check rest set with bedpp screening
int check_rest_set_hsr_bedpp(int *e1, int *e2, int *accept, vector<double> &z, 
                            XPtr<BigMatrix> xpMat, 
                            int *row_idx,vector<int> &col_idx,
                            NumericVector &center, 
                            NumericVector &scale, double lambda, 
                            double sumResid, double alpha, double *r, 
                            double *m, int n, int p) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1;
  int j, jj, violations = 0;
  
  #pragma omp parallel for private(j, sum, l1) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (accept[j] && e2[j] == 0) {
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

// compute X^Txmax for each x_j. 
// xj^Txmax = 1 / (sj*smax) * (sum_{i=1}^n (x[i, max]*x[i,j]) - n * cj * cmax)
void bedpp_init(vector<double>& sign_lammax_xtxmax,
               XPtr<BigMatrix> xMat, int xmax_idx, double *y, double lambda_max, 
               int *row_idx, vector<int>& col_idx, NumericVector& center, 
               NumericVector& scale, int n, int p) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol, *xCol_max;
  double sum_xjxmax, sum_xmaxTy, sign_xmaxTy;
  xCol_max = xAcc[xmax_idx];
  int i, j, jj;
  // sign of xmaxTy
  sum_xmaxTy = crossprod_bm(xMat, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
  sum_xmaxTy = sign(sum_xmaxTy);
  
  #pragma omp parallel for private(j, sum_xjxmax) schedule(static) 
  for (j = 0; j < p; j++) { // p = p_keep
    jj = col_idx[j]; // index in the raw XMat, not in col_idx;
    if (jj != xmax_idx) {
      xCol = xAcc[jj];
      sum_xjxmax = 0.0;
      for (int i = 0; i < n; i++) {
        sum_xjxmax = sum_xjxmax + xCol[row_idx[i]] * xCol_max[row_idx[i]];
      }
      sign_lammax_xtxmax[j] = sign_xmaxTy * lambda_max * (sum_xjxmax - n * center[jj] * 
        center[xmax_idx]) / (scale[jj] * scale[xmax_idx]);;
    } else {
      sign_lammax_xtxmax[j] = sign_xmaxTy * lambda_max * n;
    }
  }
}

// Basic (non-sequential) EDPP test to determine the accept set of features;
void bedpp_screen(int *accept, const vector<double>& sign_lammax_xtxmax,
                  const vector<double>& XTy, double ynorm_sq, int *row_idx, 
                  vector<int>& col_idx, double lambda, 
                  double lambda_max, int n, int p) {
  double LHS = 0.0;
  double RHS = 2 * n * lambda * lambda_max - (lambda_max - lambda) * 
    sqrt(n * ynorm_sq - pow(n * lambda_max, 2));
  int j;
  
  #pragma omp parallel for private(j, LHS) schedule(static)
  for (j = 0; j < p; j++) { // p = p_keep
    LHS = (lambda + lambda_max) * XTy[j] - (lambda_max - lambda) * sign_lammax_xtxmax[j];
    if (fabs(LHS) < RHS) {
      accept[j] = 0;
    } else {
      accept[j] = 1;
    }
  }
}

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_hsr_bedpp(SEXP X_, SEXP y_, SEXP row_idx_,  
                                        SEXP lambda_, SEXP nlambda_,
                                        SEXP lam_scale_,
                                        SEXP lambda_min_, SEXP alpha_, 
                                        SEXP user_, SEXP eps_,
                                        SEXP max_iter_, SEXP multiplier_, 
                                        SEXP dfmax_, SEXP ncore_, 
                                        SEXP safe_thresh_,
                                        SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  // int n_total = xMat->nrow(); // number of total observations
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double bedpp_thresh = REAL(safe_thresh_)[0]; // threshold for safe test

  NumericVector lambda(L);
  if (user != 0) {
    lambda = Rcpp::as<NumericVector>(lambda_);
  } 
  
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];

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
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }

  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  
  // set p = p_keep, only loop over columns whose scale > 1e-6
  p = p_keep;
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }

  double *r = Calloc(n, double);
  for (int i=0; i<n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  
  // beta
  arma::sp_mat beta = arma::sp_mat(p, L);
  double *a = Calloc(p, double); //Beta from previous iteration

  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L); // number of total rejections;
  IntegerVector n_bedpp_reject(L); 
  
  double l1, l2, cutoff, shift;
  double max_update, update, thresh; // for convergence check
  int converged, lstart = 0, violations;
  int j, jj, l; // temp index

  // lambda, equally spaced on log scale
  if (user == 0) {
    if (lam_scale) {
      // set up lambda, equally spaced on log scale
      double log_lambda_max = log(lambda_max);
      double log_lambda_min = log(lambda_min*lambda_max);
      
      double delta = (log_lambda_max - log_lambda_min) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = exp(log_lambda_max - l * delta);
      }
    } else { // equally spaced on linear scale
      double delta = (lambda_max - lambda_min*lambda_max) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = lambda_max - l * delta;
      }
    }
    // lstart = 1;
    // n_reject[0] = p; // strong rule rejects all variables at lambda_max
  } 
  loss[0] = gLoss(r,n);
  thresh = eps * loss[0];
  
  int *e1 = Calloc(p, int); // ever-active set
  int *e2 = Calloc(p, int); // strong set;

  /* Variables for BEDPP test */
  vector<double> xty;
  vector<double> sign_lammax_xtxmax;
  double ynorm_sq;
  int *bedpp_accept = Calloc(p, int);
  int *bedpp_accept_old = Calloc(p, int);
  for (j = 0; j < p; j++) { // initialize
    bedpp_accept[j] = 1;
    bedpp_accept_old[j] = 1;
  }
  int accept_size; 
  
  int bedpp; // if 0, don't perform bedpp test
  if (bedpp_thresh < 1) {
    bedpp = 1; // turn on dome
    xty.resize(p);
    sign_lammax_xtxmax.resize(p);
    
    for (j = 0; j < p; j++) {
      xty[j] = z[j] * n;
    }
    ynorm_sq = sqsum(y, n, 0);
    bedpp_init(sign_lammax_xtxmax, xMat, xmax_idx, y, lambda_max, row_idx, col_idx, center, scale, n, p);
  } else {
    bedpp = 0; // turn off bedpp test
  }

  // set up omp
  int useCores = INTEGER(ncore_)[0];
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
  
  // Path
  for (l=lstart;l<L;l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
  
    if (l != 0) {
      // Assign a by previous b
      for (j = 0; j < p; j++) {
        a[j] = beta(j, l-1);
      }
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll = l; ll < L; ll++) iter[ll] = NA_INTEGER;
        free_memo_hsr(a, r, e1, e2);
        free(bedpp_accept);
        free(bedpp_accept_old);
        return List::create(beta, center, scale, lambda, loss, iter, 
                            n_reject, n_bedpp_reject, Rcpp::wrap(col_idx));
      }
      cutoff = 2*lambda[l] - lambda[l-1];
    } else {
      cutoff = 2*lambda[l] - lambda_max;
    }
    
    if (bedpp) {
      bedpp_screen(bedpp_accept, sign_lammax_xtxmax, xty, ynorm_sq, row_idx, 
                   col_idx, lambda[l], lambda_max, n, p);
      accept_size = sum_int(bedpp_accept, p);
      n_bedpp_reject[l] = p - accept_size;
      
      // update z[j] for features which are rejected at previous lambda but accepted at current one.
      update_zj(z, bedpp_accept, bedpp_accept_old, xMat, row_idx, col_idx, 
                center, scale, sumResid, r, m, n, p);

      #pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        // update bedpp_accept_old with bedpp_accept
        bedpp_accept_old[j] = bedpp_accept[j];
        // hsr screening
        if (bedpp_accept[j] && (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]]))) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      n_bedpp_reject[l] = 0; // no dome test;
      // hsr screening over all
      #pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum_int(e2, p); // e2 set means not reject by bedpp or hsr;

    while(iter[l] < max_iter) {
      while(iter[l] < max_iter){
        while(iter[l] < max_iter) {
          iter[l]++;
          //solve lasso over ever-active set
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
              // Update beta_j
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(z[j], l1, l2, 1);
              // Update r
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // compute objective update for checking convergence
                update =  - z[j] * shift + 0.5 * (1 + l2) * (pow(beta(j, l), 2) - \
                  pow(a[j], 2)) + l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj);
                sumResid = sum(r, n); //update sum of residual
              }
            }
          }
          
          // Check for convergence
          if (max_update < thresh) {
            converged = 1;
          } else {
            converged = 0;
          }
          //converged = checkConvergence(beta, a, eps, l, p);
          
          // update a; only for ever-active set
          for (j = 0; j < p; j++) {
            a[j] = beta(j, l);
          }
          if (converged) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx,
                                      center, scale, lambda[l], sumResid, 
                                      alpha, r, m, n, p);

        if (violations == 0) break;
      }
      
      if (bedpp) {
        violations = check_rest_set_hsr_bedpp(e1, e2, bedpp_accept, z, xMat, 
                                             row_idx, col_idx,
                                             center, scale, lambda[l], 
                                             sumResid, alpha, r, m, n, p);
      } else {
        violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center,  
                                    scale, lambda[l], sumResid, alpha, r, m, n, p);
      }
      
      if (violations == 0) {
        loss[l] = gLoss(r, n);
        break;
      }
    }
    
    if (n_bedpp_reject[l] < p * bedpp_thresh) {
      bedpp = 0; // turn off bedpp for next iteration if not efficient
    }
  }
  
  free_memo_hsr(a, r, e1, e2);
  free(bedpp_accept);
  free(bedpp_accept_old);
  return List::create(beta, center, scale, lambda, loss, iter, 
                      n_reject, n_bedpp_reject, Rcpp::wrap(col_idx));
}


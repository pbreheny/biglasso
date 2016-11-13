// #include <RcppArmadillo.h>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include "bigmemory/BigMatrix.h"
// #include "bigmemory/MatrixAccessor.hpp"
// #include "bigmemory/bigmemoryDefines.h"
// #include <time.h>
// #include <omp.h>

#include "utilities.h"

// update zj
void update_zj(vector<double> &z,
               int *bedpp_accept, int *bedpp_accept_old,
               XPtr<BigMatrix> xpMat, int *row_idx,vector<int> &col_idx,
               NumericVector &center, NumericVector &scale, 
               double sumResid, double *r, double *m, int n, int p);

// check strong set
int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx,
                     NumericVector &center, NumericVector &scale,
                     double lambda, double sumResid, double alpha, 
                     double *r, double *m, int n, int p);

// check rest set
int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                   int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale,
                   double lambda, double sumResid, double alpha, double *r, 
                   double *m, int n, int p);

// check rest set with dome screening
int check_rest_set_hsr_dome(int *e1, int *e2, int *dome_accept, vector<double> &z, 
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
    if (dome_accept[j] && e2[j] == 0) {
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

// compute x^Txmax for each x_j. Used by DOME screening test
// xj^Txmax = 1 / (sj*smax) * (sum_{i=1}^n (x[i, max]*x[i,j]) - cj * sum_{i=1}^n x[i, max])
void dome_init(vector<double> &xtxmax, vector<int> &region, // region: whether xtxmax is within range of two boundary points.
               vector<double> &tt, // tt = sqrt(n - 1/n * xtxmax*xtxmax)
               XPtr<BigMatrix> xMat, int xmax_idx, double ynorm, double lambda_max,
               int *row_idx, vector<int> &col_idx, NumericVector &center,
               NumericVector &scale, int n, int p) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol, *xCol_max;
  double sum_xjxmax;
  double sum_xmax = center[xmax_idx] * n;
  double cutoff = n * lambda_max / ynorm;
  int j, jj;
  xCol_max = xAcc[xmax_idx];

  #pragma omp parallel for private(j, sum_xjxmax) schedule(static) 
  for (j = 0; j < p; j++) {
    jj = col_idx[j];
    xCol = xAcc[jj];
    sum_xjxmax = 0.0;
    for (int i = 0; i < n; i++) {
      sum_xjxmax = sum_xjxmax + xCol[row_idx[i]] * xCol_max[row_idx[i]];
    }
    xtxmax[j] = (sum_xjxmax - center[jj] * sum_xmax) / (scale[jj] * scale[xmax_idx]);
    tt[j] = sqrt(n - xtxmax[j]*xtxmax[j] / n);

    if (xtxmax[j] < - cutoff) {
      region[j] = 1;
    } else if (xtxmax[j] > cutoff) {
      region[j] = 2;
    } else {
      region[j] = 3;
    }
  }
}

// DOME test to determine the accept set of features;
void dome_screen(int *accept, const vector<double> &xtxmax, const vector<int> &region, 
                 const vector<double> &tt, const vector<double> &xty, double ynorm, double psi, 
                 double lambda, double lambda_max, int n, int p) {
  int j;
  double L, U;
  double delta_lam = lambda_max - lambda;
  double dlam_times_psi = delta_lam * psi; // temp result
  double n_time_lam = n * lambda; // temp result
  double U1 = n * lambda - sqrt(n) * ynorm * delta_lam / lambda_max;
  double L2 = -n * lambda + sqrt(n) * ynorm * delta_lam / lambda_max;

  #pragma omp parallel for private(j, L, U) schedule(static) 
  for (j = 0; j < p; j++) {
    switch (region[j]) {
    case 1: // smaller than lower cutoff
      L = - n_time_lam + delta_lam * xtxmax[j] + dlam_times_psi * tt[j];
      U = U1;
      break;
    case 2: // larger than upper cutoff
      L = L2;
      U = n_time_lam + delta_lam * xtxmax[j] - dlam_times_psi * tt[j];
      break;
    case 3: // in between
      L = - n_time_lam + delta_lam * xtxmax[j] + dlam_times_psi * tt[j];
      U = n_time_lam + delta_lam * xtxmax[j] - dlam_times_psi * tt[j];
      break;
    default:
      L = 0;
      U = 0;
      break;
    }
    if (xty[j] > L && xty[j] < U) { // reject
      accept[j] = 0;
    } else {
      accept[j] = 1;
    }
  }
}

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_hsr_dome(SEXP X_, SEXP y_, SEXP row_idx_,  
                                        SEXP lambda_, SEXP nlambda_,
                                        SEXP lam_scale_,
                                        SEXP lambda_min_, SEXP alpha_, 
                                        SEXP user_, SEXP eps_,
                                        SEXP max_iter_, SEXP multiplier_, 
                                        SEXP dfmax_, SEXP ncore_, 
                                        SEXP dome_thresh_,
                                        SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double dome_thresh = REAL(dome_thresh_)[0]; // threshold for dome test
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  
  NumericVector lambda(L);
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
  
  // set up omp
  int useCores = INTEGER(ncore_)[0];
#ifdef BIGLASSO_OMP_H_
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }

  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); // beta
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L); // number of total rejections by dome test and strong rules combined;
  IntegerVector n_dome_reject(L); // number of rejections by dome test;
  
  double l1, l2, cutoff, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  int *e1 = Calloc(p, int); // ever-active set
  int *e2 = Calloc(p, int); // strong set;
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r, n);
  thresh = eps * loss[0] / n;
  
  // set up lambda
  if (user == 0) {
    if (lam_scale) { // set up lambda, equally spaced on log scale
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
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  } 

  // Variables for Dome test
  vector<double> xty;
  vector<double> xtxmax;
  vector<double> tt;
  vector<int> region;
  double ynorm = 0, psi = 0;
  int *dome_accept = Calloc(p, int);
  int *dome_accept_old = Calloc(p, int);
  for (j = 0; j < p; j++) {
    dome_accept[j] = 1;
    dome_accept_old[j] = 1;
  }
  int accept_size, dome;
  if (dome_thresh < 1) {
    dome = 1; // turn on dome
    xty.resize(p);
    xtxmax.resize(p);
    tt.resize(p);
    region.resize(p);
    for (j = 0; j < p; j++) {
      xty[j] = z[j] * n;
    }
    ynorm = sqrt(sqsum(y, n, 0));
    psi = sqrt(ynorm * ynorm / (lambda_max * lambda_max) - n);
    dome_init(xtxmax, region, tt, xMat, xmax_idx, ynorm, lambda_max, 
              row_idx, col_idx, center, scale, n, p);
  } else {
    dome = 0; // turn off dome test
  }

  if (dome == 1 && user == 0) n_dome_reject[0] = p;
  
  // Path
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll = l; ll < L; ll++) iter[ll] = NA_INTEGER;
        Free_memo_hsr(a, r, e1, e2);
        Free(dome_accept);
        Free(dome_accept_old);
        return List::create(beta, center, scale, lambda, loss, iter, 
                            n_reject, n_dome_reject, Rcpp::wrap(col_idx));
      }
      cutoff = 2*lambda[l] - lambda[l-1];
    } else {
      cutoff = 2*lambda[l] - lambda_max;
    }
    
    /*
    first apply dome test, get subset of features not rejected;
    then in that subset, apply strong rules
    */
    if (dome) {
      dome_screen(dome_accept, xtxmax, region, tt, xty, ynorm, psi, lambda[l], lambda_max, n, p);
      accept_size = sum(dome_accept, p);
      n_dome_reject[l] = p - accept_size;
      
      // update z[j] for features which are rejected at previous lambda but accepted at current one.
      update_zj(z, dome_accept, dome_accept_old, xMat, row_idx, col_idx, 
                center, scale, sumResid, r, m, n, p);
      
      #pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        // update dome_accept_old with dome_accept
        dome_accept_old[j] = dome_accept[j];
        // hsr screening over dome_accept
        if (dome_accept[j] && (fabs(z[j]) >= cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      n_dome_reject[l] = 0; // no dome test;
      // hsr screening over all
      #pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) >= cutoff * alpha * m[col_idx[j]]) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p); // e2 set means not reject by dome or hsr;

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
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(z[j], l1, l2, 1);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // compute objective update for convergence check
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2);
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumResid = sum(r, n); //update sum of residual
                a[j] = beta(j, l); // update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx,
                                      center, scale, lambda[l], sumResid, 
                                      alpha, r, m, n, p);
        if (violations == 0) break;
      }

      if (dome) {
        violations = check_rest_set_hsr_dome(e1, e2, dome_accept, z, xMat, 
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
    
    if (n_dome_reject[l] <= p * dome_thresh) {
      dome = 0; // turn off dome for next iteration if not efficient
    }
  }
  
  Free_memo_hsr(a, r, e1, e2);
  Free(dome_accept);
  Free(dome_accept_old);
  return List::create(beta, center, scale, lambda, loss, iter, 
                      n_reject, n_dome_reject, Rcpp::wrap(col_idx));
}


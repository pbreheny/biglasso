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

void Free_memo_bin_hsr(double *s, double *w, double *a, double *r,
                       int *e1, int *e2, double *eta);

void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j);

int check_strong_set_bin(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                         int *row_idx, vector<int> &col_idx,
                         NumericVector &center, NumericVector &scale,
                         double lambda, double sumResid, double alpha, 
                         double *r, double *m, int n, int p);

int check_rest_set_bin(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                       int *row_idx, vector<int> &col_idx,
                       NumericVector &center, NumericVector &scale,
                       double lambda, double sumResid, double alpha, 
                       double *r, double *m, int n, int p);

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_binomial_hsr_approx(SEXP X_, SEXP y_, SEXP row_idx_, 
                                          SEXP lambda_, SEXP nlambda_,
                                          SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                          SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                          SEXP ncore_, SEXP warn_,
                                          SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
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
  // set p = p_keep, only loop over columns whose scale > 1e-6
  p = p_keep;

  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration
  double *w = Calloc(n, double);
  double *s = Calloc(n, double); //y_i - pi_i
  double *eta = Calloc(n, double);
  int *e1 = Calloc(p, int); //ever-active set
  int *e2 = Calloc(p, int); //strong set
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart; // temp index

  double ybar = sum(y, n)/n;
  a0 = beta0[0] = log(ybar/(1-ybar));
  double nullDev = 0;
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i] * log(ybar) - (1 - y[i]) * log(1 - ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev; // threshold for convergence
 
  double sumS; // temp result sum of s
  double sumResid; // temp result sum of current residuals
  // double sumWResid = 0.0; // temp result: sum of w * r
  
  if (user == 0) {
    // set up lambda, equally spaced on log scale
    double log_lambda_max = log(lambda_max);
    double log_lambda_min = log(lambda_min*lambda_max);
    double delta = (log_lambda_max - log_lambda_min) / (L-1);
    for (l = 0; l < L; l++) {
      lambda[l] = exp(log_lambda_max - l * delta);
    }
    Dev[0] = nullDev;
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
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
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        Free_memo_bin_hsr(s, w, a, r, e1, e2, eta);
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
      for (int j=0; j<p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p);

    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              //w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              //w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              //w[i] = pi*(1-pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / 0.25;
            // r[i] = s[i]/w[i];
            if (y[i]==1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1 - pi);
            }
          }
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll = l; ll < L; ll++) iter[ll] = NA_INTEGER;
            Free_memo_bin_hsr(s, w, a, r, e1, e2, eta);
            return List::create(beta0, beta, center, scale, lambda, Dev, 
                                iter, n_reject, Rcpp::wrap(col_idx));
          }
          // Intercept
          xwr = 0.25 * sum(r, n);
          xwx = 0.25 * n;
//           xwr = crossprod(w, r, n, 0);
//           xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          si = beta0[l] - a0;
          if (si != 0) {
            a0 = beta0[l];
            for (i = 0; i < n; i++) {
              r[i] -= si; //update r
              eta[i] += si; //update eta
            }
          }
          sumResid = sum(r, n); // update temp result: sum of r, used for computing xwr;
          // sumWResid = wsum(r, w, n);
          
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              xwr = 0.25 * crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj);
              v = 0.25; // x^T * W * x / n = w = 0.25
              // xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[j], scale[j], w, n, j);
              // v = wsqsum_bm(xMat, w, row_idx, center[j], scale[j], n, j) / n;
              u = xwr / n + v * a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1 - alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // update change of objective function
                // update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + l1 * (fabs(beta(j, l)) - fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2) * v;
                if (update > max_update) max_update = update;
                update_resid_eta(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumResid = sum(r, n); // update temp result w * r, used for computing xwr;
                a[j] = beta(j, l); // update a
              }
            }
          }
         
          // Check for convergence
          if (max_update < thresh) break;
        }
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set_bin(e1, e2, z, xMat, row_idx, col_idx, 
                                          center, scale, lambda[l], 
                                          sumS, alpha, s, m, n, p);
        if (violations==0) break;
      }
      // Scan for violations in rest
      violations = check_rest_set_bin(e1, e2, z, xMat, row_idx, col_idx,
                                      center, scale, lambda[l], 
                                      sumS, alpha, s, m, n, p);
      if (violations==0) break;
    }
  }

  Free_memo_bin_hsr(s, w, a, r, e1, e2, eta);
  return List::create(beta0, beta, center, scale, lambda, Dev, 
                      iter, n_reject, Rcpp::wrap(col_idx));
}


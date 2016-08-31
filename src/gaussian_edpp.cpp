#include <RcppArmadillo.h>
#include <iostream>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"
#include <time.h>
#include <omp.h>

#include "utilities.h"
//#include "defines.h"
using namespace std;

void free_memo_edpp(double *a, double *r, double *z, int *nzero_beta, int *discard_beta,
               double *theta, double *v1, double *v2, double *o) {
  free(a);
  free(r);
  free(z);
  free(nzero_beta);
  free(discard_beta);
  free(theta);
  free(v1);
  free(v2);
  free(o);
}

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_edpp(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                    SEXP nlambda_, SEXP lam_scale_,
                                    SEXP lambda_min_, SEXP alpha_, 
                                    SEXP user_, SEXP eps_, SEXP max_iter_, 
                                    SEXP multiplier_, SEXP dfmax_, SEXP ncore_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  // const char *xf_bin = CHAR(Rf_asChar(xf_bin_));
  // int nchunks = INTEGER(nchunks_)[0];
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  // int n_total = xMat->nrow(); // number of total observations
  int lam_scale = INTEGER(lam_scale_)[0];
  int L = INTEGER(nlambda_)[0];
  int user = INTEGER(user_)[0];
  // int chunk_cols = p / nchunks;
  
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
  double *z = Calloc(p, double);
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;

//   char buff1[100];
//   time_t now1 = time (0);
//   strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
//   Rprintf("\nPreprocessing start: %s\n", buff1);
  // standardize: get center, scale; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
//   now1 = time (0);
//   strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
//   Rprintf("Preprocessing end: %s\n", buff1);
//   Rprintf("\n-----------------------------------------------\n");
  
  // lambda, equally spaced on log scale
  if (user == 0) {
    if (lam_scale) {
      // set up lambda, equally spaced on log scale
      double log_lambda_max = log(lambda_max);
      double log_lambda_min = log(lambda_min*lambda_max);
      
      double delta = (log_lambda_max - log_lambda_min) / (L-1);
      for (int l = 0; l < L; l++) {
        lambda[l] = exp(log_lambda_max - l * delta);
      }
    } else { // equally spaced on linear scale
      double delta = (lambda_max - lambda_min*lambda_max) / (L-1);
      for (int l = 0; l < L; l++) {
        lambda[l] = lambda_max - l * delta;
      }
    }
  } 
  
//   if (user == 0) {
//     // set up lambda, equally spaced on log scale
//     double log_lambda_max = log(lambda_max);
//     double log_lambda_min = log(lambda_min*lambda_max);
//     double delta = (log_lambda_max - log_lambda_min) / (L-1);
//     for (int l = 0; l < L; l++) {
//       lambda[l] = exp(log_lambda_max - l * delta);
//     }
//   }
  
  double *r = Calloc(n, double);
  for (int i=0; i<n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
 
  // beta
  arma::sp_mat beta = arma::sp_mat(p, L);
  double *a = Calloc(p, double); //Beta from previous iteration
//   for (int j = 0; j < p; j++) {
//     a[j] = 0.0;
//   }

  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector discard_count(L);
  
  double l1, l2;
  int converged;
 
  // EDPP
  double *theta = Calloc(n, double);
  double *v1 = Calloc(n, double);
  double *v2 = Calloc(n, double);
  double *pv2 = Calloc(n, double);
  double *o = Calloc(n, double);
  
  // index set of nonzero beta's at l+1;
  int *nzero_beta = Calloc(p, int);
  // index set of discarded features at l+1;
  int *discard_beta = Calloc(p, int);
  // number of discarded features at each lambda
//   int *discard_count = Calloc(L, int);
//   for (int l = 0; l < L; l++) discard_count[l] = 0;
  double pv2_norm = 0;
 
  // set up omp
  int useCores = INTEGER(ncore_)[0];
  // Rprintf("Number of requsted threads: %d\n", useCores);
  int haveCores=omp_get_num_procs();
  //Rprintf("Number of avaialbe processors: %d\n", haveCores);
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);

  for (int l = 0; l < L-1; l++) {
//     char buff[100];
//     time_t now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("Lambda %d. Now time: %s\n", l, buff);
   
    if (l != 0 ) {
      // Check dfmax: a is current beta. At each iteration, solving for next beta.
      int nv = 0;
      for (int j=0; j<p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        free_memo_edpp(a, r, z, nzero_beta, discard_beta, theta, v1, v2, o);
        return List::create(beta, center, scale, lambda, loss, iter, discard_count);
      }
      // update theta
      update_theta(theta, X_, row_idx, center, scale, y, beta, lambda[l], nzero_beta, n, p, l);
      // update v1
      for (int i=0; i < n; i++) {
        v1[i] = y[i] / lambda[l] - theta[i];
        //if (i == 0) Rprintf("v1[0] = %f, theta[0] = %f\n", v1[i], theta[0]);
      }
    } else { // lambda_max = lam[0]
      for (int i = 0; i < n; i++) {
        theta[i] =  y[i] / lambda[l];
      }
      // compute v1 for lambda_max
      double xty = crossprod_bm(X_, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
      for (int i = 0; i < n; i++) {
        v1[i] = sign(xty) * get_elem_bm(X_, center[xmax_idx], scale[xmax_idx], row_idx[i], xmax_idx);
      }
      loss[l] = gLoss(r,n);
      
      for (int j=0; j<p; j++) {
        nzero_beta[j] = 0;
        discard_beta[j] = 1;
      }
      discard_count[l] = sum_discard(discard_beta, p);
//       Rprintf("lambda[%d] = %f: discarded features: %d\n", l, lambda[l], discard_count[l]);
    } 
    // update v2:
    for (int i = 0; i < n; i++) {
      v2[i] = y[i] / lambda[l+1] - theta[i];
    }
    //update pv2:
    update_pv2(pv2, v1, v2, n);
    // update norm of pv2;
    for (int i = 0; i < n; i++) {
      pv2_norm += pow(pv2[i], 2);
    }
    pv2_norm = pow(pv2_norm, 0.5);
    // Rprintf("\npv2_norm = %f\n", pv2_norm);
    // update o
    for (int i = 0; i < n; i++) {
      o[i] = theta[i] + 0.5 * pv2[i];
    }
    double rhs = n - 0.5 * pv2_norm * sqrt(n); 
    
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("EDPP start: Now time: %s\n", buff);

    // apply EDPP
//     if (nchunks > 1) {   // chunk-reading or memory mapping for EDPP?
//       edpp_screen_by_chunk_omp(discard_beta, xf_bin, nchunks, chunk_cols, o, row_idx, center,
//                            scale, n, p, rhs, n_total);
//     } else {
//       edpp_screen(discard_beta, X_, o, row_idx, center, scale, n, p, rhs);
//     }
    edpp_screen(discard_beta, X_, o, row_idx, center, scale, n, p, rhs);
    discard_count[l+1] = sum_discard(discard_beta, p);
    
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("EDPP end: Now time: %s\n", buff);
//     
//     // count discarded features for next lambda
//     discard_count[l+1] = sum_discard(discard_beta, p);
//     Rprintf("lambda[%d] = %f: discarded features: %d\n", l+1, lambda[l+1], discard_count[l+1]);
//     
//     // solve lasso for beta[lam(l+1)] based on the set of non-discarded features.
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("Solve lasso start: Now time: %s\n", buff);
    double shift = 0.0;
    while (iter[l+1] < max_iter) {
      iter[l+1]++;
      for (int j=0; j<p; j++) {
        if (discard_beta[j] == 0) {
          z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[j], scale[j], n, j) / n + a[j];
          // Update beta_j
          l1 = lambda[l+1] * m[j] * alpha;
          l2 = lambda[l+1] * m[j] * (1-alpha);
          beta(j, l+1) = lasso(z[j], l1, l2, 1);
          // Update r
          shift = beta(j, l+1) - a[j];
          if (shift !=0) {
            update_resid(X_, r, shift, row_idx, center[j], scale[j], n, j);
            sumResid = sum(r, n); //update sum of residual
          }
          // update non-zero beta set
          if (beta(j, l+1) != 0) {
            nzero_beta[j] = 1;
          } else {
            nzero_beta[j] = 0;
          }
        }
      }
      // Check for convergence
      converged = checkConvergence(beta, a, eps, l+1, p);
      // update a
      for (int j = 0; j < p; j++) {
        a[j] = beta(j, l+1);
      }
      if (converged) {
        loss[l+1] = gLoss(r, n);
        break;
      } 
    }
    
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("Solve lasso end: Now time: %s\n", buff);
  }

  free_memo_edpp(a, r, z, nzero_beta, discard_beta, theta, v1, v2, o);
  return List::create(beta, center, scale, lambda, loss, iter, discard_count);
}


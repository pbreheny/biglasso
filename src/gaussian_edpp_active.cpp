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

// check edpp set
int check_edpp_set(int *ever_active, int *discard_beta, vector<double> &z, 
                   XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale,
                   double lambda, double sumResid, double alpha, 
                   double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1;
  int j, jj, violations = 0;
  
  #pragma omp parallel for private(j, sum, l1) reduction(+:violations) schedule(static) 
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
      if(fabs(z[j]) > l1) {
        ever_active[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

void free_memo_edpp_active(double *a, double *r, int *ever_active, int *nzero_beta, 
                           int *discard_beta, double *theta, double *v1, double *v2, 
                           double *o) {
  free(a);
  free(r);
  free(ever_active);
  free(nzero_beta);
  free(discard_beta);
  free(theta);
  free(v1);
  free(v2);
  free(o);
}

// theta = (y - X*beta) / lambda
//       = (y - x1*beta1 - x2*beta2 - .... - xp * betap) / lambda
void update_theta(double *theta, XPtr<BigMatrix> xpMat, int *row_idx, 
                  vector<int> &col_idx, NumericVector &center, 
                  NumericVector &scale, double *y, arma::sp_mat beta, double lambda, 
                  int *nzero_beta, int n, int p, int l);
// V2 - <v1, v2> / ||v1||^2_2 * V1
void update_pv2(double *pv2, double *v1, double *v2, int n);

// apply EDPP 
void edpp_screen(int *discard_beta, XPtr<BigMatrix> xpMat, double *o, 
                 int *row_idx, vector<int> &col_idx,
                 NumericVector &center, NumericVector &scale, int n, int p, 
                 double rhs);

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_edpp_active(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                    SEXP nlambda_, SEXP lam_scale_,
                                    SEXP lambda_min_, SEXP alpha_, 
                                    SEXP user_, SEXP eps_, SEXP max_iter_, 
                                    SEXP multiplier_, SEXP dfmax_, SEXP ncore_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int lam_scale = INTEGER(lam_scale_)[0];
  int L = INTEGER(nlambda_)[0];
  int user = INTEGER(user_)[0];
  
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
  int p_keep = 0;
  int *p_keep_ptr = &p_keep;
  
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, 
                               lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  // set p = p_keep, only loop over columns whose scale > 1e-6
  p = p_keep;
  
  // beta
  arma::sp_mat beta = arma::sp_mat(p, L);
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector discard_count(L);
  
  double l1, l2;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l; //temp index
  
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r, n);
  thresh = eps * loss[0] / n;
  
  // EDPP
  double *theta = Calloc(n, double);
  double *v1 = Calloc(n, double);
  double *v2 = Calloc(n, double);
  double *pv2 = Calloc(n, double);
  double *o = Calloc(n, double);
  
  // ever-active set
  int *ever_active = Calloc(p, int);
  // index set of nonzero beta's at l+1;
  int *nzero_beta = Calloc(p, int);
  // index set of discarded features at l+1;
  int *discard_beta = Calloc(p, int);
  double pv2_norm = 0;
  int violations;
  
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
  } 
  
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
  
  for (l = 0; l < L-1; l++) {
    if (l != 0 ) {
      // Check dfmax: a is current beta. At each iteration, solving for next beta.
      int nv = 0;
      for (int j=0; j<p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        free_memo_edpp_active(a, r, ever_active, nzero_beta, discard_beta, theta, v1, v2, o);
        return List::create(beta, center, scale, lambda, loss, iter, 
                            discard_count, Rcpp::wrap(col_idx));
      }
      // update theta
      update_theta(theta, xMat, row_idx, col_idx, center, scale, y, beta, lambda[l], nzero_beta, n, p, l);
      // update v1
      for (i = 0; i < n; i++) {
        v1[i] = y[i] / lambda[l] - theta[i];
      }
    } else { // lambda_max = lam[0]
      for (i = 0; i < n; i++) {
        theta[i] =  y[i] / lambda[l];
      }
      // compute v1 for lambda_max
      double xty = crossprod_bm(xMat, y, row_idx, center[xmax_idx], 
                                scale[xmax_idx], n, xmax_idx);
      for (i = 0; i < n; i++) {
        v1[i] = sign(xty) * get_elem_bm(xMat, center[xmax_idx], 
                     scale[xmax_idx], row_idx[i], xmax_idx);
      }
      // loss[l] = gLoss(r,n);
      for (j = 0; j < p; j++) {
        // nzero_beta[j] = 0;
        discard_beta[j] = 1;
      }
      discard_count[l] = sum_int(discard_beta, p);
    } 
    // update v2:
    for (i = 0; i < n; i++) {
      v2[i] = y[i] / lambda[l+1] - theta[i];
    }
    //update pv2:
    update_pv2(pv2, v1, v2, n);
    // update norm of pv2;
    for (i = 0; i < n; i++) {
      pv2_norm += pow(pv2[i], 2);
    }
    pv2_norm = pow(pv2_norm, 0.5);
    // update o
    for (i = 0; i < n; i++) {
      o[i] = theta[i] + 0.5 * pv2[i];
    }
    double rhs = n - 0.5 * pv2_norm * sqrt(n); 
    
    // apply EDPP
    edpp_screen(discard_beta, xMat, o, row_idx, col_idx, center, scale, n, p, rhs);
    discard_count[l+1] = sum_int(discard_beta, p);
    
    double shift = 0.0;
    while (iter[l+1] < max_iter) {
      while (iter[l+1] < max_iter) {
        iter[l+1]++;
        max_update = 0.0;
        for (int j=0; j<p; j++) {
          if (ever_active[j] == 1) {
            jj = col_idx[j];
            z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
            // Update beta_j
            l1 = lambda[l+1] * m[jj] * alpha;
            l2 = lambda[l+1] * m[jj] * (1-alpha);
            beta(j, l+1) = lasso(z[j], l1, l2, 1);
            // Update r
            shift = beta(j, l+1) - a[j];
            if (shift !=0) {
              // compute objective update for checking convergence
              //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l+1), 2) - \
              //  pow(a[j], 2)) - l1 * (fabs(beta(j, l+1)) -  fabs(a[j]));
              update = pow(beta(j, l+1) - a[j], 2);
              if (update > max_update) {
                max_update = update;
              }
              update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj);
              sumResid = sum(r, n); //update sum of residual
              a[j] = beta(j, l+1); //update a
            }
            // update non-zero beta and ever active sets
            if (beta(j, l+1) != 0) {
              nzero_beta[j] = ever_active[j] = 1;
            } else {
              nzero_beta[j] = 0;
            }
          }
        }
        // Check for convergence
        // converged = checkConvergence(beta, a, eps, l+1, p);
        if (max_update < thresh) break;
      }
      
      // Scan for violations in edpp set
      violations = check_edpp_set(ever_active, discard_beta, z, xMat, row_idx, 
                                  col_idx, center, scale, lambda[l+1],sumResid, alpha, r, m, n, p); 
      if (violations==0) {
        loss[l+1] = gLoss(r, n);
        break;
      }
    }
  }
  
  free_memo_edpp_active(a, r, ever_active, nzero_beta, discard_beta, theta, v1, v2, o);
  return List::create(beta, center, scale, lambda, loss, iter, 
                      discard_count, Rcpp::wrap(col_idx));
}


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
SEXP cdfit_gaussian_edpp_cpp(XPtr<BigMatrix> xMat,
                             SEXP y_, SEXP row_idx_, SEXP lambda_, 
                             SEXP nlambda_, SEXP lam_scale_,
                             SEXP lambda_min_, SEXP alpha_, 
                             SEXP user_, SEXP eps_, SEXP max_iter_, 
                             SEXP multiplier_, SEXP dfmax_, 
                             SEXP ncore_, SEXP verbose_);

void free_memo_edpp(double *a, double *r, int *nzero_beta, int *discard_beta,
               double *theta, double *v1, double *v2, double *o) {
  free(a);
  free(r);
  free(nzero_beta);
  free(discard_beta);
  free(theta);
  free(v1);
  free(v2);
  free(o);
}

// theta = (y - X*beta) / lambda
//       = (y - x1*beta1 - x2*beta2 - .... - xp * betap) / lambda
template<typename T>
void update_theta(double *theta, XPtr<BigMatrix> xpMat, int *row_idx, 
                  vector<int> &col_idx, NumericVector &center, 
                  NumericVector &scale, double *y, const arma::sp_mat& beta, double lambda, 
                  int *nzero_beta, int n, int p, int l) {
  // Rprintf("\nStart updating theta: \n");
  int i, j, jj;
  T *xCol;
  double temp[n];
  for (i = 0; i < n; i++) {
    temp[i] = 0;
  }

  MatrixAccessor<T> xAcc(*xpMat);
  // first compute sum_xj*betaj: loop for variables with nonzero beta's.
  for (j = 0; j < p; j++) {
    if (nzero_beta[j] != 0) {
      jj = col_idx[j];
      // Rprintf("\tjj = %d;", jj);
      xCol = xAcc[jj];
      // Rprintf("\n\n");
      // if (j < 10) {
      //   Rprintf("beta(%d, %d) = %f; ", j, l, beta(j, l));
      // }

      for (i = 0; i < n; i++) {
        temp[i] += beta(j, l) * (xCol[row_idx[i]] - center[jj]) / scale[jj];
        // if (i < 10) Rprintf("temp[%d]=%f; ", i, temp[i]);
      }
    }
  }
  // then compute (y - sum_xj*betaj) / lambda
  for (i = 0; i < n; i++) {
    theta[i] = (y[i] - temp[i]) / lambda;
    // if (i < 10) {
    //   Rprintf("\ttheta[%d] = %f; ", i, theta[i]);
    // }
  }
  // Rprintf("\nEnd updating theta: \n");
}


// V2 - <v1, v2> / ||v1||^2_2 * V1
void update_pv2(double *pv2, double *v1, double *v2, int n) {
  
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
template<typename T>
void edpp_screen(int *discard_beta, XPtr<BigMatrix> xpMat, double *o, 
                 int *row_idx, vector<int> &col_idx,
                 NumericVector &center, NumericVector &scale, int n, int p, 
                 double rhs) {
  MatrixAccessor<T> xAcc(*xpMat);
  
  int j, jj;
  double lhs, sum_xy, sum_y;
  T *xCol;
  
  #pragma omp parallel for private(j, lhs, sum_xy, sum_y) default(shared) schedule(static) 
  for (j = 0; j < p; j++) {
    sum_xy = 0.0;
    sum_y = 0.0;
    
    jj = col_idx[j];
    xCol = xAcc[jj];
    for (int i=0; i < n; i++) {
      sum_xy = sum_xy + xCol[row_idx[i]] * o[i];
      sum_y = sum_y + o[i];
    }
    lhs = fabs((sum_xy - center[jj] * sum_y) / scale[jj]);
    if (lhs < rhs) {
      discard_beta[j] = 1;
    } else {
      discard_beta[j] = 0;
    }
  }
}

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_edpp(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                    SEXP nlambda_, SEXP lam_scale_,
                                    SEXP lambda_min_, SEXP alpha_, 
                                    SEXP user_, SEXP eps_, SEXP max_iter_, 
                                    SEXP multiplier_, SEXP dfmax_, 
                                    SEXP ncore_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  int xtype = xMat->matrix_type();
  
  switch(xtype)
  {
  case 2:
    return cdfit_gaussian_edpp_cpp<short>(xMat, y_, row_idx_, lambda_,nlambda_, 
                                         lam_scale_, lambda_min_,alpha_, 
                                         user_, eps_, max_iter_, multiplier_, 
                                         dfmax_, ncore_, verbose_);
  case 4:
    return cdfit_gaussian_edpp_cpp<int>(xMat, 
                                       y_, row_idx_, lambda_,nlambda_, 
                                       lam_scale_, lambda_min_,alpha_, 
                                       user_, eps_, max_iter_, multiplier_, 
                                       dfmax_, ncore_, verbose_);
  case 6:
    return cdfit_gaussian_edpp_cpp<float>(xMat,
                                         y_, row_idx_, lambda_,nlambda_, 
                                         lam_scale_, lambda_min_,alpha_, 
                                         user_, eps_, max_iter_, multiplier_, 
                                         dfmax_, ncore_, verbose_);
  case 8:
    return cdfit_gaussian_edpp_cpp<double>(xMat,
                                          y_, row_idx_, lambda_,nlambda_, 
                                          lam_scale_, lambda_min_,alpha_, 
                                          user_, eps_, max_iter_, multiplier_, 
                                          dfmax_, ncore_, verbose_);
  default:
    throw Rcpp::exception("the type defined for big.matrix is not supported!");
  }
}

template<typename T>  
SEXP cdfit_gaussian_edpp_cpp(XPtr<BigMatrix> xMat,
                             SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                    SEXP nlambda_, SEXP lam_scale_,
                                    SEXP lambda_min_, SEXP alpha_, 
                                    SEXP user_, SEXP eps_, SEXP max_iter_, 
                                    SEXP multiplier_, SEXP dfmax_, 
                                    SEXP ncore_, SEXP verbose_) {
  // XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int lam_scale = INTEGER(lam_scale_)[0];
  int L = INTEGER(nlambda_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  
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

//   char buff1[100];
//   time_t now1 = time (0);
//   strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
//   Rprintf("\nPreprocessing start: %s\n", buff1);
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual<T>(center, scale, p_keep_ptr, col_idx, z, 
                               lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  
  // set p = p_keep, only loop over columns whose scale > 1e-6
  p = p_keep;
    
    //now1 = time (0);
//   strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
//   Rprintf("Preprocessing end: %s\n", buff1);
//   Rprintf("\n-----------------------------------------------\n");

  double l1, l2;
  int converged;
  int i, j, jj, l; //temp index
  double max_update, update, thresh; // for convergence check

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

  // beta
  arma::sp_mat beta = arma::sp_mat(p, L);
  Rprintf("beta(100, 1) = %f; cols = %d, rows = %d\n\n", beta(99, 1), 
          beta.n_cols, beta.n_rows);
  
  double *a = Calloc(p, double); //Beta from previous iteration

  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector discard_count(L);
  
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r, n);
  thresh = eps * loss[0]; // threshold for convergence of CD
 
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
  double pv2_norm = 0;
 
  // set up omp
  int useCores = INTEGER(ncore_)[0];
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);

  for (l = 0; l < L-1; l++) {
//     char buff[100];
//     time_t now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("Lambda %d. Now time: %s\n", l, buff);
    // Rprintf("\n*********l = %d *********\n\n", l);
    if (l != 0 ) {
      // Check dfmax: a is current beta. At each iteration, solving for next beta.
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        free_memo_edpp(a, r, nzero_beta, discard_beta, theta, v1, v2, o);
        return List::create(beta, center, scale, lambda, loss, iter, 
                            discard_count, Rcpp::wrap(col_idx));
      }
      
      // Rprintf("Update theta: \n");
      // // update theta
      // Rprintf("theta[0] = %f; col_idx[0] = %d; beta[0] = %f\n",
      //             theta[0], col_idx[0], beta(0, l));
      // 
      // for (j = 0; j < 10; j++) {
      //   Rprintf("\t beta(%d, %d) = %f; ", j, l, beta(j, l));
      // }
      // Rprintf("\n\n");
      
      update_theta<T>(theta, xMat, row_idx, col_idx, center, scale, y, beta, lambda[l], nzero_beta, n, p, l);
      
      // for (j = 0; j < 10; j++) {
      //   Rprintf("\t beta(%d, %d) = %f; ", j, l, beta(j, l));
      // }
      // Rprintf("\n\n");
     
      // update v1
      for (i = 0; i < n; i++) {
        v1[i] = y[i] / lambda[l] - theta[i];
        // if (i == 0) Rprintf("v1[0] = %f, theta[0] = %f\n", v1[i], theta[0]);
      }
    } else { // lambda_max = lam[0]
      for (i = 0; i < n; i++) {
        theta[i] =  y[i] / lambda[l];
      }
      // compute v1 for lambda_max
      double xty = crossprod_bm<T>(xMat, y, row_idx, center[xmax_idx], 
                                   scale[xmax_idx], n, xmax_idx);
      for (i = 0; i < n; i++) {
        double temp = get_elem_bm<T>(xMat, center[xmax_idx], scale[xmax_idx], row_idx[i], xmax_idx);
        v1[i] = sign(xty) * temp;
      }
      // loss[l] = gLoss(r,n);
      for (j = 0; j < p; j++) {
        nzero_beta[j] = 0;
        discard_beta[j] = 1;
      }
      discard_count[l] = sum_int(discard_beta, p);
//       Rprintf("lambda[%d] = %f: discarded features: %d\n", l, lambda[l], discard_count[l]);
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
    Rprintf("pv2_norm = %f\n", pv2_norm);
    
   // update o
    for (i = 0; i < n; i++) {
      o[i] = theta[i] + 0.5 * pv2[i];
    }
    double rhs = n - 0.5 * pv2_norm * sqrt(n); 
    Rprintf("rhs = %f\n", rhs);
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("EDPP start: Now time: %s\n", buff);

    // apply EDPP
    Rprintf("apply EDPP:\n");
    edpp_screen<T>(discard_beta, xMat, o, row_idx, col_idx, center, scale, n, p, rhs);
    discard_count[l+1] = sum_int(discard_beta, p);
    Rprintf("discard_count[%d] = %d\n", l+1, discard_count[l+1]);
    
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
      
      //solve lasso over ever-active set
      max_update = 0.0;
      for (j = 0; j < p; j++) {
        if (discard_beta[j] == 0) {
          jj = col_idx[jj];
          z[j] = crossprod_resid<T>(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
          // Update beta_j
          l1 = lambda[l+1] * m[jj] * alpha;
          l2 = lambda[l+1] * m[jj] * (1-alpha);
          beta(j, l+1) = lasso(z[j], l1, l2, 1);
          // Update r
          shift = beta(j, l+1) - a[j];
          if (shift != 0) {
            // compute objective update for checking convergence
            update =  - z[j] * shift + 0.5 * (1 + l2) * (pow(beta(j, l+1), 2) - \
              pow(a[j], 2)) + l1 * (fabs(beta(j, l+1)) -  fabs(a[j]));
            if (update > max_update) {
              max_update = update;
            }
            update_resid<T>(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj);
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
      if (max_update < thresh) {
        converged = 1;
      } else {
        converged = 0;
      }
      //converged = checkConvergence(beta, a, eps, l+1, p);
      // update a
      for (j = 0; j < p; j++) {
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

  free_memo_edpp(a, r, nzero_beta, discard_beta, theta, v1, v2, o);
  return List::create(beta, center, scale, lambda, loss, iter, 
                      discard_count, Rcpp::wrap(col_idx));
}


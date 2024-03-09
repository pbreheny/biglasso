#include "utilities.h"

// T. Peter's addition ---------------------------

// Coordinate descent for gaussian models -- NO adapting or SSR 
// NOTE: in this simple function, lambda is a SINGLE VALUE, not a path!! 
RcppExport SEXP cdfit_gaussian_simple(SEXP X_, SEXP y_, SEXP row_idx_, 
                                      SEXP lambda_, SEXP alpha_,
                                      SEXP eps_, SEXP max_iter_,
                                      SEXP multiplier_, 
                                      SEXP ncore_, 
                                      SEXP verbose_) {
  
  Rf_PrintValue(X_);
  // for debugging 
  Rprintf("Entering cdfit_gaussian_simple");
  
  // declarations 
  XPtr<BigMatrix> xMat(X_);
  // if (!Rcpp::is<NumericMatrix>(X_)) {
  //   Rcpp::stop("X_ is not a numeric matrix");
  // }

  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int iter;
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  // int dfmax = INTEGER(dfmax_)[0];
  // double update_thresh = REAL(update_thresh_)[0];
  double lambda_val = REAL(lambda_)[0];
  double *lambda = &lambda_val;
  vector<int> col_idx;
  vector<double> z; //vector to hold residuals; to be filled in by get_residual()
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
  
  // Get residual
  get_residual(col_idx, z, lambda, xmax_ptr, xMat, y, row_idx, alpha, n, p);
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  

  // objects to be returned to R 
  vector<double> beta; // vector to hold estimated coefficients from current iteration
  double *a = R_Calloc(p, double); //Beta from previous iteration
  if(a == NULL){
    Rprintf("Problem: 'a' is NULL");
  }
  
  double l1, l2, shift;
  double max_update, update, thresh, loss, n_reject; // for convergence check
  int i, j, jj; //temp index
  int *ever_active = R_Calloc(p, int); // ever-active set
  if(ever_active == NULL){
    Rprintf("Problem: 'ever active' is NULL");
  }
  
  int *discard_beta = R_Calloc(p, int); // index set of discarded features;
  if(discard_beta == NULL){
    Rprintf("Problem: 'discard_beta' is NULL");
  }
  double *r = R_Calloc(n, double);
  if(r == NULL){
    Rprintf("Problem: 'r' is NULL");
  }
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss = gLoss(r, n);
  thresh = eps * loss / n;
  
  Rprintf("Made it to the loop");
  
  while(iter < max_iter) {
    while (iter < max_iter) {
      R_CheckUserInterrupt();
      while (iter < max_iter) {
        iter++;
        max_update = 0.0;
        for (j = 0; j < p; j++) {
          if (ever_active[j]) {
            Rprintf("Made it to active set");
            jj = col_idx[j];
            z[j] = crossprod_resid_no_std(xMat, r, sumResid, row_idx, n, jj)/n + a[j];
            l1 = *lambda * m[jj] * alpha;
            l2 = *lambda * m[jj] * (1-alpha);
            beta[j] = lasso(z[j], l1, l2, 1);
            
            shift = beta[j] - a[j];
            if (shift != 0) {
              update = pow(beta[j] - a[j], 2);
              if (update > max_update) {
                max_update = update;
              }
              update_resid_no_std(xMat, r, shift, row_idx, n, jj);
              sumResid = sum(r, n); //update sum of residual
              a[j] = beta[j]; //update a
            }
            // update ever active sets
            if (beta[j] != 0) {
              ever_active[j] = 1;
            } 
          }
        }
        // Check for convergence 
        if (max_update < thresh) break;
        
      }
    }
    
  }
  
  R_Free(ever_active);
  R_Free(r); 
  R_Free(a); 
  R_Free(discard_beta); 
  return List::create(beta, *lambda, loss, iter, z, n_reject,
                      Rcpp::wrap(col_idx));
}




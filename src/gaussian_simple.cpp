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
  
  
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  double alpha = REAL(alpha_)[0];
  int *row_idx = INTEGER(row_idx_);
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = 1; // again, note that here lambda is a *single* value 
  // int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int iter = 0;
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  //int dfmax = INTEGER(dfmax_)[0];
  // double update_thresh = REAL(update_thresh_)[0];
  double lambda = REAL(lambda_)[0];
  
  // the 'p' here won't need to change, since constant features are assumed 
  //  to have been already removed 
  int p_keep = p; 
  int *p_keep_ptr = &p_keep;
  
  vector<int> col_idx;
  vector<double> z;//vector to hold residuals; to be filled in by get_residual()
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
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  
  // Get residual
  get_residual(p_keep_ptr, col_idx, z, lambda, xmax_ptr, xMat, y,
               row_idx, alpha, n, p);
  
  // for debugging 
  // Rprintf("The lambda value is: ",lambda);
  
  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); //Beta
  double *a = R_Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector n_reject(L);
  IntegerVector n_safe_reject(L);
  
  double l1, l2, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, violations; //temp index
  int *ever_active = R_Calloc(p, int); // ever-active set
  int *strong_set = R_Calloc(p, int); // strong set
  int *discard_beta = R_Calloc(p, int); // index set of discarded features;
  double cutoff = 0; // cutoff for strong rule
  //int *discard_old = R_Calloc(p, int);
  double *r = R_Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r, n);
  thresh = eps * loss[0] / n;
  
  // here, loop thru residuals and determine which variables (features) are in 
  //  the 'strong set' -- i.e., which variables are most likely to be selected 
  //  in some candidate model 
  for(j = 0; j < p; j++) {
    if(discard_beta[j]) continue;
    if(fabs(z[j]) > cutoff * alpha * m[col_idx[j]]) {
      strong_set[j] = 1;
    } else {
      strong_set[j] = 0;
    }
  }
  n_reject = p - sum(strong_set, p);
  
  while(iter < max_iter) {
    while (iter < max_iter) {
      //R_CheckUserInterrupt();
      while (iter < max_iter) {
        iter++;
        max_update = 0.0;
        for (j = 0; j < p; j++) {
          if (ever_active[j]) {
            jj = col_idx[j];
            z[j] = crossprod_resid_no_std(xMat, r, sumResid, row_idx, n, jj) / n + a[j];
            l1 = lambda * m[jj] * alpha;
            l2 = lambda * m[jj] * (1-alpha);
            beta[j] = lasso(z[j], l1, l2, 1);
            
            shift = beta[j] - a[j];
            if (shift != 0) {
              update = pow(beta[j] - a[j], 2);
              if (update > max_update) {
                max_update = update;
              }
              update_resid_no_std(xMat, r, shift, row_idx, n, jj);
              sumResid = sum(r, n); //update sum of residual
              a[j] = beta(j); //update a
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
      violations = check_strong_set_no_std(ever_active, strong_set, z, xMat,
                                           row_idx, col_idx, a, lambda,
                                           sumResid, alpha, r, m, n, p); 
      if (violations==0) break;
    }	
    // Scan for violations in edpp set
    violations = check_rest_safe_set_no_std(ever_active, strong_set, discard_beta, z, xMat, row_idx, col_idx, a, lambda, sumResid, alpha, r, m, n, p); 
    if (violations == 0) {
      loss = gLoss(r, n);
      break;
    }
    
  }
  
  R_Free(ever_active); R_Free(r); R_Free(a); R_Free(discard_beta); R_Free(strong_set); 
  return List::create(beta, lambda, loss, iter, z, n_reject, n_safe_reject, Rcpp::wrap(col_idx));
}




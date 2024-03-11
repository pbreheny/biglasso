#include "utilities.h"

// T. Peter's version ---------------------------

// Coordinate descent for gaussian models -- NO adapting or SSR 

// NOTE: in this simple function, lambda is a SINGLE VALUE, not a path!! 
// NOTE: this function does NOT implement any standardization of X
// NOTE: this function does NOT center y
RcppExport SEXP cdfit_gaussian_simple(SEXP X_,
                                      SEXP y_,
                                      SEXP row_idx_, 
                                      SEXP r_,
                                      SEXP init_, 
                                      SEXP xtx_,
                                      SEXP lambda_,
                                      SEXP alpha_,
                                      SEXP eps_,
                                      SEXP max_iter_,
                                      SEXP multiplier_, 
                                      SEXP ncore_, 
                                      SEXP verbose_) {

  // for debugging 
  Rprintf("\nEntering cdfit_gaussian_simple");
  
  // declarations 
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double *r = REAL(r_);// vector to hold residuals 
  double *init = REAL(init_);
  double *xtx = REAL(xtx_);
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int iter;
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  double lambda_val = REAL(lambda_)[0];
  double *lambda = &lambda_val;
  vector<int> col_idx;
  vector<double> z;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  NumericVector beta(p);
  // TODO: decide if the below would be a better way to set up beta
  // double *beta = R_Calloc(p, double); // vector to hold estimated coefficients from current iteration
  double *a = R_Calloc(p, double); // will hold beta from previous iteration
  double l1, l2, shift;
  double max_update, update, thresh, loss; // for convergence check
  int i, j, jj; //temp indices
  int *ever_active = R_Calloc(p, int); // ever-active set
  int *discard_beta = R_Calloc(p, int); // index set of discarded features;
  
  
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
 
 // set up init
 for (int j=0; j<p; j++) 
   a[j]=init[j];
 
 // TODO: add this step to set up r in case it is NA 
 if (ISNA(r[0])) {
   MatrixAccessor<double> xAcc(*xMat);
   for (int i=0; i<n; i++)
     r[i] = y[i];
   for (int j=0; j<p; j++) {
     double *xCol = xAcc[j];
     for (int i=0; i<n; i++) {
       r[i] -= xCol[i]*a[j];
     }
   }
 }
 
 Rprintf("\nGetting initial residual value");
  // get residual
  get_residual(col_idx, z, lambda, xmax_ptr, xMat, y, row_idx, alpha, n, p);
  
  
  // calculate gaussian loss 
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss = gLoss(r, n);
  thresh = eps * loss / n;
  
  Rprintf("\nMade it to the loop");
  
  while(iter < max_iter) {
    while (iter < max_iter) {
      R_CheckUserInterrupt();
      while (iter < max_iter) {
        iter++;
        max_update = 0.0;
        for (j = 0; j < p; j++) {
          if (ever_active[j]) {
            Rprintf("\nMade it to active set");
            jj = col_idx[j];
            z[j] = crossprod_resid_no_std(xMat, r, sumResid, row_idx, n, jj)/n + xtx[j]*a[j];
            l1 = *lambda * m[jj] * alpha;
            l2 = *lambda * m[jj] * (1-alpha);
            
            // TODO: add SCAD and MCP to the below
            beta[j] = lasso(z[j], l1, l2, xtx[j]);
            
            shift = beta[j] - a[j];
            // TODO: check the update below against the update in 
            // ncvreg::rawfit_gaussian.cpp lines 104-107
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
  
  Rprintf("\nAbout to return the list; class of r (for residuals) is: ");
  
  
  R_Free(ever_active);
  R_Free(a); 
  R_Free(discard_beta); 
  
  // return list of 4 items: 
  // - beta: numeric (p x 1) vector of estimated coefficients at the supplied lambda value
  // - loss: double capturing the loss at this lambda value with these coefs.
  // - iter: integer capturing the number of iterations needed in the coordinate descent
  // - r: numeric (n x 1) vector of residuals 
  return List::create(beta, loss, iter);
}




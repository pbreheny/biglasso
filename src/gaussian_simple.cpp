#include "utilities.h"

// T. Peter's version ---------------------------

// Coordinate descent for gaussian models -- NO adapting or SSR 

// NOTE: in this simple function, lambda is a SINGLE VALUE, not a path!! 
// NOTE: this function does NOT implement any standardization of X
// NOTE: this function does NOT center y
RcppExport SEXP cdfit_gaussian_simple(SEXP X_,
                                      SEXP y_,
                                      SEXP r_,
                                      SEXP init_, 
                                      SEXP xtx_,
                                      SEXP lambda_,
                                      SEXP alpha_,
                                      SEXP eps_,
                                      SEXP max_iter_,
                                      SEXP multiplier_, 
                                      SEXP ncore_) {
  
  // for debugging 
  Rprintf("\nEntering cdfit_gaussian_simple");
  
  // declarations: input
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  double *r = REAL(r_);// vector to hold residuals 
  double *init = REAL(init_);
  double *xtx = REAL(xtx_);
  double alpha = REAL(alpha_)[0];
  double lambda = REAL(lambda_)[0];
  int n = xMat->nrow(); // number of observations used for fitting model
  int p = xMat->ncol();
  double eps = REAL(eps_)[0];
  int iter = 0;
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  NumericVector z(p); 
  // int xmax_idx = 0;
  // int *xmax_ptr = &xmax_idx;
  
  Rprintf("\nDeclaring beta");
  // declarations: output 
  NumericVector b(p); // Initialize a NumericVector of size p vector to hold estimated coefficients from current iteration
  double *a =  R_Calloc(p, double);// will hold beta from previous iteration
  double l1, l2, shift, cp;
  double max_update, update, thresh, loss; // for convergence check
  int i, j; //temp indices
  int *ever_active = R_Calloc(p, int); // ever-active set
  
  // set up some initial values
  for (int j=0; j<p; j++) {
    a[j]=REAL(init_)[j];
    ever_active[j] = 1*(a[j] != 0);
    b[j] = 0;
    z[j] = 0;
  }
  
  Rprintf("\na[0]: %f\n", a[0]);
  Rprintf("xtx[0]: %f\n", xtx[0]);
  
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
  
  // calculate gaussian loss 
  for (i = 0; i < n; i++) {
    r[i] = REAL(r_)[i];
  }
  Rprintf("\nr[0]: %f", r[0]);
  double sdy = sqrt(gLoss(r, n)/n);
  thresh = eps * sdy;
  
  Rprintf("\nMade it to the loop");
  // Rprintf("\niter: %d, max_iter: %d", iter, max_iter);
  
  // while(iter < max_iter) {
  while (iter < max_iter) {
    R_CheckUserInterrupt();
    while (iter < max_iter) {
      iter++;
      max_update = 0.0;
      
      // solve over active set 
      for (j = 0; j < p; j++) {
        if (ever_active[j]) {
          Rprintf("\nSolving over active set");
          cp = crossprod_bm_no_std(xMat, r, n, j);
          z[j] = cp/n + xtx[j]*a[j];
          Rprintf("\n xtx*a: %f", xtx[j]*a[j]);
          Rprintf("\n z[%d] value is: %f", j, z[j]);
          
          // update beta
          l1 = lambda * m[j] * alpha;
          l2 = lambda * m[j] * (1-alpha);
          b[j] = lasso(z[j], l1, l2, xtx[j]); // TODO: add SCAD and MCP options
          
          Rprintf("\nCurrent beta estimate is: %f", b[j]);
          // update residuals 
          shift = b[j] - a[j];
          
          if (shift != 0) {
            update_resid_no_std(xMat, r, shift, n, j);
            
            // update = pow(b[j] - a[j], 2); 
            update = fabs(shift) * sqrt(xtx[j]);
            if (update > max_update) {
              max_update = update;
            }
            
          }
        }
      }
      // make current beta the old value 
      for(int j=0; j<p; j++)
        a[j] = b[j]; 
      
      // check for convergence 
      if (max_update < thresh) break;
    }
    
    // scan for violations 
    int violations = 0;
    Rprintf("\nMade it to the violation loop");
    for (int j=0; j<p; j++) {
      if (!ever_active[j]) {
        Rprintf("\nUpdating inactive set");
        Rprintf("\nCalling crossprod_bm_no_std");
        
        z[j] = crossprod_bm_no_std(xMat, r,  n, j)/n;
        
        
        Rprintf("\nFirst value of r is: %f", r[1]);
        Rprintf("\nValue of z[%d] is: %f", z[j]);
        
        // update beta
        l1 = lambda * m[j] * alpha;
        l2 = lambda * m[j] * (1-alpha);
        // TODO: add SCAD and MCP to the below
        Rprintf("\nCalling lasso");
        b[j] = lasso(z[j], l1, l2, xtx[j]);
        
        
        // if something enters, update active set and residuals
        if (b[j] != 0) {
          ever_active[j] = 1;
          Rprintf("\nCalling update_resid_no_std");
          update_resid_no_std(xMat, r, b[j], n, j);
          a[j] = b[j];
          violations++;
        }
        
        
      }
    }
    if (violations==0) break;
  }
  
  
  // }
  
  Rprintf("\nAbout to return the list");
  
  // cleanup steps
  R_Free(a); 
  R_Free(ever_active);
  // TODO: sort out how to use the line below 
  // REAL(loss)[0] = gLoss(r, n);
  
  // return list: 
  // - beta: numeric (p x 1) vector of estimated coefficients at the supplied lambda value
  // - loss: double capturing the loss at this lambda value with these coefs.
  // - iter: integer capturing the number of iterations needed in the coordinate descent
  // - r: numeric (n x 1) vector of residuals // TODO: add this! 
  return List::create(b, loss, iter);
}




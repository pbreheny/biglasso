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
                                      SEXP penalty_,
                                      SEXP lambda_,
                                      SEXP alpha_,
                                      SEXP gamma_,
                                      SEXP eps_,
                                      SEXP max_iter_,
                                      SEXP multiplier_, 
                                      SEXP ncore_) {
  
  // declarations: input
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  double *init = REAL(init_);
  double *xtx = REAL(xtx_);
  double alpha = REAL(alpha_)[0];
  double gamma = REAL(gamma_)[0];
  double lambda = REAL(lambda_)[0];
  const char *penalty = CHAR(STRING_ELT(penalty_, 0));
  int n = xMat->nrow(); // number of observations used for fitting model
  int p = xMat->ncol();
  double eps = REAL(eps_)[0];
  int iter = 0;
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  NumericVector z(p); 

  // declarations: output 
  NumericVector b(p); // Initialize a NumericVector of size p vector to hold estimated coefficients from current iteration
  double *a =  R_Calloc(p, double);// will hold beta from previous iteration
  NumericVector resid(n);
  double *r = REAL(resid); // pointer for the COPY of residuals (will modify/update these...)
  double l1, l2, shift, cp;
  double max_update, update, thresh, loss; // for convergence check
  int i, j; //temp indices
  int *ever_active = R_Calloc(p, int); // ever-active set
  
  // set up some initial values
  for (int j=0; j<p; j++) {
    a[j]=init[j];
    ever_active[j] = 1*(a[j] != 0);
    b[j] = 0;
    z[j] = 0;
  }
  
  for (i = 0; i < n; i++) {
    r[i] = REAL(r_)[i]; // again, we're making a copy of the residuals for our function to edit
  }
  
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
  
  // calculate threshold value
  double sdy = sqrt(gLoss(y, n)/n);
  thresh = eps * sdy;
  
  while (iter < max_iter) {
    R_CheckUserInterrupt();
    while (iter < max_iter) {
      iter++;
      max_update = 0.0;
      
      // solve over active set 
      for (j = 0; j < p; j++) {
        if (ever_active[j]) {
          cp = crossprod_bm_no_std(xMat, r, n, j);
          z[j] = cp/n + xtx[j]*a[j];
          
          // update beta
          l1 = lambda * m[j] * alpha;
          l2 = lambda * m[j] * (1-alpha);
          if (strcmp(penalty,"MCP")==0) b[j] = MCP(z[j], l1, l2, gamma, xtx[j]);
          if (strcmp(penalty,"SCAD")==0) b[j] = SCAD(z[j], l1, l2, gamma, xtx[j]);
          if (strcmp(penalty,"lasso")==0) b[j] = lasso(z[j], l1, l2, xtx[j]);
          
          // update residuals 
          shift = b[j] - a[j];
          
          if (shift != 0) {
            update_resid_no_std(xMat, r, shift, n, j);
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
    for (int j=0; j<p; j++) {
      if (!ever_active[j]) {
        z[j] = crossprod_bm_no_std(xMat, r,  n, j)/n;
  
        // update beta
        l1 = lambda * m[j] * alpha;
        l2 = lambda * m[j] * (1-alpha);
     if (strcmp(penalty,"MCP")==0) b[j] = MCP(z[j], l1, l2, gamma, xtx[j]);
     if (strcmp(penalty,"SCAD")==0) b[j] = SCAD(z[j], l1, l2, gamma, xtx[j]);
     if (strcmp(penalty,"lasso")==0) b[j] = lasso(z[j], l1, l2, xtx[j]);
  
        // if something enters, update active set and residuals
        if (b[j] != 0) {
          ever_active[j] = 1;
          update_resid_no_std(xMat, r, b[j], n, j);
          a[j] = b[j];
          violations++;
        }
      }
    }
    if (violations==0) break;
  }

  
  // calculate loss
  loss = gLoss(r, n);
  
  // cleanup steps
  R_Free(a); 
  R_Free(ever_active);
  
  // return list: 
  // - b: numeric (p x 1) vector of estimated coefficients at the supplied lambda value
  // - loss: double capturing the loss at this lambda value with these coefs.
  // - iter: integer capturing the number of iterations needed in the coordinate descent
  // - resid: numeric (n x 1) vector of residuals 
  return List::create(b, loss, iter, resid);
}


// Coordinate descent for gaussian models -- NO adapting or SSR 
// NOTE: in this simple function, lambda is a user-supplied path! 
//  This function does not set up a path on its own. 
// NOTE: this function does NOT implement any standardization of X
// NOTE: this function does NOT center y
RcppExport SEXP cdfit_gaussian_simple_path(SEXP X_,
                                           SEXP y_,
                                           SEXP r_,
                                           SEXP init_, 
                                           SEXP xtx_,
                                           SEXP penalty_,
                                           SEXP lambda_,
                                           SEXP nlambda_, 
                                           SEXP alpha_,
                                           SEXP gamma_,
                                           SEXP eps_,
                                           SEXP max_iter_,
                                           SEXP multiplier_, 
                                           SEXP ncore_) {
  
  // declarations: input
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  double *init = REAL(init_);
  double *xtx = REAL(xtx_);
  double alpha = REAL(alpha_)[0];
  double gamma = REAL(gamma_)[0];
  int L = INTEGER(nlambda_)[0];
  NumericVector lambda(L);
  int n = xMat->nrow(); // number of observations used for fitting model
  int p = xMat->ncol();
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  const char *penalty = CHAR(STRING_ELT(penalty_, 0));
  NumericVector z(p); 
  

  // declarations: output 
  arma::sp_mat beta = arma::sp_mat(p, L); // beta matrix to return (rows are features, columns are lambda values)
  double *a =  R_Calloc(p, double);// will hold beta from previous iteration
  NumericVector resid(n); // create vector of residuals that will be returned 
  double *r = REAL(resid); // pointer for the COPY of residuals (will modify/update these...)
  NumericVector loss(L);
  IntegerVector iter(L);
  double l1, l2, shift, cp;
  double max_update, update, thresh; // for convergence check
  int i, j, l, lstart; //temp indices
  int *ever_active = R_Calloc(p, int); // ever-active set
  lstart = 0;
  lambda = Rcpp::as<NumericVector>(lambda_);
  
  // set up some initial values
  for (int j=0; j<p; j++) {
    a[j]=init[j];
    ever_active[j] = 1*(a[j] != 0);
    z[j] = 0;
  }
  
  for (i = 0; i < n; i++) {
    r[i] = REAL(r_)[i]; // again, note that we're making a copy of the residuals for our function to edit
  }
  
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
  
  // calculate threshold value 
  double sdy = sqrt(gLoss(y, n)/n);
  thresh = eps * sdy;

  for (l = lstart; l < L; l++) {
    while (iter[l] < max_iter) {
      R_CheckUserInterrupt();
      while (iter[l] < max_iter) {
        iter[l]++;
        max_update = 0.0;
        
        // solve over active set 
        for (j = 0; j < p; j++) {
          if (ever_active[j]) {
            cp = crossprod_bm_no_std(xMat, r, n, j);
            z[j] = cp/n + xtx[j]*a[j];
            
            // update beta
            l1 = lambda[l] * m[j] * alpha;
            l2 = lambda[l] * m[j] * (1-alpha);
            if (strcmp(penalty,"MCP")==0) beta(j,l) = MCP(z[j], l1, l2, gamma, xtx[j]);
            if (strcmp(penalty,"SCAD")==0) beta(j,l) = SCAD(z[j], l1, l2, gamma, xtx[j]);
            if (strcmp(penalty,"lasso")==0) beta(j,l) = lasso(z[j], l1, l2, xtx[j]);

            // update residuals 
            shift = beta(j, l) - a[j];
            
            if (shift != 0) {
              update_resid_no_std(xMat, r, shift, n, j);
              
              update = fabs(shift) * sqrt(xtx[j]);
              if (update > max_update) {
                max_update = update;
              }
              
            }
          }
        }
        // make current beta the old value 
        for(int j=0; j<p; j++)
          a[j] = beta(j,l); 
        
        // check for convergence 
        if (max_update < thresh) break;
      }
      
      // scan for violations 
      int violations = 0;
      for (int j=0; j<p; j++) {
        if (!ever_active[j]) {

          z[j] = crossprod_bm_no_std(xMat, r,  n, j)/n;
    
          // update beta
          l1 = lambda[l] * m[j] * alpha;
          l2 = lambda[l] * m[j] * (1-alpha);
          if (strcmp(penalty,"MCP")==0) beta(j,l) = MCP(z[j], l1, l2, gamma, xtx[j]);
          if (strcmp(penalty,"SCAD")==0) beta(j,l) = SCAD(z[j], l1, l2, gamma, xtx[j]);
          if (strcmp(penalty,"lasso")==0) beta(j,l) = lasso(z[j], l1, l2, xtx[j]);
          
          // if something enters, update active set and residuals
          if (beta(j,l) != 0) {
            ever_active[j] = 1;
            update_resid_no_std(xMat, r, beta(j,l), n, j);
            a[j] = beta(j,l);
            violations++;
          }
        }
      }
      if (violations==0) break;
    }
    // calculate loss 
    loss[l] = gLoss(r, n);
  }
  
  // cleanup steps
  R_Free(a); 
  R_Free(ever_active);
  
  // return list: 
  // - beta: numeric (p x 1) vector of estimated coefficients at the supplied lambda value
  // - loss: double capturing the loss at this lambda value with these coefs.
  // - iter: integer capturing the number of iterations needed in the coordinate descent
  // - resid: numeric (n x 1) vector of residuals 
  return List::create(beta, loss, iter, resid);
}



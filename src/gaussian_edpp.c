
#include <time.h>
#include <omp.h>

#include "utilities.h"

// Memory handling, output formatting (Gaussian)
SEXP cleanup(double *a, double *r, double *z, int *nzero_beta, int *discard_beta,
             double *theta, double *v1, double *v2, double *o,
             SEXP beta, SEXP loss, SEXP iter) {
  free(a);
  free(r);
  free(z);
  free(nzero_beta);
  free(discard_beta);
  free(theta);
  free(v1);
  free(v2);
  free(o);
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 3));
  SET_VECTOR_ELT(res, 0, beta);
  SET_VECTOR_ELT(res, 1, loss);
  SET_VECTOR_ELT(res, 2, iter);
  UNPROTECT(4);
  return(res);
}

// Coordinate descent for gaussian models
SEXP cdfit_gaussian_edpp(SEXP X_, SEXP y_, SEXP row_idx_, SEXP center_, SEXP scale_, 
                         SEXP lambda, SEXP eps_, SEXP max_iter_, SEXP multiplier, 
                         SEXP alpha_, SEXP dfmax_, SEXP user_, SEXP ncore_) {
  // Declarations
  int n = length(row_idx_);
  int p = get_col_bm(X_);
  int L = length(lambda);
  int L_p = L*p;
  
  SEXP res, beta, loss, iter;
  PROTECT(beta = allocVector(REALSXP, L_p));
  double *b = REAL(beta);
  for (int j=0; j<L_p; j++) b[j] = 0;
  PROTECT(loss = allocVector(REALSXP, L));
  for (int i=0; i<L; i++) REAL(loss)[i] = 0.0;
  PROTECT(iter = allocVector(INTSXP, L));
  for (int i=0; i<L; i++) INTEGER(iter)[i] = 0;
  double *a = Calloc(p, double); // Beta from previous iteration
  for (int j=0; j<p; j++) a[j]=0;

  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double *center = REAL(center_);
  double *scale = REAL(scale_);
  double *lam = REAL(lambda);
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier);
  double alpha = REAL(alpha_)[0];
  int dfmax = INTEGER(dfmax_)[0];
  int user = INTEGER(user_)[0];
  double *r = Calloc(n, double);
  for (int i=0; i<n; i++) r[i] = y[i];
  double *z = Calloc(p, double);
  double sumResid = sum(r, n);

  // EDPP: also need to get x[, j] corresponding to lambda_max
  int xmax_idx = 0;
  double temp = 0;
  // record time 1
  //clock_t start, end;
  //double cpu_time_used;
  //char buff1[100];
  //time_t now1 = time (0);
  //strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
  //Rprintf("Crossprod initialization time: %s\n", buff1);
  
  //start = clock();
  for (int j=0; j<p; j++) {
    z[j] = crossprod_bm(X_, r, row_idx, center[j], scale[j], n, j)/n;
    if (z[j] >= temp) {
      temp = z[j];
      xmax_idx = j;
      //Rprintf("\tz[%d] = %f, temp = %f, xmax_idx = %d\n", j, z[j], temp, xmax_idx);
    }
  }
  //end = clock();
  
  //char buff2[100];
  //time_t now2 = time (0);
  //strftime (buff2, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now2));
  //Rprintf("Crossprod initialization time: %s\n", buff2);
  
  //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  //Rprintf("crossprod initialization time: %.15f\n", cpu_time_used);
  
  double l1, l2;
  int converged;

  // EDPP
  double *theta = Calloc(n, double);
  double *v1 = Calloc(n, double);
  double *v2 = Calloc(n, double);
  double *pv2 = Calloc(n, double);
  double *o = Calloc(n, double);
  for (int i = 0; i < n; i++) {
    theta[i] = 0;
    v1[i] = 0;
    v2[i] = 0;
    pv2[i] = 0;
    o[i] = 0;
  }
  
  // index set of nonzero beta's at l+1;
  int *nzero_beta = Calloc(p, int);
  // index set of discarded features at l+1;
  int *discard_beta = Calloc(p, int);
  // number of discarded features at each lambda
  int *discard_count = Calloc(L, int);
  for (int l = 0; l < L; l++) discard_count[l] = 0;
  double pv2_norm = 0;
 
  // set up omp
  int useCores = INTEGER(ncore_)[0];
  //Rprintf("Number of requsted threads: %d\n", useCores);
  int haveCores=omp_get_num_procs();
  //Rprintf("Number of avaialbe processors: %d\n", haveCores);
  if(useCores < 1 || useCores > haveCores) {
    useCores = haveCores;
  }
  //  Rprintf("useCores = %d\n", useCores);
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
  //Rprintf("Number of used threads: %d\n", omp_get_num_threads());
  
  for (int l = 0; l < L-1; l++) {
    //Rprintf("\n-----------------------------------------------\n");
    //char buff[100];
    //time_t now = time (0);
    //strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
    //Rprintf("Lambda %d. Now time: %s\n", l, buff);
    
    if (l == 0) { // lambda_max = lam[0]
      // set beta[, 0] = 0, already done by initialization
      // get theta at lam[0]
      for (int i = 0; i < n; i++) {
        theta[i] =  y[i] / lam[l];
      }
      // compute v1 for lambda_max
      double xty = crossprod_bm(X_, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
      for (int i = 0; i < n; i++) {
        v1[i] = sign(xty) * get_elem_bm(X_, center[xmax_idx], scale[xmax_idx], row_idx[i], xmax_idx);
      }
      REAL(loss)[l] = gLoss(r,n);
      
      for (int j=0; j<p; j++) {
        nzero_beta[j] = 0;
        discard_beta[j] = 1;
      }
      //discard_count[l] = sum_discard(discard_beta, p);
      //Rprintf("lambda[%d] = %f: discarded features: %d\n", l, lam[l], discard_count[l]);
    } else {
      // Check dfmax: a is current beta. At each iteration, solving for next beta.
      int nv = 0;
      for (int j=0; j<p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
       for (int ll=l; ll<L; ll++) INTEGER(iter)[ll] = NA_INTEGER;
        res = cleanup(a, r, z, nzero_beta, discard_beta, theta, v1, v2, o, beta, loss, iter);
        return(res);
      }
      // update theta
      update_theta(theta, X_, row_idx, center, scale, y, b, lam[l], nzero_beta, n, p, l);
      // update v1
      for (int i=0; i < n; i++) {
        v1[i] = y[i] / lam[l] - theta[i];
        //if (i == 0) Rprintf("v1[0] = %f, theta[0] = %f\n", v1[i], theta[0]);
      }
    }
    // update v2:
    for (int i = 0; i < n; i++) {
      v2[i] = y[i] / lam[l+1] - theta[i];
    }
    //update pv2:
    update_pv2(pv2, v1, v2, n);
    // update norm of pv2;
    for (int i = 0; i < n; i++) {
      pv2_norm += pow(pv2[i], 2);
    }
    pv2_norm = pow(pv2_norm, 0.5);
    //Rprintf("\npv2_norm = %f\n", pv2_norm);
   // update o
    for (int i = 0; i < n; i++) {
      o[i] = theta[i] + 0.5 * pv2[i];
    }
    // apply EDPP rule: beta[j+1] = 0 if |xj^T * o| < 1 - 0.5 * pv2_norm * ||xj||_2;
    //    note ||xj||_2 = sqrt(n) since standardization.
    double rhs = n - 0.5 * pv2_norm * sqrt(n); 
    
    //now = time (0);
    //strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
    //Rprintf("EDPP start: Now time: %s\n", buff);
  
    // apply EDPP
    edpp_screen(discard_beta, X_, o, row_idx, center, scale, n, p, rhs);
    //edpp_screen2(discard_beta, X_, o, row_idx, center, scale, n, p, rhs);
    
    //now = time (0);
    //strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
    //Rprintf("EDPP end: Now time: %s\n", buff);
    
    // count discarded features for next lambda
    //discard_count[l+1] = sum_discard(discard_beta, p);
    //Rprintf("lambda[%d] = %f: discarded features: %d\n", l+1, lam[l+1], discard_count[l+1]);
    
    // solve lasso for beta[lam(l+1)] based on the set of non-discarded features.
    //now = time (0);
    //strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
    //Rprintf("Solve lasso start: Now time: %s\n", buff);
    
    while (INTEGER(iter)[l+1] < max_iter) {
      INTEGER(iter)[l+1]++;
      for (int j=0; j<p; j++) {
        if (discard_beta[j] == 0) {
          z[j] = crossprod_resid(X_, r, sumResid, row_idx, center[j], scale[j], n, j) / n + a[j];
          // Update beta_j
          l1 = lam[l+1] * m[j] * alpha;
          l2 = lam[l+1] * m[j] * (1-alpha);
          b[(l+1)*p+j] = lasso(z[j], l1, l2, 1);
          // Update r
          double shift = b[(l+1)*p+j] - a[j];
          if (shift !=0) {
            update_resid(X_, r, shift, row_idx, center[j], scale[j], n, j);
            sumResid = sum(r, n); //update sum of residual
          }
          // update non-zero beta set
          if (b[(l+1)*p+j] != 0) {
            nzero_beta[j] = 1;
          } else {
            nzero_beta[j] = 0;
          }
        }
      }
      // Check for convergence
      converged = checkConvergence(b, a, eps, l+1, p);
      
      // update a
      for (int j=0; j<p; j++) {
        a[j] = b[(l+1)*p+j];   
      }
      if (converged) {
        REAL(loss)[l+1] = gLoss(r, n);
        break;
      } 
    }
    
    //now = time (0);
    //strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
    //Rprintf("Solve lasso end: Now time: %s\n", buff);
  }

  res = cleanup(a, r, z, nzero_beta, discard_beta, theta, v1, v2, o, beta, loss, iter);
  return(res);
}

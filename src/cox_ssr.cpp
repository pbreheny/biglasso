#include "utilities.h"

void Free_memo_cox_ssr(double *s, double *w, double *a, double *r, 
                   int *e1, int*e2, double *eta) {
  Free(s);
  Free(w);
  Free(a);
  Free(r);
  Free(e1);
  Free(e2);
  Free(eta);
}

void standardize_and_get_residual_cox(NumericVector &center, NumericVector &scale, 
                                      int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                      vector<double> &z, double *lambda_max_ptr,
                                      int *xmax_ptr, XPtr<BigMatrix> xMat, 
                                      double *y, double *d, int *d_idx, int *row_idx,
                                      double lambda_min, double alpha, int n, int f, int p);

int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                         int *row_idx, vector<int> &col_idx,
                         NumericVector &center, NumericVector &scale, double *a,
                         double lambda, double sumResid, double alpha, 
                         double *r, double *m, int n, int p);

int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, int *row_idx, 
                   vector<int> &col_idx, NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, double *r, double *m, int n, int p);

void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j);


// Coordinate descent for cox models with SSR
RcppExport SEXP cdfit_cox_ssr(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                              SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                              SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                              SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                              SEXP ncore_, SEXP warn_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_); // Failure indicator for subjects
  double *d = REAL(d_); // Number of failure at unique failure times
  int *d_idx = INTEGER(d_idx_); // Index of unique failure time for subjects with failure; Index of the last unique failure time if censored
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int f = Rf_length(d_); // Number of unique failure times
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
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
  standardize_and_get_residual_cox(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, d, d_idx, row_idx, lambda_min, alpha, n, f, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = Calloc(p, double); //Beta from previous iteration
  double *w = Calloc(n, double); //weights from diagnal of hessian matrix
  double *s = Calloc(n, double); //y_i - yhat_i
  double *r = Calloc(n, double); //s/w
  double *eta = Calloc(n, double); //X\beta
  double *haz = Calloc(n, double); //exp(eta)
  double *rsk = Calloc(f, double); //Sum of hazard over at risk set
  int *e1 = Calloc(p, int); //ever-active set
  int *e2 = Calloc(p, int); //strong set
  double xwr, xwx, yhat, u, v, cutoff, l1, l2, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, k, l, violations, lstart;
  for(j = 0; j < p; j++) e1[j] = 0;
  for(i = 0; i < n; i++) eta[i] = 0;
  double sumWResid = 0.0; //sum w*r
  
  double nullDev = 0;
  double satDev = 0;
  rsk[0] = n;
  k = 0;
  for(i = 0; i < n; i++) {
    if(d_idx[i] >= k) {
      k++;
      if(k >= f) break;
      rsk[k] = rsk[k-1];
    }
    rsk[k] -= 1;
  }
  for (k = 0; k < f; k++) {
    nullDev += 2 * d[k] * log(rsk[k]);
    satDev += 2 * d[k] * log(d[k]);
  }
  nullDev -= satDev;
  thresh = eps * nullDev / n;
  
  
  // set up lambda
  if (user == 0) {
    if (lam_scale) { // set up lambda, equally spaced on log scale
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
        Free_memo_cox_ssr(s, w, a, r, e1, e2, eta);
        return List::create(beta, center, scale, lambda, Dev, 
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
      for (j = 0; j < p; j++) {
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
          
          // Calculate haz, rsk, Dev
          for(i = 0; i < n; i++) haz[i] = exp(eta[i]);
          rsk[0] = sum(haz, n);
          k = 0;
          for(i = 0; i < n; i++) {
            if(d_idx[i] >= k) {
              k++;
              if(k >= f) break;
              rsk[k] = rsk[k-1];
            }
            rsk[k] -= haz[i];
          }
          for(i = 0; i < n; i++) {
            Dev[l] -= 2 * y[i] * (eta[i] - log(rsk[d_idx[i]])); 
          }
          Dev[l] -= satDev;
          
          // Check for saturation
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            Free_memo_cox_ssr(s, w, a, r, e1, e2, eta);
            return List::create(beta, center, scale, lambda, Dev,
                                iter, n_reject, Rcpp::wrap(col_idx));
          }
          
          // Calculate w, s, r
          for(i = 0; i < n; i++) {
            w[i] = 0.0;
            s[i] = y[i];
            for(k = 0; k <= d_idx[i]; k++) {
              w[i] += d[k] * (rsk[k] - haz[i]) / rsk[k] / rsk[k];
              s[i] -= d[k] * haz[i] / rsk[k];
            }
            w[i] *= haz[i];
            if(w[i] == 0) r[i] = 0.0;
            else r[i] = s[i] / w[i];
          }
          
          
          
          // Update beta
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[jj], scale[jj], w, n, jj);
              xwx = wsqsum_bm(xMat, w, row_idx, center[jj], scale[jj], n, jj);
              u = xwr / n + xwx * a[j] / n;
              v = xwx / n;
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                
                update = pow(beta(j, l) - a[j], 2) * v;
                if (update > max_update) max_update = update;
                update_resid_eta(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumWResid = wsum(r, w, n); // update temp result w * r, used for computing xwr;
                a[j] = beta(j, l); // update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh)  break;
        }
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], 0.0, alpha, s, m, n, p);
        if (violations==0) break;
      }
      // Scan for violations in rest
      violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], 0.0, alpha, s, m, n, p);
      if (violations==0) break;
    }
  }
  Free_memo_cox_ssr(s, w, a, r, e1, e2, eta);
  return List::create(beta, center, scale, lambda, Dev, iter, n_reject, Rcpp::wrap(col_idx));
  
}
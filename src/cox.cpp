#include "utilities.h"


// standardize
void standardize_and_get_residual_cox(NumericVector &center, NumericVector &scale, 
                                      int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                      vector<double> &z, double *lambda_max_ptr,
                                      int *xmax_ptr, XPtr<BigMatrix> xMat, 
                                      double *y, double *d, int *d_idx, int *row_idx,
                                      double lambda_min, double alpha, int n, int f, int p) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  double sum_xs;
  double zmax = 0.0, zj = 0.0;
  int i, j, k;
  double *s = Calloc(n, double);
  double *rsk = Calloc(f, double);
  
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
  for(i = 0; i < n; i++) {
    s[i] = y[i];
    for(k = 0; k <= d_idx[i]; k++) {
      s[i] -= d[k] / rsk[k];
    }
  }
  
  for (j = 0; j < p; j++) {
    xCol = xAcc[j];
    sum_xs = 0.0;
    
    for (i = 0; i < n; i++) {
      center[j] += xCol[row_idx[i]];
      scale[j] += pow(xCol[row_idx[i]], 2);
      sum_xs = sum_xs + xCol[row_idx[i]] * s[i];
    }
    
    center[j] = center[j] / n; //center
    scale[j] = sqrt(scale[j] / n - pow(center[j], 2)); //scale
    
    if (scale[j] > 1e-6) {
      col_idx.push_back(j);
      zj = sum_xs / (scale[j] * n); 
      if (fabs(zj) > zmax) {
        zmax = fabs(zj);
        *xmax_ptr = j; // xmax_ptr is the index in the raw xMat, not index in col_idx!
      }
      z.push_back(zj);
    }
  }
  *p_keep_ptr = col_idx.size();
  *lambda_max_ptr = zmax / alpha;
  Free(s);
  Free(rsk);
}

// dual function g(lambda/lambda_0*Theta)
double dual_cox(double *haz, double *rsk, double lambda, double lambda_0,
                int n, int f, double *y, double *d, int *d_idx) {
  double res = 0.0;
  double lam_ratio = lambda / lambda_0;
  int i, k;
  for (i = 0; i < n; i++) {
    for(k = 0; k < d_idx[i]; k++) {
      res += lam_ratio * d[k] * haz[i] / rsk[k] *
        log(lam_ratio * haz[i] / rsk[k]);
    }
    k = d_idx[i];
    res += ((1 - lam_ratio) + lam_ratio * d[k] * haz[i] / rsk[k]) *
      log((1 - lam_ratio) / d[k] + lam_ratio * haz[i] / rsk[k]);
  }
  return res;
}

// Scox initialization
void scox_init(double *g_theta_lam_ptr, double *prod_deriv_theta_lam_ptr,
               vector<double>& prodP_X_xmax, vector<double>& scaleP_X,
               vector<double>& X_theta_lam, XPtr<BigMatrix> xMat,
               double *haz, double *rsk, vector<double>& z, int xmax_col_idx,
               int *row_idx, vector<int> &col_idx,
               NumericVector &center, NumericVector &scale,
               int n, int p, int f, double *y, double *d, int *d_idx) {
  
  *g_theta_lam_ptr = dual_cox(haz, rsk, 1.0, 1.0, n, f, y, d, d_idx);
  double prod_deriv_theta_lam = 0.0;
  int i, k;
  for (i = 0; i < n; i++) {
    for(k = 0; k < d_idx[i]; k++) {
      prod_deriv_theta_lam += d[k] * haz[i] / rsk[k] * log(haz[i] / rsk[k]);
    }
    k = d_idx[i];
    prod_deriv_theta_lam += (d[k] * haz[i] / rsk[k] - 1) * log(haz[i] / rsk[k]);
  }
  *prod_deriv_theta_lam_ptr = prod_deriv_theta_lam;
  double sign_xmaxTs = sign(z[col_idx[xmax_col_idx]]);
  
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  int j, jj;
  double max_x, min_x;
  double *xmax = xAcc[col_idx[xmax_col_idx]];
  double scale_max = scale[col_idx[xmax_col_idx]];
  double *diff_xmax = Calloc(f, double);
  
  // Initialize ||xmax||_{diff,k}
  i = n-1;
  max_x = min_x = xmax[row_idx[n-1]];
  for(k = f-1; k >= 0; k--) {
    for(; i >=0 && d_idx[i] >= k; i--) {
      if(xmax[row_idx[i]] > max_x) max_x = xmax[row_idx[i]];
      if(xmax[row_idx[i]] < min_x) min_x = xmax[row_idx[i]];
    }
    diff_xmax[k] = max_x - min_x;
  }
  
  // Initialize ||x_j||_P and <x_j,xmax>_P
#pragma omp parallel for private(j, jj, i, k, max_x, min_x) schedule(static) 
  for (j = 0; j < p; j++) {
    jj = col_idx[j];
    xCol = xAcc[jj];
    X_theta_lam[j] = -z[j];
    scaleP_X[j] = 0; 
    prodP_X_xmax[j] = 0;
    i = n-1;
    max_x = min_x = xCol[row_idx[n-1]];
    for(k = f-1; k >= 0; k--) {
      for(; i >=0 && d_idx[i] >= k; i--) {
        if(xCol[row_idx[i]] > max_x) max_x = xCol[row_idx[i]];
        if(xCol[row_idx[i]] < min_x) min_x = xCol[row_idx[i]];
      }
      scaleP_X[j] += d[k] * pow(max_x - min_x, 2);
      prodP_X_xmax[j] += d[k] * (max_x - min_x) * diff_xmax[k];
    }
    scaleP_X[j] = sqrt(scaleP_X[j]) / scale[jj] / 2;
    prodP_X_xmax[j] = prodP_X_xmax[j] / scale[jj] / scale_max / 4;
  }
  Free(diff_xmax);
}

// Scox screening
void scox_screen(int* scox_reject, double lambda, double lambda_0,
                 double *haz, double *rsk, double g_theta_lam,
                 double prod_deriv_theta_lam, vector<double>& prodP_X_xmax,
                 vector<double>& scaleP_X, vector<double>& X_theta_lam,
                 int xmax_col_idx, int *row_idx, vector<int> &col_idx,
                 NumericVector &center, NumericVector &scale,
                 int n, int p, int f, double *y, double *d, int *d_idx) {
  
  double TOLERANCE = 1e-8;
  double scaleP_Xmax = scaleP_X[xmax_col_idx];
  double r = sqrt(2 * (dual_cox(haz, rsk, lambda, lambda_0, n, f, y, d, d_idx) - 
                  g_theta_lam + (1 - lambda/lambda_0) * prod_deriv_theta_lam));
  double D = n * (lambda_0 - lambda) / r / scaleP_Xmax;
  double a2 = (1 - pow(D, 2)) * pow(scaleP_Xmax, 4);
  double a1, a0, Delta, u2, rho, T;
  int j;
  
#pragma omp parallel for private(j, a1, a0, Delta, u2, rho, T) schedule(static)
  for (j = 0; j < p; j++) {
    // xi=1
    scox_reject[j] = 1;
    rho = prodP_X_xmax[j] / scaleP_Xmax / scaleP_X[j];
    a1 = 2 * (1 - pow(D, 2)) * prodP_X_xmax[j] * pow(scaleP_Xmax, 2);
    a0 = pow(prodP_X_xmax[j], 2) - pow(D * scaleP_Xmax * scaleP_X[j], 2);
    Delta = pow(a1, 2) - 4 * a0 * a2;
    if (Delta < 0.0) Delta = 0.0; // in case of -0.0 (at xmax)
    u2 = (-a1 + sqrt(Delta)) / 2 / a2;
    if(j==p-1) Rprintf("rho=%f, Delta=%f, u2=%f\n", rho, Delta, u2);
    if(rho >= D) {
      T = X_theta_lam[j] + r * scaleP_X[j] / n;
      if(j==p-1) Rprintf("T+[%i]/lam=%f+%f*%f/%f\n", j, X_theta_lam[j], r/sqrt(n), scaleP_X[j]/sqrt(n), lambda);
      
    } else {
      T = X_theta_lam[j] - u2 * (lambda_0 - lambda) +
        r * sqrt(pow(scaleP_X[j],2) + 2 * u2 * prodP_X_xmax[j] + pow(u2 * scaleP_Xmax, 2)) / n;
      if(j==p-1) Rprintf("T+[%i]/lam=%f+%f+%f*%f/%f\n", j, X_theta_lam[j], -u2 * (lambda_0 - lambda), r/sqrt(n), sqrt(pow(scaleP_X[j],2) + 2 * u2 * prodP_X_xmax[j] + pow(u2 * scaleP_Xmax, 2))/sqrt(n), lambda);
    }
    if(T + TOLERANCE > lambda) {
      scox_reject[j] = 0;
    }
    // xi=-1
    //rho = -rho;
    //a1 = -a1;
    //u2 = (-a1 + sqrt(Delta)) / 2 / a2;
    if(rho >= D) {
      T = -X_theta_lam[j] + r * scaleP_X[j] / n;
      if(j==p-1) Rprintf("T-[%i]/lam=%f+%f*%f/%f\n", j, -X_theta_lam[j], r/sqrt(n), scaleP_X[j]/sqrt(n), lambda);
    } else {
      T = -X_theta_lam[j] - u2 * (lambda_0 - lambda) +
        r * sqrt(pow(scaleP_X[j],2) + 2 * u2 * prodP_X_xmax[j] + pow(u2 * scaleP_Xmax, 2)) / n;
      if(j==p-1) Rprintf("T-[%i]/lam=%f+%f+%f*%f/%f\n", j, -X_theta_lam[j], -u2 * (lambda_0 - lambda), r/sqrt(n), sqrt(pow(scaleP_X[j],2) + 2 * u2 * prodP_X_xmax[j] + pow(u2 * scaleP_Xmax, 2))/sqrt(n), lambda);
    }
    if(T + TOLERANCE > lambda) {
      scox_reject[j] = 0;
    }
  }
}

// Coordinate descent for cox models
RcppExport SEXP cdfit_cox(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
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
  double xwr, xwx, u, v, l1, l2, shift;
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
        Free(s); Free(w); Free(a); Free(r); Free(e1); Free(eta); Free(haz); Free(rsk);
        return List::create(beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
    }
    
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        iter[l]++;
        Dev[l] = 0.0;
        
        // Calculate haz, rsk, Dev
        for(i = 0; i < n; i++) haz[i] = exp(eta[i]);
        rsk[f-1] = haz[n-1];
        k = f-1;
        for(i = n-2; i >= 0; i--) {
          if(d_idx[i] < k) {
            k--;
            rsk[k] = rsk[k+1];
          }
          rsk[k] += haz[i];
        }
        for(i = 0; i < n; i++) {
          Dev[l] -= 2 * y[i] * (eta[i] - log(rsk[d_idx[i]])); 
        }
        Dev[l] -= satDev;
        
        // Check for saturation
        if (Dev[l] / nullDev < .01) {
          if (warn) warning("Model saturated with deviance %f; exiting...", Dev[l]);
          for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
          Free(s); Free(w); Free(a); Free(r); Free(e1); Free(eta); Free(haz); Free(rsk);
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
      // Scan for violations in inactive set
      violations = check_inactive_set(e1, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], 0.0, alpha, s, m, n, p);
      if (violations==0) break;
    }
  }
  Free(s); Free(w); Free(a); Free(r); Free(e1); Free(eta); Free(haz); Free(rsk);
  return List::create(beta, center, scale, lambda, Dev, iter, n_reject, Rcpp::wrap(col_idx));
  
}

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
  double xwr, xwx, u, v, cutoff, l1, l2, shift;
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
        Free(s); Free(w); Free(a); Free(r); Free(e1); Free(e2); Free(eta); Free(haz); Free(rsk);
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
          rsk[f-1] = haz[n-1];
          k = f-1;
          for(i = n-2; i >= 0; i--) {
            if(d_idx[i] < k) {
              k--;
              rsk[k] = rsk[k+1];
            }
            rsk[k] += haz[i];
          }
          for(i = 0; i < n; i++) {
            Dev[l] -= 2 * y[i] * (eta[i] - log(rsk[d_idx[i]])); 
          }
          Dev[l] -= satDev;
          
          // Check for saturation
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated with deviance %f; exiting...", Dev[l]);
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            Free(s); Free(w); Free(a); Free(r); Free(e1); Free(e2); Free(eta); Free(haz); Free(rsk);
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
  Free(s); Free(w); Free(a); Free(r); Free(e1); Free(e2); Free(eta); Free(haz); Free(rsk);
  return List::create(beta, center, scale, lambda, Dev, iter, n_reject, Rcpp::wrap(col_idx));
  
}

// Coordinate descent for cox models with Scox
RcppExport SEXP cdfit_cox_scox(SEXP X_, SEXP y_, SEXP d_, SEXP d_idx_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                               SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP warn_, SEXP safe_thresh_, SEXP verbose_) {
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
  double safe_thresh = REAL(safe_thresh_)[0]; // threshold for safe test
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
  int xmax_col_idx = 0;
  int *xmax_ptr = &xmax_col_idx;
  
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
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_col_idx;
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
  double xwr, xwx, u, v, cutoff, l1, l2, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, k, l, violations, lstart;
  for(j = 0; j < p; j++) e1[j] = 0;
  for(i = 0; i < n; i++) {
    eta[i] = 0;
    haz[i] = 1;
  }
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
  
  // Scox variables
  double g_theta_lam = 0.0;
  double prod_deriv_theta_lam = 0.0;
  double *g_theta_lam_ptr = &g_theta_lam;
  double *prod_deriv_theta_lam_ptr = &prod_deriv_theta_lam;
  vector<double> X_theta_lam; 
  vector<double> prodP_X_xmax;
  vector<double> scaleP_X;
  int *safe_reject = Calloc(p, int);
  
  int scox; // if 0, don't perform Scox rule
  if (safe_thresh < 1) {
    scox = 1; // turn on scox
    X_theta_lam.resize(p);
    prodP_X_xmax.resize(p);
    scaleP_X.resize(p);
    
    scox_init(g_theta_lam_ptr, prod_deriv_theta_lam_ptr, prodP_X_xmax,
              scaleP_X, X_theta_lam, xMat, haz, rsk, z, xmax_col_idx,
              row_idx, col_idx, center, scale, n, p, f, y, d, d_idx);
  } else {
    scox = 0;
  }
  
  if (scox == 1 && user == 0) n_reject[0] = p;
  
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
        Free(s); Free(w); Free(a); Free(r); Free(e1); Free(safe_reject); Free(eta); Free(haz); Free(rsk);
        return List::create(beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
    } 
    
    if(scox) {
      scox_screen(safe_reject, lambda[l], lambda_max, haz, rsk, g_theta_lam,
                  prod_deriv_theta_lam, prodP_X_xmax, scaleP_X, X_theta_lam,
                  xmax_col_idx, row_idx, col_idx, center, scale, n, p, f, y, d, d_idx);
    }
    n_reject[l] = sum(safe_reject, p);
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        iter[l]++;
        Dev[l] = 0.0;
        
        // Calculate haz, rsk, Dev
        for(i = 0; i < n; i++) haz[i] = exp(eta[i]);
        rsk[f-1] = haz[n-1];
        k = f-1;
        for(i = n-2; i >= 0; i--) {
          if(d_idx[i] < k) {
            k--;
            rsk[k] = rsk[k+1];
          }
          rsk[k] += haz[i];
        }
        for(i = 0; i < n; i++) {
          Dev[l] -= 2 * y[i] * (eta[i] - log(rsk[d_idx[i]])); 
        }
        Dev[l] -= satDev;
        
        // Check for saturation
        if (Dev[l] / nullDev < .01) {
          if (warn) warning("Model saturated with deviance %f; exiting...", Dev[l]);
          for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
          Free(s); Free(w); Free(a); Free(r); Free(e1); Free(safe_reject); Free(eta); Free(haz); Free(rsk);
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
      // Scan for violations in safe set
      violations = check_safe_set(e1, safe_reject, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], 0.0, alpha, s, m, n, p);
      if (violations==0) break;
    }
  }
  Free(s); Free(w); Free(a); Free(r); Free(e1); Free(safe_reject); Free(eta); Free(haz); Free(rsk);
  return List::create(beta, center, scale, lambda, Dev, iter, n_reject, Rcpp::wrap(col_idx));
}

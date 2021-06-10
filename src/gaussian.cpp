#include "utilities.h"
//#include "gperftools/profiler.h"

// apply EDPP 
void edpp_screen(int *discard_beta, int n, int p, double *Xtr, double *Xty,
                 double y_norm2, double ytyhat, double yhat_norm2,
                 double beta_norm, double lambda0, double lambda1,
                 double *m, double alpha, vector<int> &col_idx, bool basic) {
  int j;
  double c, t, RHS, cr, cy;
  c = (lambda0 - lambda1) / lambda0 / lambda1;
  if (basic) {
    t = c * lambda0 / (1 + (1-alpha) * lambda1);
    RHS = n*alpha - c/2*sqrt(n*(1+(1-alpha)*lambda1)*(y_norm2-lambda0*lambda0*alpha*alpha*n/(1+(1-alpha)*lambda1)));
    cr = t *alpha / 2;
    cy = (1 / lambda0 + c / 2);
  } else {
    double gamma_norm2 = n*(1-alpha)*lambda1*beta_norm*beta_norm;
    double delta = yhat_norm2+lambda0/lambda1/lambda1*(2*lambda1-lambda0)*gamma_norm2;
    if(delta <= 0) t = 0;
    else t = 1 + c*lambda0*ytyhat/(yhat_norm2+gamma_norm2)-
      c*c*lambda0*lambda0/(yhat_norm2+gamma_norm2)*
      sqrt(gamma_norm2*(y_norm2*(yhat_norm2+gamma_norm2)-ytyhat)/delta);
    if(t < 0) t = 0;
    RHS = n * alpha - (t+1+fabs(1-t)) / 2 * c * sqrt(n*gamma_norm2) -
      sqrt(n*((1-t)*((1-t)*(yhat_norm2+gamma_norm2)+2*c*lambda0*ytyhat)+c*c*lambda0*lambda0*y_norm2))/2/lambda0;
    cr = (t + 1) / 2 / lambda0;
    cy = ((1-t) / 2 / lambda0 + c / 2);
  }
  for(j = 0; j < p; j++) {
    if(fabs(cr*Xtr[j] + cy*Xty[j]) < RHS) discard_beta[j] = 1;
    else discard_beta[j] = 0;
  }
}

// Initialize EDPP rule
void edpp_init(XPtr<BigMatrix> xpMat, double *Xtr, double sxty, int xmax_idx,
               int *row_idx, vector<int>& col_idx, NumericVector& center,
               NumericVector& scale, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, *xCol_max;
  double sum;
  xCol_max = xAcc[xmax_idx];
  int j, jj, i;
  
#pragma omp parallel for private(i, j, sum) schedule(static) 
  for (j = 0; j < p; j++) { 
    jj = col_idx[j]; 
    xCol = xAcc[jj];
    sum = 0.0;
    for (i = 0; i < n; i++) {
      sum += xCol[row_idx[i]] * xCol_max[row_idx[i]];
    }
    sum = (sum - n * center[jj] * center[xmax_idx]) / (scale[jj] * scale[xmax_idx]);
    Xtr[j] = -sxty * sum;
  }
}

// Update EDPP rule
void edpp_update(XPtr<BigMatrix> xpMat, double *r, double sumResid,
                 double *Xtr, int *row_idx, vector<int>& col_idx, 
                 NumericVector& center, NumericVector& scale, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol;
  int j, jj;
  double sum;
#pragma omp parallel for schedule(static) private(j, jj, xCol, sum)
  for(j = 0; j < p; j++){
    jj = col_idx[j];
    xCol = xAcc[jj];
    sum = 0.0;
    for(int i = 0; i < n; i++) {
      sum = sum + xCol[row_idx[i]] * r[i];
    }
    sum = (sum - center[jj] * sumResid) / scale[jj];
    Xtr[j] = sum;
  }
}

// compute quantities needed in bedpp
void bedpp_init(vector<double>& sign_lammax_xtxmax,
                XPtr<BigMatrix> xMat, int xmax_idx, double *y, double lambda_max, 
                int *row_idx, vector<int>& col_idx, NumericVector& center, 
                NumericVector& scale, int n, int p) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol, *xCol_max;
  double sum_xjxmax, sum_xmaxTy, sign_xmaxTy;
  xCol_max = xAcc[xmax_idx];
  int j, jj;
  // sign of xmaxTy
  sum_xmaxTy = crossprod_bm(xMat, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
  sign_xmaxTy = sign(sum_xmaxTy);
  
#pragma omp parallel for private(j, sum_xjxmax) schedule(static) 
  for (j = 0; j < p; j++) { // p = p_keep
    jj = col_idx[j]; // index in the raw XMat, not in col_idx;
    if (jj != xmax_idx) {
      xCol = xAcc[jj];
      sum_xjxmax = 0.0;
      for (int i = 0; i < n; i++) {
        sum_xjxmax = sum_xjxmax + xCol[row_idx[i]] * xCol_max[row_idx[i]];
      }
      sign_lammax_xtxmax[j] = sign_xmaxTy * lambda_max * (sum_xjxmax - n * center[jj] * 
        center[xmax_idx]) / (scale[jj] * scale[xmax_idx]);;
    } else {
      sign_lammax_xtxmax[j] = sign_xmaxTy * lambda_max * n;
    }
  }
}

// Basic (non-sequential) EDPP test
void bedpp_screen(int *bedpp_reject, const vector<double>& sign_lammax_xtxmax,
                  const vector<double>& XTy, double ynorm_sq, int *row_idx, 
                  vector<int>& col_idx, double lambda, double lambda_max, 
                  double alpha, int n, int p) {
  double LHS = 0.0;
  double RHS = 2 * n * alpha * lambda * lambda_max - (lambda_max - lambda) * 
    sqrt(n * ynorm_sq * (1 + lambda * (1 - alpha)) - pow(n * alpha * lambda_max, 2));
  int j;
  
#pragma omp parallel for private(j, LHS) schedule(static)
  for (j = 0; j < p; j++) { // p = p_keep
    LHS = (lambda + lambda_max) * XTy[j] - (lambda_max - lambda) * alpha * sign_lammax_xtxmax[j] / (1 + lambda * (1 - alpha));
    if (fabs(LHS) < RHS) {
      bedpp_reject[j] = 1;
    } else {
      bedpp_reject[j] = 0;
    }
  }
}

// Coordinate descent for gaussian models with ssr
RcppExport SEXP cdfit_gaussian_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_, 
                                   SEXP lam_scale_, SEXP lambda_min_, 
                                   SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  
  NumericVector lambda(L);
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
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr,
                               xmax_ptr, xMat, y, row_idx, alpha, n, p);
  
  p = p_keep;   // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); // beta
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  
  double l1, l2, cutoff, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  int *e1 = Calloc(p, int); // ever active set
  int *e2 = Calloc(p, int); // strong set
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r,n);
  thresh = eps * loss[0] / n;
  
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
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  // Path
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
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        Free(a); Free(r); Free(e1); Free(e2);
        return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
      }
      // strong set
      cutoff = 2 * lambda[l] - lambda[l-1];
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
    
    while(iter[l] < max_iter) {
      while(iter[l] < max_iter){
        while(iter[l] < max_iter) {
          iter[l]++;
          
          //solve lasso over ever-active set
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(z[j], l1, l2, 1);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // compute objective update for checking convergence
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2);
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumResid = sum(r, n); //update sum of residual
                a[j] = beta(j, l); //update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p); 
        if (violations==0) break;
      }
      
      // Scan for violations in rest set
      violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p);
      if (violations == 0) {
        loss[l] = gLoss(r, n);
        break;
      }
    }
  }
  
  Free(a); Free(r); Free(e1); Free(e2);
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
}



// Coordinate descent for gaussian models with ada-edpp-ssr
RcppExport SEXP cdfit_gaussian_ada_edpp_ssr(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                            SEXP lam_scale_, SEXP lambda_min_, SEXP alpha_, SEXP user_,
                                            SEXP eps_, SEXP max_iter_, SEXP multiplier_, SEXP dfmax_,
                                            SEXP ncore_, SEXP update_thresh_, SEXP verbose_) {
  //ProfilerStart("Ada_EDPP.out");
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
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  double update_thresh = REAL(update_thresh_)[0];
  
  NumericVector lambda(L);
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
  
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, 
                               lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, alpha, n, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  
  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); //Beta
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  IntegerVector n_safe_reject(L);
  
  double l1, l2, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart; //temp index
  int *ever_active = Calloc(p, int); // ever-active set
  int *strong_set = Calloc(p, int); // strong set
  int *discard_beta = Calloc(p, int); // index set of discarded features;
  //int *discard_old = Calloc(p, int);
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r, n);
  thresh = eps * loss[0] / n;
  
  // EDPP
  double *Xty = Calloc(p, double);
  double *Xtr = Calloc(p, double); // Xtr at previous update of EDPP
  for(j = 0; j < p; j++) {
    Xty[j] = z[j] * n;
    Xtr[j] = Xty[j];
  }
  double *yhat = Calloc(n, double); // yhat at previous rupdate of EDPP
  double yhat_norm2;
  double ytyhat;
  double y_norm2 = 0; // ||y||^2
  for(i = 0; i < n; i++) y_norm2 += y[i] * y[i];
  double beta_norm = 0; // ||beta||_2
  bool basic = true; // Whether using BEDPP or EDPP
  double cutoff = 0; // cutoff for adaptive strong rule
  double cutoff0 = 0; // cutoff for sequential strong rule
  int gain = 0; // gain from updating EDPP
  
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
    lstart = 1;
    n_safe_reject[0] = p;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  } 
  
  int l_prev = 0; // lambda index at previous update of EDPP
  // compute v1 for lambda_max
  double sxty = sign(crossprod_bm(xMat, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx));
  
  
  
  // Path
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    if(l != lstart) {
      int nv = 0;
      for (int j=0; j<p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        Free(ever_active); Free(r); Free(a); Free(discard_beta); Free(Xty); Free(Xtr); Free(yhat); Free(strong_set); //Free(discard_old); 
        return List::create(beta, center, scale, lambda, loss, iter,  n_reject, n_safe_reject, Rcpp::wrap(col_idx));
      }
      
      // strong set
      //update_zj(z, discard_beta, discard_old, xMat, row_idx, col_idx, center, scale, 
      //sumResid, r, m, n, p);
      if(l != lstart){
        cutoff= 2 * lambda[l] - lambda[l_prev];
        cutoff0 = 2 * lambda[l] - lambda[l-1];
      } 
      for(j = 0; j < p; j++) {
        if(discard_beta[j]) {
          if(fabs(z[j]) > cutoff * alpha * m[col_idx[j]]) {
            strong_set[j] = 1;
          } else {
            strong_set[j] = 0;
          }
        } else {
          if(fabs(z[j]) > cutoff0 * alpha * m[col_idx[j]]) {
            strong_set[j] = 1;
          } else {
            strong_set[j] = 0;
          }
        }
        
      }
      n_reject[l] = p - sum(strong_set, p);
      
      if(gain - n_safe_reject[l - 1] * (l - l_prev - 1) > update_thresh * p && l != L - 1) { // Update EDPP if not discarding enough
        if(verbose) {
          // output time
          char buff[100];
          time_t now = time (0);
          strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
          Rprintf("Start updating EDPP rule at lambda %d. Now time: %s\n", l, buff);
        }
        basic = false;
        l_prev = l-1;
        yhat_norm2 = 0;
        ytyhat = 0;
        for(i = 0; i < n; i++) {
          yhat[i] = y[i] - r[i];
          yhat_norm2 += yhat[i] * yhat[i];
          ytyhat += y[i] * yhat[i];
        }
        // Update EDPP rule
        edpp_update(xMat, r, sumResid, Xtr, row_idx, col_idx, center, scale, n, p);
        beta_norm = 0;
        for(j = 0; j < p; j++) {
          beta_norm += a[j] * a[j];
          z[j] = Xtr[j] / n;
        }
        beta_norm = sqrt(beta_norm);
        if(verbose) {
          // output time
          char buff[100];
          time_t now = time (0);
          strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
          Rprintf("Done updating EDPP rule at lambda %d. Now time: %s\n", l, buff);
        }
        
        // Reapply EDPP
        edpp_screen(discard_beta, n, p, Xtr, Xty, y_norm2, ytyhat, yhat_norm2,
                    beta_norm, lambda[l_prev], lambda[l], m, alpha, col_idx, false);
        n_safe_reject[l] = sum(discard_beta, p);
        gain = n_safe_reject[l];
      } else {
        // Apply EDPP to discard features
        if(basic) { // Apply BEDPP check
          edpp_screen(discard_beta, n, p, Xtr, Xty, y_norm2, ytyhat, yhat_norm2,
                      0, lambda_max, lambda[l], m, alpha, col_idx, true);
        } else { // Apply EDPP check
          edpp_screen(discard_beta, n, p, Xtr, Xty, y_norm2, ytyhat, yhat_norm2,
                      beta_norm, lambda[l_prev], lambda[l], m, alpha, col_idx, false);
        }
        n_safe_reject[l] = sum(discard_beta, p);
        gain += n_safe_reject[l];
      }
      
    } else { //First check with lambda max
      if(verbose) {
        // output time
        char buff[100];
        time_t now = time (0);
        strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
        Rprintf("Start calculating BEDPP rule. Now time: %s\n", buff);
      }
      // For bedpp, pretend Xtr = -sign(x_max^ty) * X^Tx_max
      edpp_init(xMat, Xtr, sxty, xmax_idx, row_idx, col_idx, center, scale, n, p);
      if(verbose) {
        // output time
        char buff[100];
        time_t now = time (0);
        strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
        Rprintf("Done calculating BEDPP rule. Now time: %s\n", buff);
      }
      edpp_screen(discard_beta, n, p, Xtr, Xty, y_norm2, ytyhat, yhat_norm2, 0,
                  lambda_max, lambda[l], m, alpha, col_idx, true);
      n_safe_reject[l] = sum(discard_beta, p);
      gain = n_safe_reject[l];
    }
    //for(j = 0; j < p; j++) discard_old[j] = discard_beta[j];
    
    while(iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (ever_active[j]) {
              jj = col_idx[j];
              z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(z[j], l1, l2, 1);
              
              shift = beta(j, l) - a[j];
              if (shift != 0) {
                // compute objective update for checking convergence
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l+1), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l+1)) -  fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2);
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj);
                sumResid = sum(r, n); //update sum of residual
                a[j] = beta(j, l); //update a
              }
              // update ever active sets
              if (beta(j, l) != 0) {
                ever_active[j] = 1;
              } 
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        violations = check_strong_set(ever_active, strong_set, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p); 
        if (violations==0) break;
      }	
      // Scan for violations in edpp set
      violations = check_rest_safe_set(ever_active, strong_set, discard_beta, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p); 
      if (violations == 0) {
        loss[l] = gLoss(r, n);
        break;
      }
      
    }
  }
  
  Free(ever_active); Free(r); Free(a); Free(discard_beta); Free(Xty); Free(Xtr); Free(yhat); Free(strong_set); //Free(discard_old);
  //ProfilerStop();
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, n_safe_reject, Rcpp::wrap(col_idx));
}

// Coordinate descent for gaussian models with bedpp_ssr
RcppExport SEXP cdfit_gaussian_bedpp_ssr(SEXP X_, SEXP y_, SEXP row_idx_,  
                                         SEXP lambda_, SEXP nlambda_,
                                         SEXP lam_scale_,
                                         SEXP lambda_min_, SEXP alpha_, 
                                         SEXP user_, SEXP eps_,
                                         SEXP max_iter_, SEXP multiplier_, 
                                         SEXP dfmax_, SEXP ncore_, 
                                         SEXP safe_thresh_,
                                         SEXP verbose_) {
  //ProfilerStart("HSR.out");
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double bedpp_thresh = REAL(safe_thresh_)[0]; // threshold for safe test
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  
  NumericVector lambda(L);
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
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr,
                               xmax_ptr, xMat, y, row_idx, alpha, n, p);
  
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); // Beta
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L); // number of total rejections;
  IntegerVector n_bedpp_reject(L); 
  
  double l1, l2, cutoff, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart; 
  int *e1 = Calloc(p, int); // ever-active set
  int *e2 = Calloc(p, int); // strong set
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) r[i] = y[i];
  double sumResid = sum(r, n);
  loss[0] = gLoss(r,n);
  thresh = eps * loss[0] / n;
  
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
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  /* Variables used for BEDPP test */
  vector<double> xty;
  vector<double> sign_lammax_xtxmax;
  double ynorm_sq = 0;
  int *bedpp_reject = Calloc(p, int);
  int *bedpp_reject_old = Calloc(p, int);
  int bedpp; // if 0, don't perform bedpp test
  if (bedpp_thresh < 1) {
    bedpp = 1; // turn on bedpp test
    xty.resize(p);
    sign_lammax_xtxmax.resize(p);
    for (j = 0; j < p; j++) {
      xty[j] = z[j] * n;
    }
    ynorm_sq = sqsum(y, n, 0);
    bedpp_init(sign_lammax_xtxmax, xMat, xmax_idx, y, lambda_max, row_idx, col_idx, center, scale, n, p);
  } else {
    bedpp = 0; // turn off bedpp test
  }
  
  if (bedpp == 1 && user == 0) n_bedpp_reject[0] = p;
  
  // Path
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
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll = l; ll < L; ll++) iter[ll] = NA_INTEGER;
        Free(a); Free(r); Free(e1); Free(e2); Free(bedpp_reject); Free(bedpp_reject_old);
        return List::create(beta, center, scale, lambda, loss, iter, 
                            n_reject, n_bedpp_reject, Rcpp::wrap(col_idx));
      }
      cutoff = 2*lambda[l] - lambda[l-1];
    } else {
      cutoff = 2*lambda[l] - lambda_max;
    }
    
    if (bedpp) {
      bedpp_screen(bedpp_reject, sign_lammax_xtxmax, xty, ynorm_sq, row_idx, col_idx, lambda[l], lambda_max, alpha, n, p);
      n_bedpp_reject[l] = sum(bedpp_reject, p);
      
      // update z[j] for features which are rejected at previous lambda but accepted at current one.
      update_zj(z, bedpp_reject, bedpp_reject_old, xMat, row_idx, col_idx, center, scale, sumResid, r, m, n, p);
           
#pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        // update bedpp_reject_old with bedpp_reject
        bedpp_reject_old[j] = bedpp_reject[j];
        // hsr screening
        if (bedpp_reject[j] == 0 && (fabs(z[j]) >= (cutoff * alpha * m[col_idx[j]]))) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      n_bedpp_reject[l] = 0; // no bedpp test;
      // hsr screening over all
#pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) >= (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p); // e2 set means not reject by bedpp or hsr;
    
    while(iter[l] < max_iter) {
      while(iter[l] < max_iter){
        while(iter[l] < max_iter) {
          iter[l]++;
          
          //solve lasso over ever-active set
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) { 
              jj = col_idx[j];
              z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(z[j], l1, l2, 1);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // compute objective update for checking convergence
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2);
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj); // Update r
                sumResid = sum(r, n); //update sum of residual
                a[j] = beta(j, l); //update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p);
        if (violations == 0) break;
      }
      
      // Scan for violations in rest set
      if (bedpp) {
        violations = check_rest_safe_set(e1, e2, bedpp_reject, z, xMat, row_idx, col_idx,center, scale, a, lambda[l], sumResid, alpha, r, m, n, p);
      } else {
        violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p);
      }
      
      if (violations == 0) {
        loss[l] = gLoss(r, n);
        break;
      }
    }
    
    if (n_bedpp_reject[l] <= p * bedpp_thresh) {
      bedpp = 0; // turn off bedpp for next iteration if not efficient
    }
  }
  
  Free(a); Free(r); Free(e1); Free(e2); Free(bedpp_reject); Free(bedpp_reject_old);
  //ProfilerStop();
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, n_bedpp_reject, Rcpp::wrap(col_idx));
}

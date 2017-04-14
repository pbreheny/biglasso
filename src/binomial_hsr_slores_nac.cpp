
#include "utilities.h"

void Free_memo_bin_hsr_nac(double *s, double *w, double *a, double *r, int *e2, double *eta) {
  Free(s);
  Free(w);
  Free(a);
  Free(r);
  Free(e2);
  Free(eta);
}

void update_resid_eta(double *r, double *eta, XPtr<BigMatrix> xpMat, double shift, 
                      int *row_idx_, double center_, double scale_, int n, int j);

void update_zj(vector<double> &z, int *bedpp_reject, int *bedpp_reject_old,
               XPtr<BigMatrix> xpMat, int *row_idx,vector<int> &col_idx,
               NumericVector &center, NumericVector &scale, 
               double sumResid, double *r, double *m, int n, int p);

// dual function
double dual(vector<double>& theta, double lambda, double lambda_max, int n);

// Slores initialization
void slores_init(vector<double>& theta_lam, 
                 double *g_theta_lam_ptr, double *prod_deriv_theta_lam_ptr,
                 vector<double>& X_theta_lam_xi_pos,
                 vector<double>& prod_PX_Pxmax_xi_pos,
                 vector<double>& cutoff_xi_pos,
                 XPtr<BigMatrix> xMat, double *y, vector<double>& z, int xmax_idx,
                 int *row_idx, vector<int> &col_idx,
                 NumericVector &center, NumericVector &scale,
                 IntegerVector& ylab, int n_pos, int n, int p);

// Slores screening
void slores_screen(int *slores_reject, vector<double>& theta_lam, 
                   double g_theta_lam, double prod_deriv_theta_lam,
                   vector<double>& X_theta_lam_xi_pos,
                   vector<double>& prod_PX_Pxmax_xi_pos,
                   vector<double>& cutoff_xi_pos,
                   int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale, int xmax_idx,
                   IntegerVector& ylab, double lambda, 
                   double lambda_max, int n_pos, int n, int p);

// check rest set with slores screening (no active cycling)
int check_rest_set_bin_nac(int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                           int *row_idx, vector<int> &col_idx, NumericVector &center, 
                           NumericVector &scale, double *a, double lambda, 
                           double sumResid, double alpha, double *r, double *m, int n, int p) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
  
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e2[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        e2[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check rest set with slores screening (no active cycling)
int check_rest_set_hsr_slores_nac(int *e2, int *reject, vector<double> &z, 
                              XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                              NumericVector &center, NumericVector &scale, double *a,
                              double lambda, double sumResid, double alpha, double *r, 
                              double *m, int n, int p) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
  
  #pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (reject[j] == 0 && e2[j] == 0) { // set not rejected by slores but by hsr
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        e2[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// Coordinate descent for logistic models (no active set cycling)
RcppExport SEXP cdfit_binomial_hsr_slores_nac(SEXP X_, SEXP y_, SEXP n_pos_, SEXP ylab_, 
                                          SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                          SEXP lam_scale_, SEXP lambda_min_, 
                                          SEXP alpha_, SEXP user_, 
                                          SEXP eps_, SEXP max_iter_, SEXP multiplier_, 
                                          SEXP dfmax_, SEXP ncore_, SEXP warn_,
                                          SEXP safe_thresh_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int n_pos = INTEGER(n_pos_)[0];
  IntegerVector ylabel = Rcpp::as<IntegerVector>(ylab_); // label vector of {-1, 1}
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  double slores_thresh = REAL(safe_thresh_)[0]; // threshold for safe test
  int verbose = INTEGER(verbose_)[0];

  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L); // number of total rejections;
  IntegerVector n_slores_reject(L); // number of safe rejections;
  NumericVector beta0(L);
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
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
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
  double a0 = 0.0; //beta0 from previousiteration
  double *w = Calloc(n, double);
  double *s = Calloc(n, double); //y_i - pi_i
  double *eta = Calloc(n, double);
  // int *e1 = Calloc(p, int); //ever-active set
  int *e2 = Calloc(p, int); //strong set
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  
  double ybar = sum(y, n) / n;
  a0 = beta0[0] = log(ybar / (1-ybar));
  double nullDev = 0;
  double *r = Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i]*log(ybar) - (1-y[i])*log(1-ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev / n;
  double sumS = sum(s, n); // temp result sum of s
  double sumWResid = 0.0; // temp result: sum of w * r

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

  // Slores variables
  vector<double> theta_lam;
  double g_theta_lam = 0.0;
  double prod_deriv_theta_lam = 0.0;
  double *g_theta_lam_ptr = &g_theta_lam;
  double *prod_deriv_theta_lam_ptr = &prod_deriv_theta_lam;
  vector<double> X_theta_lam_xi_pos; 
  vector<double> prod_PX_Pxmax_xi_pos;
  vector<double> cutoff_xi_pos;
  int *slores_reject = Calloc(p, int);
  int *slores_reject_old = Calloc(p, int);
  for (int j = 0; j < p; j++) slores_reject_old[j] = 1;
  
  int slores; // if 0, don't perform Slores rule
  if (slores_thresh < 1) {
    slores = 1; // turn on slores
    theta_lam.resize(n);
    X_theta_lam_xi_pos.resize(p);
    prod_PX_Pxmax_xi_pos.resize(p);
    cutoff_xi_pos.resize(p);

    slores_init(theta_lam, g_theta_lam_ptr, prod_deriv_theta_lam_ptr, cutoff_xi_pos,
                X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, 
                xMat, y, z, xmax_idx, row_idx, col_idx, 
                center, scale, ylabel, n_pos, n, p);
  } else {
    slores = 0;
  }
  
  if (slores == 1 && user == 0) n_slores_reject[0] = p;

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
        Free(slores_reject);
        Free(slores_reject_old);
        Free_memo_bin_hsr_nac(s, w, a, r, e2, eta);
        return List::create(beta0, beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
      cutoff = 2*lambda[l] - lambda[l-1];
    } else {
      cutoff = 2*lambda[l] - lambda_max;
    }

    if (slores) {
      slores_screen(slores_reject, theta_lam, g_theta_lam, prod_deriv_theta_lam,
                    X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, cutoff_xi_pos,
                    row_idx, col_idx, center, scale, xmax_idx, ylabel, 
                    lambda[l], lambda_max, n_pos, n, p);
      n_slores_reject[l] = sum(slores_reject, p);
      
      // update z[j] for features which are rejected at previous lambda but accepted at current one.
      update_zj(z, slores_reject, slores_reject_old, xMat, row_idx, col_idx, center, scale, sumS, s, m, n, p);
      
      #pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        slores_reject_old[j] = slores_reject[j];
        // hsr screening
        // if (slores_reject[j] == 0 && (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]]))) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      n_slores_reject[l] = 0; 
      // hsr screening over all
      #pragma omp parallel for private(j) schedule(static) 
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
          
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              w[i] = pi * (1 - pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / w[i];
            if (y[i] == 1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1-pi);
            }
          }
          
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            Free(slores_reject);
            Free(slores_reject_old);
            Free_memo_bin_hsr_nac(s, w, a, r, e2, eta);
            return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, n_slores_reject, Rcpp::wrap(col_idx));
          }
          // Intercept
          xwr = crossprod(w, r, n, 0);
          xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          si = beta0[l] - a0;
          if (si != 0) {
            a0 = beta0[l];
            for (i = 0; i < n; i++) {
              r[i] -= si; //update r
              eta[i] += si; //update eta
            }
          }
          sumWResid = wsum(r, w, n); // update temp result: sum of w * r, used for computing xwr;
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e2[j]) {
              jj = col_idx[j];
              xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[jj], scale[jj], w, n, jj);
              v = wsqsum_bm(xMat, w, row_idx, center[jj], scale[jj], n, jj) / n;
              u = xwr/n + v * a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);

              shift = beta(j, l) - a[j];
              if (shift != 0) {
                // update change of objective function
                // update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + l1 * (fabs(beta(j, l)) - fabs(a[j]));
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

      }
      // Scan for violations in rest
      if (slores) {
        violations = check_rest_set_hsr_slores_nac(e2, slores_reject, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      } else {
        violations = check_rest_set_bin_nac(e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      }
      if (violations == 0) break;
      if (n_slores_reject[l] <= p * slores_thresh) {
        slores = 0; // turn off slores screening for next iteration if not efficient
      }
    }
  }
  Free(slores_reject);
  Free(slores_reject_old);
  Free_memo_bin_hsr_nac(s, w, a, r, e2, eta);
  return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, n_slores_reject, Rcpp::wrap(col_idx));
}


#include "utilities.h"


// standardize for multiresponse
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, 
                                  NumericMatrix &Y, int *row_idx, double lambda_min,
                                  double alpha, int n, int p, int m) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  double *sum_xy = Calloc(m, double);
  double *sum_y = Calloc(m, double);
  double zmax = 0.0, zj = 0.0;
  int i, j, k;
  
  for(k = 0; k < m; k++) {
    sum_y[k] = 0.0;
    for(i = 0; i < n; i++) {
      sum_y[k] += Y(k, i);
    }
  }
  
  for (j = 0; j < p; j++) {
    xCol = xAcc[j];
    for(k  = 0; k < m; k++) {
      sum_xy[k] = 0.0;
    }

    for (i = 0; i < n; i++) {
      center[j] += xCol[row_idx[i]];
      scale[j] += pow(xCol[row_idx[i]], 2);
      for(k  = 0; k < m; k++) {
        sum_xy[k] += xCol[row_idx[i]] * Y.at(k, i);
      }
    }
    
    center[j] = center[j] / n; //center
    scale[j] = sqrt(scale[j] / n - pow(center[j], 2)); //scale
    
    if (scale[j] > 1e-6) {
      col_idx.push_back(j);
      zj = 0;
      for(k = 0; k < m; k++) {
        zj += pow(sum_xy[k] - center[j] * sum_y[k], 2);
      }
      zj = sqrt(zj) / (scale[j] * n * sqrt(m)); //residual
      if (fabs(zj) > zmax) {
        zmax = fabs(zj);
        *xmax_ptr = j; // xmax_ptr is the index in the raw xMat, not index in col_idx!
      }
      z.push_back(zj);
    }
  }
  *p_keep_ptr = col_idx.size();
  *lambda_max_ptr = zmax / alpha;
  Free(sum_xy); Free(sum_y);
}

// Crossproduct xjTR
void crossprod_resid(double *xTR, XPtr<BigMatrix> xMat, double *R,
                     double *sumResid, int *row_idx,
                     double center, double scale, int n, int m, int j) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol = xAcc[j];
  double xi;
  int i, k;
  for(k = 0; k < m; k++) xTR[k] = 0.0;
  for (i = 0; i < n; i++) {
    xi = xCol[row_idx[i]];
    for(k = 0; k < m; k++) {
      xTR[k] += xi * R[i*m+k];
    }
  }
  for(k = 0; k < m; k++){
    xTR[k] = (xTR[k] - center * sumResid[k]) / scale;
  } 
}

// Update beta
void lasso(arma::field<arma::sp_mat> &beta, double *xTR, double z, double l1, double l2, int j, int l, int m) {
  if(z <= l1) {
    for(int k = 0; k < m; k++) {
      beta.at(k).at(j, l) = 0;
    }
  } else {
    for(int k = 0; k < m; k++) {
      beta.at(k).at(j, l) = sqrt(m) * xTR[k] * (1 - l1 / z) / (1 + l2);
    }
  }
}

// update residul matrix
void update_resid(XPtr<BigMatrix> xpMat, double *R, double *shift,
                  int *row_idx, double center, double scale, int n, int m, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  double xi;
  for (int i =0; i < n; i++) {
    xi = (xCol[row_idx[i]] - center) / scale;
    for(int k = 0; k < m; k++) {
      R[i*m+k] -= shift[k] * xi;
    }
  }
}

// check KKT conditions over features in the strong set
int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx, 
                     NumericVector &center, NumericVector &scale, double *a,
                     double lambda, double *sumResid, double alpha, 
                     double *R, double *mp, int n, int p, int m) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, *xTR, l1, l2, sum;
  int j, jj, violations = 0;
  
#pragma omp parallel for private(j, xTR, l1, l2, sum) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e1[j] == 0 && e2[j] == 1) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      z[j] = 0;
      sum = 0;
      xTR = Calloc(m, double);
      for(int k=0; k < m; k++) xTR[k] = 0;
      for (int i=0; i < n; i++) {
        for (int k=0; k < m; k++) {
          xTR[k] += xCol[row_idx[i]] * R[i*m+k];
        }
      }
      l1 = lambda * mp[jj] * alpha;
      l2 = lambda * mp[jj] * (1 - alpha);
      for(int k=0; k < m; k++){
        xTR[k] = (xTR[k] - center[jj] * sumResid[k]) / scale[jj];
        z[j] += pow(xTR[k], 2);
        sum += pow(xTR[k] - n * sqrt(m) * l2 * a[j * m + k], 2);
      } 
      z[j] = sqrt(z[j]) / (n * sqrt(m));
      if(sum > m * pow(l1 * n, 2)) {
        e1[j] = 1;
        violations++;
      }
      Free(xTR);
    }
  }
  return violations;
}

// check KKT conditions over features in the rest set
int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                   int *row_idx, vector<int> &col_idx, NumericVector &center,
                   NumericVector &scale, double *a, double lambda, double *sumResid,
                   double alpha, double *R, double *mp, int n, int p, int m) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, *xTR, l1, l2, sum;
  int j, jj, violations = 0;
#pragma omp parallel for private(j, xTR, l1, l2, sum) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e2[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      z[j] = 0;
      sum = 0;
      xTR = Calloc(m, double);
      for(int k=0; k < m; k++) xTR[k] = 0;
      for (int i=0; i < n; i++) {
        for (int k=0; k < m; k++) {
          xTR[k] += xCol[row_idx[i]] * R[i*m+k];
        }
      }
      l1 = lambda * mp[jj] * alpha;
      l2 = lambda * mp[jj] * (1 - alpha);
      for(int k=0; k < m; k++){
        xTR[k] = (xTR[k] - center[jj] * sumResid[k]) / scale[jj];
        z[j] += pow(xTR[k], 2);
        sum += pow(xTR[k] - n * sqrt(m) * l2 * a[j * m + k], 2);
      } 
      z[j] = sqrt(z[j]) / (n * sqrt(m));
      if(sum > m * pow(l1 * n, 2)) {
        e1[j] = e2[j] = 1;
        violations++;
      }
      Free(xTR);
    }
  }
  return violations;
}

// Coordinate descent for gaussian models with ssr
RcppExport SEXP cdfit_mgaussian_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                    SEXP lambda_, SEXP nlambda_, 
                                    SEXP lam_scale_, SEXP lambda_min_, 
                                    SEXP alpha_, SEXP user_, SEXP eps_, 
                                    SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                    SEXP ncore_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  NumericMatrix Y(y_); // m responses * n samples matrix
  int m = Y.nrow(); // 
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
  double *mp = REAL(multiplier_);
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
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, 
                               lambda_max_ptr, xmax_ptr, xMat, Y, row_idx,
                               lambda_min, alpha, n, p, m);
  
  p = p_keep;   // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::field<arma::sp_mat> beta(m);
  beta.for_each( [&](arma::sp_mat& beta_class) { beta_class.set_size(p, L); } );
  //arma::sp_mat beta = arma::sp_mat(m*p, L); // beta
  double *a = Calloc(m*p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  
  double l1, l2, cutoff;
  double* shift = Calloc(m, double);
  double max_update, update, thresh; // for convergence check
  int i, j, k, jj, l, violations, lstart;
  
  int *e1 = Calloc(p, int); // ever active set
  int *e2 = Calloc(p, int); // strong set
  double *R = Calloc(m*n, double); // residual matrix
  double *sumResid = Calloc(m, double);
  loss[0] = 0;
  for(k = 0; k < m; k++) sumResid[k] = 0;
  for(i = 0; i < n; i++) {
    for(k = 0; k < m; k++){
      R[i*m+k] = Y.at(k, i);
      sumResid[k] += Y.at(k, i);
      loss[0] += pow(Y.at(k, i), 2);
    } 
  }
  double *xTR = Calloc(m, double);
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
        if (a[j*m] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        Free(a); Free(e1); Free(e2); Free(xTR); Free(shift); Free(R); Free(sumResid);
        return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
      }
      // strong set
      cutoff = 2 * lambda[l] - lambda[l-1];
      for (j = 0; j < p; j++) {
        if (z[j] > (cutoff * alpha * mp[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      } 
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (j = 0; j < p; j++) {
        if (z[j] > (cutoff * alpha * mp[col_idx[j]])) {
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
              crossprod_resid(xTR, xMat, R, sumResid, row_idx, center[jj], scale[jj], n, m, jj);
              z[j] = 0;
              for(k = 0; k < m; k++) {
                xTR[k] = (xTR[k] / n + a[j * m + k]) / sqrt(m);
                z[j] += pow(xTR[k], 2);
              }
              z[j] = sqrt(z[j]);
              l1 = lambda[l] * mp[jj] * alpha;
              l2 = lambda[l] * mp[jj] * (1-alpha);
              lasso(beta, xTR, z[j], l1, l2, j, l, m);
              
              update = 0;
              for(k = 0; k < m; k++){
                shift[k] = beta.at(k).at(j, l) - a[j * m + k];
                update += pow(shift[k], 2);
              } 
              
              if (update !=0) {
                // compute objective update for checking convergence
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, R, shift, row_idx, center[jj], scale[jj], n, m, jj); // update R
                //update sum of residual
                for(k = 0; k < m; k++) sumResid[k] = 0;
                for(i = 0; i < n; i++) {
                  for(k = 0; k < m; k++) sumResid[k] += R[i*m+k];  
                }
                for(k = 0; k < m; k++) a[j * m + k] = beta.at(k).at(j, l);
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, R, mp, n, p, m); 
        if (violations==0) break;
      }
      
      // Scan for violations in rest set
      violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, R, mp, n, p, m);
      if (violations == 0) {
        loss[l] = 0;
        for(i = 0; i < n; i++) {
          for(k = 0; k < m; k++) loss[l] += pow(R[i*m+k], 2);  
        }
        break;
      }
    }
  }
  
  Free(a); Free(e1); Free(e2); Free(xTR); Free(shift); Free(R); Free(sumResid);
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
}